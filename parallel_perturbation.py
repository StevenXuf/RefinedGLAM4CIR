"""Partition text-only perturbation inference across the configured GPUs."""
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml

from generation import validate_perturbation
from run import experiment_signature, load_manifest
from utils import file_hash, get_config, read_json, save_json, setup_logging


def merge_records(original, partitions, cfg):
    expected = {record['query_id']: record for record in original}
    merged = {}
    for partition in partitions:
        for record in partition:
            query_id = record['query_id']
            if query_id not in expected or query_id in merged:
                raise ValueError('Unexpected or duplicate partition query ID: ' + query_id)
            baseline = expected[query_id]
            for key in ['reference_id', 'target_ids', 'modifier']:
                if record[key] != baseline[key]:
                    raise ValueError(f'Partition changed {query_id} {key}')
            if set(record['perturbations']) != set(cfg['perturbation']['types']):
                raise ValueError('Incomplete partition: ' + query_id)
            for kind, text in record['perturbations'].items():
                validate_perturbation({'modifier': text}, cfg, baseline['modifier'], kind)
            item = copy.deepcopy(baseline)
            item['perturbations'] = record['perturbations']
            item['perturbation_reviews'].update(record['perturbation_reviews'])
            merged[query_id] = item
    if set(merged) != set(expected):
        raise ValueError('Partition query coverage is incomplete')
    return [merged[record['query_id']] for record in original]


def main(config_path, split):
    cfg = get_config(config_path, split=split)
    manifest = load_manifest(cfg)
    base = Path(cfg['paths']['output_dir'])
    original = read_json(base / 'modifiers.json')
    if [r['query_id'] for r in original] != [q['query_id'] for q in manifest['queries']]:
        raise ValueError('Complete clean generation before parallel perturbation')
    logger = setup_logging(base / 'parallel_perturbation.log')
    gpus = list(dict.fromkeys(cfg['runtime']['gpu_ids'].values()))
    work = base / 'perturbation_partitions'
    work.mkdir(exist_ok=True)
    save_json(work / 'original_modifiers.json', original)
    raw = yaml.safe_load(Path(config_path).read_text())
    jobs, streams, directories, configs = [], [], [], []
    try:
        for index, gpu in enumerate(gpus):
            part_cfg = copy.deepcopy(raw)
            part_cfg['split'] = split
            part_cfg['runtime']['gpu_ids'][split] = gpu
            part_cfg['perturbation']['quality_review'] = False  # audit the global seeded sample after merging
            part_cfg['paths'] = dict(cfg['paths'])
            part_cfg['paths']['output_dir'] = str(work / f'gpu_{gpu}')
            path = work / f'gpu_{gpu}.yaml'
            path.write_text(yaml.safe_dump(part_cfg, sort_keys=False))
            resolved = get_config(path, split=split)
            directory = Path(resolved['paths']['output_dir'])
            directory.mkdir(parents=True, exist_ok=True)
            records = original[index::len(gpus)]
            part_manifest = copy.deepcopy(manifest)
            part_manifest['queries'] = manifest['queries'][index::len(gpus)]
            part_manifest['config'] = resolved
            part_manifest['signature'] = experiment_signature(resolved)
            part_manifest['query_scope'] = 'execution partition only; no metrics computed here'
            # Each partition retains the entire available gallery, without subsampling.
            if not (directory / 'modifiers.json').exists():
                save_json(directory / 'manifest.json', part_manifest)
                save_json(directory / 'modifiers.json', records)
            save_json(directory / 'progress.json', {'completed': sum(len(r['perturbations']) for r in records), 'total': len(records) * len(cfg['perturbation']['types'])})
            stream = (work / f'gpu_{gpu}.console.log').open('a')
            streams.append(stream)
            command = [sys.executable, str(Path(__file__).with_name('run.py')), '--config', str(path), '--split', split, '--stage', 'perturb']
            jobs.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
            directories.append(directory)
            configs.append({'gpu': gpu, 'config': str(path), 'signature': part_manifest['signature'], 'queries': len(records)})
            logger.info('Started GPU %s, queries=%d pid=%d', gpu, len(records), jobs[-1].pid)
        while any(job.poll() is None for job in jobs):
            completed = sum(len(record['perturbations']) for directory in directories for record in read_json(directory / 'modifiers.json'))
            save_json(base / 'progress.json', {'split': split, 'stage': 'parallel perturbations', 'completed': completed, 'total': len(original) * len(cfg['perturbation']['types'])})
            logger.info('Parallel perturbations %d/%d', completed, len(original) * len(cfg['perturbation']['types']))
            time.sleep(20)
        if any(job.returncode for job in jobs):
            raise RuntimeError('A partition failed; successful checkpoints remain under ' + str(work))
        merged = merge_records(original, [read_json(directory / 'modifiers.json') for directory in directories], cfg)
        save_json(base / 'modifiers.json', merged)
        manifest['perturbation_execution'] = {'method': 'independent query partitions, merged before global evaluation', 'executor_sha256': file_hash(__file__), 'partitions': configs, 'initial_records': str(work / 'original_modifiers.json'), 'global_quality_audit': 'Run the normal perturb stage after merging'}
        save_json(base / 'manifest.json', manifest)
        logger.info('Merged all %d queries in original CSV order. Global audit/evaluation can resume.', len(merged))
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--split', choices=['test_id', 'test_ood'], required=True)
    args = parser.parse_args()
    main(Path(args.config).resolve(), args.split)
