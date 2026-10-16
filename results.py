"""Run both inference-only test partitions and print/save their full-gallery tables."""
import csv
import subprocess
import sys
import time
from pathlib import Path

from utils import get_config, read_json, save_json, setup_logging

COLUMNS = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@50', 'AVG', 'mAP', 'MRR']


def table_lines(results):
    headers = ['Split', 'Queries', 'Gallery', 'Condition'] + COLUMNS
    lines = ['| ' + ' | '.join(headers) + ' |', '|---|---:|---:|---|' + '---:|' * len(COLUMNS)]
    for result in results:
        if result['scope'] != 'full':
            raise ValueError('Refusing to present partial-gallery results')
        for condition, metrics in result['metrics'].items():
            label = 'unperturbed' if condition == 'clean' else condition
            row = [result['split'], str(result['queries']), str(result['gallery_size']), label]
            row += [f'{metrics[column]:.2f}' for column in COLUMNS]
            lines.append('| ' + ' | '.join(row) + ' |')
    return lines


def print_result_table(results, logger=None):
    text = '\n'.join(table_lines(results))
    if logger:
        logger.info('All available test-gallery results (percent):\n%s', text)
    else:
        print(text, flush=True)
    return text


def collect_results(config_path, destination=None):
    base_cfg = get_config(config_path)
    results = []
    for split in base_cfg['evaluation']['splits']:
        cfg = get_config(config_path, split=split)
        directory = Path(cfg['paths']['output_dir'])
        result = read_json(directory / 'metrics.json')
        manifest = read_json(directory / 'manifest.json')
        if result['gallery_size'] != manifest['full_gallery_size'] or result['queries'] != len(manifest['queries']):
            raise ValueError(f'{split}: metrics do not cover the entire frozen query/gallery manifest')
        result['split'] = split
        results.append(result)
    destination = Path(destination or Path(base_cfg['paths']['output_dir']).parent)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / 'results.json', results)
    with (destination / 'results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['split', 'queries', 'gallery', 'condition'] + COLUMNS)
        for result in results:
            for kind, metrics in result['metrics'].items():
                writer.writerow([result['split'], result['queries'], result['gallery_size'], kind] +
                                [metrics[key] for key in COLUMNS])
    text = '# Full GLAM test inference\n\nAll values are percentages. Each condition uses every available image from the corresponding test gallery, without sampling.\n\n'
    text += '\n'.join(table_lines(results))
    text += '\n\nGallery availability:\n\n'
    for result in results:
        declared = result.get('declared_gallery_size', result['gallery_size'])
        missing = result.get('missing_gallery_count', 0)
        text += f"- {result['split']}: {result['gallery_size']:,} available / {declared:,} declared; {missing} missing.\n"
    text += ('\n\nPair deduplication follows the original loader configuration: test_id has 2,559 raw CSV rows '
             'and test_ood has 88. Query counts in the table are after the configured deduplication and missing-image checks. '
             'No target-count filtering or query subsampling is used in these full runs.\n\n'
             'Recall is macro multi-positive recall. AVG is the mean of Recall@1/5/10/50. '
             'Clean and perturbed queries use identical reference-image/text fusion, gallery, ground truth, and reference exclusions. '
             'Modifiers are target-conditioned from the CSV pairs. The LLM quality audit is diagnostic and sampled; it does not certify caption accuracy.\n')
    (destination / 'results.md').write_text(text)
    return results


def run_both(config_path):
    cfg = get_config(config_path)
    splits = cfg['evaluation']['splits']
    if len(splits) != 2 or set(splits) != {'test_id', 'test_ood'}:
        raise ValueError('evaluation.splits must contain exactly test_ood and test_id')
    if cfg['subset'].get('query_limit') is not None:
        raise ValueError('Both-split full evaluation requires subset.query_limit: null')
    base = Path(cfg['paths']['output_dir']).parent
    logger = setup_logging(base / 'both.log')
    jobs, streams = {}, []
    try:
        for split in splits:
            console_path = base / f'{split}.console.log'
            stream = console_path.open('a')
            streams.append(stream)
            command = [sys.executable, str(Path(__file__).with_name('run.py')), '--config', str(Path(config_path).resolve()),
                       '--stage', 'all', '--split', split]
            jobs[split] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            logger.info('Started %s pid=%d GPU=%s console=%s', split, jobs[split].pid,
                        cfg['runtime']['gpu_ids'][split], console_path)
        while any(job.poll() is None for job in jobs.values()):
            for split, job in jobs.items():
                progress_path = base / split / 'progress.json'
                state = read_json(progress_path) if progress_path.exists() else {'stage': 'starting'}
                logger.info('%s status=%s progress=%s', split, 'running' if job.poll() is None else job.returncode, state)
            time.sleep(30)
        failures = {split: job.returncode for split, job in jobs.items() if job.returncode}
        if failures:
            raise RuntimeError(f'Incomplete full inference: {failures}; see split console logs and resume checkpoints')
        results = collect_results(config_path)
        print_result_table(results, logger)
        logger.info('Both full test runs complete. Tables: %s', base / 'results.md')
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    args = parser.parse_args()
    print_result_table(collect_results(args.config))
