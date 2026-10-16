"""Experiment orchestration; every stage reads/writes explicit, resumable artifacts."""
import gc
import importlib.metadata
import platform
import random
import time
import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import prompts
from feature_extractor import FeatureExtractor
from generation import (TextGenerator, typo_plan, word_preserving_request,
                        validate_modifier, validate_perturbation)
from loading_data import GLAMGalleryDataset, GLAMQueryDataset, collate_samples, load_image, search_image_path
from metrics_computation import compute_metrics, metric_deltas, rank_queries
from utils import (file_hash, fingerprint, get_config, get_device, output_path,
                   parse_args, read_json, save_json, seed_everything, setup_logging)


def experiment_signature(cfg):
    return fingerprint({'config': cfg, 'cir': file_hash(cfg['paths']['cir_file']),
                        'gallery': file_hash(cfg['paths']['db_processed_name']),
                        'sources': {name: file_hash(Path(__file__).parent / name) for name in
                                    ['run.py', 'loading_data.py', 'generation.py', 'prompts.py',
                                     'feature_extractor.py', 'metrics_computation.py', 'utils.py', 'results.py']}})


def load_manifest(cfg):
    manifest = read_json(output_path(cfg, 'manifest.json'))
    if manifest['signature'] != experiment_signature(cfg):
        raise ValueError('Config/data changed: choose a new paths.output_dir to avoid mixing runs')
    restored = search_image_path(manifest.get('missing_gallery_ids', []), cfg['image_storage_folders'], cfg['paths']['glam_root'])
    if restored:
        raise ValueError('Image availability changed: choose a new paths.output_dir to include restored gallery images')
    return manifest


def prepare(cfg, logger):
    path = output_path(cfg, 'manifest.json')
    if path.exists():
        manifest = load_manifest(cfg)
        logger.info('Reusing manifest: %d queries, %d gallery images', len(manifest['queries']), len(manifest['gallery']))
        return manifest
    queries = GLAMQueryDataset(cfg, logger).records
    gallery = GLAMGalleryDataset(cfg, logger, queries)
    for query in queries:
        query['conditioning_target_ids'] = list(query['target_ids'])
        query['conditioning_target_paths'] = list(query['target_paths'])
    manifest = {'signature': experiment_signature(cfg), 'config': cfg, 'queries': queries,
                'gallery': gallery.records, 'full_gallery_size': gallery.full_size,
                'declared_gallery_size': gallery.declared_size, 'missing_gallery_ids': gallery.missing_ids,
                'scope': 'full',
                'query_source': 'cir_db.csv id1/id2/query/partition only',
                'gallery_source': 'db_processed.txt rows matching the requested test split',
                'conditioning_note': 'Target-aware modifier construction; not a target-blind benchmark.',
                'source_hashes': {name: file_hash(Path(__file__).parent / name)
                                  for name in ['prompts.py', 'generation.py', 'loading_data.py',
                                               'feature_extractor.py', 'metrics_computation.py', 'run.py']}}
    save_json(path, manifest)
    save_json(output_path(cfg, 'config.resolved.json'), cfg)
    logger.info('Prepared manifest at %s', path)
    return manifest


def bool_review(keys):
    def validate(result):
        if any(type(result.get(key)) is not bool for key in keys) or not isinstance(result.get('reason'), str):
            raise ValueError('Review requires boolean checks and a string reason')
        return result
    return validate


def string_fields(keys):
    def validate(result):
        if any(not isinstance(result.get(key), str) for key in keys):
            raise ValueError('Expected string fields: ' + ', '.join(keys))
        return result
    return validate


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def progress(cfg, stage, completed, total):
    save_json(output_path(cfg, 'progress.json'),
              {'split': cfg['split'], 'stage': stage, 'completed': completed, 'total': total})


def audit_ids(records, cfg, section):
    if not cfg[section]['quality_review']:
        return set()
    count = cfg[section].get('quality_review_samples')
    ids = [r['query_id'] for r in records]
    return set(ids if count is None else random.Random(cfg['seed']).sample(ids, min(count, len(ids))))


def checkpoint_batch(cfg, records_by_id, manifest):
    records = [records_by_id[q['query_id']] for q in manifest['queries'] if q['query_id'] in records_by_id]
    save_json(output_path(cfg, 'modifiers.json'), records)
    return records


def fail_batch(errors, logger):
    if errors:
        logger.error('Batch checkpointed; %d invalid items remain. First error: %s', len(errors), errors[0])
        raise ValueError(f'{len(errors)} LLM items failed validation: {errors[0]}')


def generate(cfg, logger):
    manifest = load_manifest(cfg)
    modifier_path = output_path(cfg, 'modifiers.json')
    records = read_json(modifier_path) if modifier_path.exists() else []
    expected = {q['query_id'] for q in manifest['queries']}
    records_by_id = {r['query_id']: r for r in records}
    if set(records_by_id) - expected:
        raise ValueError('Generation checkpoint contains queries outside the CSV manifest')
    pending = [q for q in manifest['queries'] if q['query_id'] not in records_by_id]
    review_ids = audit_ids(manifest['queries'], cfg, 'generation')
    if not pending and all('quality_review' in records_by_id[i] for i in review_ids):
        logger.info('Clean generation already complete')
        quality_report(cfg, records, logger)
        return
    generator = TextGenerator(cfg, logger)
    observations_path = output_path(cfg, 'observations.json')
    observations = read_json(observations_path) if observations_path.exists() else {}
    # Image captions are unconditional and reusable; only IDs explicitly present in CSV pairs are requested.
    image_paths = {}
    for query in manifest['queries']:
        image_paths[query['reference_id']] = query['reference_path']
        image_paths.update(zip(query['target_ids'], query['target_paths']))
    missing = [(name, path) for name, path in sorted(image_paths.items()) if name not in observations]
    for batch in batches(missing, cfg['generation']['image_batch_size']):
        tasks = [{'system': prompts.TARGET_DESCRIPTION_SYSTEM, 'prompt': 'Describe this image in one sentence.',
                  'images': [load_image(path)], 'image_labels': ['Image']} for _, path in batch]
        outputs = generator.request_json_many(tasks, [string_fields(['description'])] * len(tasks),
                                             attempts=cfg['generation']['max_attempts'])
        errors = []
        for (name, path), result in zip(batch, outputs):
            if isinstance(result, Exception):
                errors.append(f'{name}: {result}')
            else:
                observations[name] = {'path': path, 'description': result['description']}
        save_json(observations_path, observations)
        progress(cfg, 'image observations', len(observations), len(image_paths))
        logger.info('CSV image observations %d/%d', len(observations), len(image_paths))
        fail_batch(errors, logger)
    def validate_synthesis(result):
        if any(not isinstance(result.get(key), (str, list)) for key in ['shared_traits', 'limitations']):
            raise ValueError('Synthesis needs shared_traits and limitations')
        result['modifier'] = validate_modifier(result, cfg)
        return result
    for batch in batches(pending, cfg['generation']['batch_size']):
        tasks = []
        for query in batch:
            request = 'BEFORE reference: ' + observations[query['reference_id']]['description']
            request += '\nAFTER alternatives: ' + json.dumps([observations[name]['description'] for name in query['target_ids']])
            if cfg['generation']['use_original_hint']:
                request += '\nNoisy original CSV modifier: ' + query['original_modifier']
            request += f"\nThe modifier must have {cfg['generation']['min_words']}-{cfg['generation']['max_words']} words."
            tasks.append({'system': prompts.SYNTHESIS_SYSTEM, 'prompt': request})
        outputs = generator.request_json_many(tasks, [validate_synthesis] * len(tasks),
                                             attempts=cfg['generation']['max_attempts'])
        errors = []
        for query, result in zip(batch, outputs):
            if isinstance(result, Exception):
                errors.append(f"{query['query_id']}: {result}")
                continue
            records_by_id[query['query_id']] = dict(query, modifier=result['modifier'],
                reference_observation=observations[query['reference_id']]['description'],
                target_observations=[observations[name]['description'] for name in query['target_ids']],
                synthesis=result, perturbations={}, perturbation_reviews={})
        records = checkpoint_batch(cfg, records_by_id, manifest)
        progress(cfg, 'clean modifiers', len(records), len(manifest['queries']))
        logger.info('Clean modifiers %d/%d', len(records), len(manifest['queries']))
        fail_batch(errors, logger)
    for record in records:
        if record['query_id'] not in review_ids or 'quality_review' in record:
            continue
        paths = record['target_paths'][:cfg['generation']['review_target_images']]
        images = [load_image(record['reference_path'])] + [load_image(p) for p in paths]
        record['quality_review'] = generator.request_json(prompts.REVIEW_SYSTEM,
            'Modifier: ' + record['modifier'] + '\nTarget observations: ' + json.dumps(record['target_observations']),
            images, attempts=cfg['generation']['max_attempts'],
            validator=bool_review(['grounded', 'direction_correct', 'coherent', 'shared_across_targets']))
        record['reviewed_target_ids'] = record['target_ids'][:len(paths)]
        save_json(modifier_path, records)
    del generator
    release_model(None)
    quality_report(cfg, records, logger)


def perturb(cfg, logger):
    manifest = load_manifest(cfg)
    path = output_path(cfg, 'modifiers.json')
    records = read_json(path)
    if [r['query_id'] for r in records] != [q['query_id'] for q in manifest['queries']]:
        raise ValueError('Complete CSV-aligned generation before perturbation')
    review_ids = audit_ids(records, cfg, 'perturbation')
    kinds = cfg['perturbation']['types']
    complete = all(set(kinds) <= set(r['perturbations']) for r in records)
    audited = all(set(kinds) <= set(r['perturbation_reviews']) for r in records if r['query_id'] in review_ids)
    if complete and audited:
        logger.info('Perturbation already complete')
        return
    generator = TextGenerator(cfg, logger)
    total = len(records) * len(kinds)
    for kind, instruction in kinds.items():
        pending = [r for r in records if kind not in r['perturbations']]
        for batch in batches(pending, cfg['generation']['batch_size']):
            tasks, validators, plans = [], [], []
            for record in batch:
                start = len(tasks)
                if kind == 'typos':
                    jobs, checks, finish = typo_plan(cfg, record['modifier'], record['query_id'])
                    tasks.extend(jobs)
                    validators.extend(checks)
                elif cfg['perturbation']['preserve_content_words']:
                    task, validator = word_preserving_request(cfg, record['modifier'], kind, instruction)
                    tasks.append(task)
                    validators.append(validator)
                    def finish(values):
                        return values[0]
                else:
                    tasks.append({'system': prompts.PERTURB_SYSTEM,
                                  'prompt': prompts.perturbation_prompt(record['modifier'], instruction)})
                    validators.append(lambda result, original=record['modifier']: validate_perturbation(result, cfg, original, kind))
                    def finish(values):
                        return values[0]
                plans.append((record, start, len(tasks), finish))
            outputs = generator.request_json_many(tasks, validators, attempts=cfg['perturbation']['max_attempts'],
                                                 decoding=cfg['perturbation']['decoding'])
            errors = []
            for record, start, end, finish in plans:
                values = outputs[start:end]
                if any(isinstance(value, Exception) for value in values):
                    if kind != 'typos' and cfg['perturbation']['preserve_content_words']:
                        # Keep a complete complex instruction intact if finer rephrasing fails.
                        try:
                            text = generator.choose_valid_rewrite(record['modifier'], kind)
                        except ValueError as error:
                            errors.append(f"{record['query_id']} {kind}: {error}")
                            continue
                    else:
                        errors.append(f"{record['query_id']} {kind}: {values}")
                        continue
                else:
                    text = finish(values)
                record['perturbations'][kind] = text
            save_json(path, records)
            completed = sum(len(r['perturbations']) for r in records)
            progress(cfg, 'perturbations', completed, total)
            logger.info('Perturbations %d/%d', completed, total)
            fail_batch(errors, logger)
    for record in records:
        if record['query_id'] not in review_ids:
            continue
        for kind, text in record['perturbations'].items():
            if kind in record['perturbation_reviews']:
                continue
            review = generator.request_json(prompts.PERTURB_REVIEW_SYSTEM,
                f'Original: {record["modifier"]}\nPerturbed: {text}',
                attempts=cfg['perturbation']['max_attempts'], validator=bool_review(['meaning_preserved']))
            record['perturbation_reviews'][kind] = review
            # Audits are diagnostics; retain checked text and flag any semantic concern transparently.
            if not review['meaning_preserved']:
                logger.warning('Semantic audit flagged %s %s: %s', record['query_id'], kind, review['reason'])
            save_json(path, records)
    del generator
    release_model(None)
    quality_report(cfg, records, logger)


def quality_report(cfg, records, logger):
    reviewed = [r for r in records if 'quality_review' in r]
    checks = ['grounded', 'direction_correct', 'coherent', 'shared_across_targets']
    perturbation_flags = [{'query_id': r['query_id'], 'kind': kind, 'reason': review['reason']}
                          for r in records for kind, review in r['perturbation_reviews'].items()
                          if not review['meaning_preserved']]
    report = {'kind': 'sampled model-assisted diagnostic audit', 'queries': len(records), 'reviewed': len(reviewed),
              'pass_counts': {k: sum(r['quality_review'][k] for r in reviewed) for k in checks},
              'flagged_query_ids': [r['query_id'] for r in reviewed if not all(r['quality_review'][k] for k in checks)],
              'perturbation_flags': perturbation_flags,
              'unique_clean_modifiers': len({r['modifier'] for r in records}),
              'mean_clean_word_count': sum(len(r['modifier'].split()) for r in records) / len(records),
              'generation_scope': 'Every target from the same cir_db.csv row; no gallery-derived pairings',
              'review_scope': 'Configured query audit sample and first review_target_images targets; no human accuracy claim'}
    save_json(output_path(cfg, 'quality_report.json'), report)
    logger.info('Generation quality: %s', report)

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, records, path_key):
        self.records, self.path_key = records, path_key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {'pil_image': load_image(self.records[index][self.path_key])}


def encode_images(records, path_key, extractor, cfg, logger):
    loader = DataLoader(PathDataset(records, path_key), batch_size=cfg['retrieval']['batch_size'],
                        shuffle=False, num_workers=cfg['runtime']['num_workers'], collate_fn=collate_samples)
    features = []
    for index, batch in enumerate(loader):
        features.append(extractor.get_image_features(batch['pil_image']).cpu())
        if index % 10 == 0:
            logger.info('Image features: %d/%d batches', index + 1, len(loader))
    return torch.cat(features)


def retrieve(cfg, logger, clean_only=False):
    manifest = load_manifest(cfg)
    records = read_json(output_path(cfg, 'modifiers.json'))
    if [r['query_id'] for r in records] != [q['query_id'] for q in manifest['queries']]:
        raise ValueError('Generation checkpoint incomplete or misaligned')
    conditions = ['clean'] if clean_only else ['clean'] + list(cfg['perturbation']['types'])
    for condition in conditions[1:]:
        if not all(condition in r['perturbations'] for r in records):
            raise ValueError(f'Incomplete perturbation: {condition}')
    extractor = FeatureExtractor(cfg, logger)
    cache_path = output_path(cfg, 'image_features.pt')
    identity = fingerprint({'manifest': manifest['signature'], 'retrieval': cfg['retrieval'],
                            'source': file_hash(Path(__file__).parent / 'feature_extractor.py'),
                            'images': [(p, os.stat(p).st_size, os.stat(p).st_mtime_ns)
                                       for p in [g['path'] for g in manifest['gallery']] +
                                                [r['reference_path'] for r in records]]})
    cache = torch.load(cache_path, map_location='cpu', weights_only=True) if cache_path.exists() else None
    if cache is None or cache.get('identity') != identity:
        cache = {'identity': identity,
                 'gallery': encode_images(manifest['gallery'], 'path', extractor, cfg, logger),
                 'reference': encode_images(records, 'reference_path', extractor, cfg, logger)}
        tmp = cache_path.with_suffix('.tmp')
        torch.save(cache, tmp)
        tmp.replace(cache_path)
    else:
        logger.info('Reusing cached gallery/reference image features')
    gallery_features = cache['gallery'].to(extractor.device)
    reference_features = cache['reference'].to(extractor.device)
    gallery_ids = [g['image_id'] for g in manifest['gallery']]
    retrieval, metrics, per_query = {}, {}, {}
    batch_size = cfg['evaluation']['query_batch_size']
    for condition in conditions:
        rankings, details = [], []
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [r['modifier'] if condition == 'clean' else r['perturbations'][condition] for r in batch]
            query_features = extractor.compose(reference_features[start:start + len(batch)], texts)
            ranked, scores = rank_queries(query_features, gallery_features, gallery_ids, batch,
                                         cfg['evaluation']['exclude_reference'])
            rankings.extend(ranked)
            gallery_index = {name: i for i, name in enumerate(gallery_ids)}
            for row, (record, ranking) in enumerate(zip(batch, ranked)):
                details.append({'query_id': record['query_id'], 'modifier': texts[row],
                    'target_ranks': {name: ranking.index(name) + 1 for name in record['target_ids']},
                    'top_results': [{'image_id': name, 'score': float(scores[row, gallery_index[name]])}
                                    for name in ranking[:max(cfg['visualization']['top_k'], max(cfg['evaluation']['top_ks']))]]})
        metrics[condition], per_query[condition] = compute_metrics(rankings, records, cfg['evaluation']['top_ks'])
        retrieval[condition] = details
        logger.info('METRICS condition=%s scope=%s queries=%d gallery=%d %s', condition, manifest['scope'],
                    len(records), len(gallery_ids), metrics[condition])
    del extractor
    release_model(None)
    deltas = {kind: metric_deltas(metrics['clean'], metrics[kind]) for kind in conditions[1:]}
    for kind, delta in deltas.items():
        logger.info('DELTA perturbed-minus-clean percentage points %s: %s', kind, delta)
    save_json(output_path(cfg, 'retrieval_clean.json' if clean_only else 'retrieval.json'), retrieval)
    save_json(output_path(cfg, 'metrics_clean.json' if clean_only else 'metrics.json'),
              {'scope': manifest['scope'], 'queries': len(records), 'gallery_size': len(gallery_ids),
               'declared_gallery_size': manifest['declared_gallery_size'],
               'missing_gallery_count': len(manifest['missing_gallery_ids']),
               'gallery_protocol': 'All available test-partition images; no sampling',
               'units': 'percent; deltas in percentage points', 'metrics': metrics,
               'delta_vs_clean': deltas, 'per_query': per_query,
               'protocol': 'Target-conditioned modifiers; fixed gallery and reference+text fusion.'})
    # CSV is convenient for plotting condition/metric changes outside Python.
    if not clean_only:
        import csv
        with output_path(cfg, 'metrics.csv').open('w') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['condition', 'metric', 'value_percent', 'delta_pp_vs_clean'])
            for condition, values in metrics.items():
                for metric, value in values.items():
                    writer.writerow([condition, metric, value, deltas.get(condition, {}).get(metric, 0)])


def main():
    args = parse_args(['both', 'all', 'prepare', 'generate', 'retrieve-clean', 'perturb', 'evaluate', 'visualize'])
    if args.stage == 'both':
        from results import run_both
        run_both(args.config)
        return
    cfg = get_config(args.config, split=args.split)
    logger = setup_logging(output_path(cfg, 'run.log'))
    start = time.monotonic()
    seed_everything(cfg)
    device = torch.device('cpu') if args.stage in ['prepare', 'visualize'] else get_device(cfg)
    versions = {}
    for package in ['torch', 'torchvision', 'transformers', 'open_clip_torch', 'pandas', 'Pillow']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    logger.info('Stage=%s device=%s config=%s', args.stage, device, args.config)
    save_json(output_path(cfg, 'environment.json'), {'python': platform.python_version(), 'packages': versions,
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
    try:
        if args.stage in ['all', 'prepare']:
            prepare(cfg, logger)
        if args.stage in ['all', 'generate']:
            generate(cfg, logger)
        if args.stage in ['all', 'retrieve-clean']:
            retrieve(cfg, logger, clean_only=True)
        if args.stage in ['all', 'perturb']:
            perturb(cfg, logger)
        if args.stage in ['all', 'evaluate']:
            retrieve(cfg, logger)
        if args.stage in ['all', 'visualize']:
            from figure import visualize
            visualize(cfg, logger)
        progress(cfg, 'complete ' + args.stage, 1, 1)
        if args.stage in ['all', 'evaluate']:
            from results import print_result_table
            print_result_table([read_json(output_path(cfg, 'metrics.json')) | {'split': cfg['split']}], logger)
        logger.info('Completed stage=%s elapsed=%.2fs', args.stage, time.monotonic() - start)
    except Exception:
        logger.exception('Experiment failed; completed artifacts are preserved for resume')
        raise


if __name__ == '__main__':
    main()
