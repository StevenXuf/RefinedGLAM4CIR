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
from generation import (TextGenerator, generate_typos, generate_word_preserving,
                        validate_modifier, validate_perturbation)
from loading_data import GLAMGalleryDataset, GLAMQueryDataset, collate_samples, load_image
from metrics_computation import compute_metrics, metric_deltas, rank_queries
from utils import (file_hash, fingerprint, get_config, get_device, output_path,
                   parse_args, read_json, save_json, seed_everything, setup_logging)


def experiment_signature(cfg):
    return fingerprint({'config': cfg, 'cir': file_hash(cfg['paths']['cir_file']),
                        'gallery': file_hash(cfg['paths']['db_processed_name']),
                        'sources': {name: file_hash(Path(__file__).parent / name) for name in
                                    ['run.py', 'loading_data.py', 'generation.py', 'prompts.py',
                                     'feature_extractor.py', 'metrics_computation.py', 'utils.py']}})


def load_manifest(cfg):
    manifest = read_json(output_path(cfg, 'manifest.json'))
    if manifest['signature'] != experiment_signature(cfg):
        raise ValueError('Config/data changed: choose a new paths.output_dir to avoid mixing runs')
    return manifest


def prepare(cfg, logger):
    path = output_path(cfg, 'manifest.json')
    if path.exists():
        manifest = load_manifest(cfg)
        logger.info('Reusing manifest: %d queries, %d gallery images', len(manifest['queries']), len(manifest['gallery']))
        return manifest
    queries = GLAMQueryDataset(cfg, logger).records
    gallery = GLAMGalleryDataset(cfg, logger, queries)
    by_id = {g['image_id']: g['path'] for g in gallery.records}
    for query in queries:
        indices = list(range(len(query['target_ids'])))
        rng = random.Random(f"{cfg['seed']}:{query['query_id']}")
        selected = sorted(rng.sample(indices, min(len(indices), cfg['generation']['max_target_images'])))
        query['conditioning_target_ids'] = [query['target_ids'][i] for i in selected]
        # Resolve conditioning images through the same gallery records used for retrieval.
        query['conditioning_target_paths'] = [by_id[name] for name in query['conditioning_target_ids']]
    manifest = {'signature': experiment_signature(cfg), 'config': cfg, 'queries': queries,
                'gallery': gallery.records, 'full_gallery_size': gallery.full_size,
                'scope': 'full' if len(gallery) == gallery.full_size else 'subset',
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


def generate(cfg, logger):
    manifest = load_manifest(cfg)
    path = output_path(cfg, 'modifiers.json')
    records = read_json(path) if path.exists() else []
    expected = [q['query_id'] for q in manifest['queries']]
    if [q['query_id'] for q in records] != expected[:len(records)]:
        raise ValueError('Generation checkpoint does not match manifest query order')
    if len(records) == len(expected):
        logger.info('Clean generation already complete')
        return
    generator = TextGenerator(cfg, logger)
    observations_path = output_path(cfg, 'observations.json')
    observations = read_json(observations_path) if observations_path.exists() else {}
    for query in manifest['queries'][len(records):]:
        images = [load_image(query['reference_path'])] + [load_image(p) for p in query['conditioning_target_paths']]
        hint = query['original_modifier'] if cfg['generation']['use_original_hint'] else ''
        request = prompts.refinement_prompt(hint, len(images) - 1)
        descriptions = []
        reference_description = None
        synthesis = None
        if cfg['generation']['describe_targets_first']:
            observed = observations.setdefault(query['query_id'], {'reference': None, 'targets': {}})
            if observed['reference'] is None:
                observed['reference'] = generator.request_json(prompts.TARGET_DESCRIPTION_SYSTEM,
                    'Describe this reference image in one sentence.', images[:1], image_labels=['Reference image'],
                    attempts=cfg['generation']['max_attempts'], validator=string_fields(['description']))['description']
                save_json(observations_path, observations)
            reference_description = observed['reference']
            for index, image in enumerate(images[1:], 1):
                target_id = query['conditioning_target_ids'][index - 1]
                if target_id not in observed['targets']:
                    observed['targets'][target_id] = generator.request_json(prompts.TARGET_DESCRIPTION_SYSTEM,
                        'Describe only this target image.', [image], image_labels=['ONE target alternative'],
                        attempts=cfg['generation']['max_attempts'],
                        validator=string_fields(['description']))['description']
                    save_json(observations_path, observations)
                description = observed['targets'][target_id]
                descriptions.append(description)
            request = 'BEFORE reference: ' + reference_description + '\nAFTER alternatives: ' + json.dumps(descriptions)
            if hint:
                request += '\nNoisy old instruction (may be wrong): ' + hint
            def validate_synthesis(result):
                if any(not isinstance(result.get(key), (str, list)) for key in ['shared_traits', 'limitations']):
                    raise ValueError('Synthesis needs shared_traits and limitations')
                result['modifier'] = validate_modifier(result, cfg)
                return result
            synthesis = generator.request_json(prompts.SYNTHESIS_SYSTEM, request,
                attempts=cfg['generation']['max_attempts'], validator=validate_synthesis)
            text = synthesis['modifier']
        else:
            text = generator.request_json(prompts.REFINE_SYSTEM, request, images,
                attempts=cfg['generation']['max_attempts'], validator=lambda r: validate_modifier(r, cfg))
        record = dict(query, modifier=text, reference_observation=reference_description,
                      target_observations=descriptions, synthesis=synthesis, perturbations={}, perturbation_reviews={})
        if cfg['generation']['quality_review']:
            record['quality_review'] = generator.request_json(prompts.REVIEW_SYSTEM,
                'Modifier: ' + text + '\nTarget observations: ' + json.dumps(descriptions),
                images, attempts=cfg['generation']['max_attempts'],
                validator=bool_review(['grounded', 'direction_correct', 'coherent', 'shared_across_targets']))
        records.append(record)
        save_json(path, records)
        logger.info('Generated %d/%d %s: %s', len(records), len(expected), query['query_id'], text)
    del generator
    release_model(None)
    quality_report(cfg, records, logger)


def perturb(cfg, logger):
    manifest = load_manifest(cfg)
    records = read_json(output_path(cfg, 'modifiers.json'))
    if [r['query_id'] for r in records] != [q['query_id'] for q in manifest['queries']]:
        raise ValueError('Complete generation before perturbation')
    kinds = cfg['perturbation']['types']
    if all(set(kinds) <= set(r['perturbations']) for r in records):
        logger.info('Perturbation already complete')
        return
    generator = TextGenerator(cfg, logger)
    for record in records:
        for kind, instruction in kinds.items():
            if kind in record['perturbations']:
                continue
            request = prompts.perturbation_prompt(record['modifier'], instruction)
            style = instruction
            for attempt in range(cfg['perturbation']['max_attempts']):
                if kind == 'typos':
                    text = generate_typos(generator, cfg, record['modifier'], record['query_id'])
                elif cfg['perturbation']['preserve_content_words']:
                    text = generate_word_preserving(generator, cfg, record['modifier'], kind, style)
                else:
                    text = generator.request_json(prompts.PERTURB_SYSTEM, request,
                        attempts=cfg['perturbation']['max_attempts'],
                        decoding=cfg['perturbation']['decoding'],
                        validator=lambda r: validate_perturbation(r, cfg, record['modifier'], kind))
                review = None
                if cfg['perturbation']['quality_review']:
                    review = generator.request_json(prompts.PERTURB_REVIEW_SYSTEM,
                        f'Original: {record["modifier"]}\nPerturbed: {text}',
                        attempts=cfg['perturbation']['max_attempts'], validator=bool_review(['meaning_preserved']))
                    if not review['meaning_preserved']:
                        logger.warning('Rejected meaning-changing %s: %s', kind, review['reason'])
                        request += '\nPrevious rejected version: ' + text + '\nPreserve the original constraints. Previous rejection: ' + review['reason']
                        style += '\nPrevious rejected version: ' + text + '\nFix: ' + review['reason']
                        continue
                break
            else:
                raise ValueError(f"No meaning-preserving {kind} after retries for {record['query_id']}")
            record['perturbations'][kind] = text
            if review is not None:
                record['perturbation_reviews'][kind] = review
            save_json(output_path(cfg, 'modifiers.json'), records)
            logger.info('Perturbed %s %s: %s', record['query_id'], kind, text)
    del generator
    release_model(None)
    quality_report(cfg, records, logger)


def quality_report(cfg, records, logger):
    reviewed = [r for r in records if 'quality_review' in r]
    checks = ['grounded', 'direction_correct', 'coherent', 'shared_across_targets']
    report = {'kind': 'model-assisted checks; human review still needed', 'queries': len(records),
              'reviewed': len(reviewed),
              'pass_counts': {k: sum(r['quality_review'][k] for r in reviewed) for k in checks},
              'flagged_query_ids': [r['query_id'] for r in reviewed if not all(r['quality_review'][k] for k in checks)],
              'unique_clean_modifiers': len({r['modifier'] for r in records}),
              'mean_clean_word_count': sum(len(r['modifier'].split()) for r in records) / len(records),
              'review_scope': 'Only the sampled conditioning targets; no claim about unshown targets'}
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
               'units': 'percent; deltas in percentage points', 'metrics': metrics,
               'delta_vs_clean': deltas, 'per_query': per_query,
               'protocol': 'Target-conditioned modifiers; fixed gallery and reference+text fusion.'})
    # CSV is convenient for plotting condition/metric changes outside Python.
    if not clean_only:
        import csv
        with output_path(cfg, 'metrics.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['condition', 'metric', 'value_percent', 'delta_pp_vs_clean'])
            for condition, values in metrics.items():
                for metric, value in values.items():
                    writer.writerow([condition, metric, value, deltas.get(condition, {}).get(metric, 0)])


def main():
    args = parse_args(['all', 'prepare', 'generate', 'retrieve-clean', 'perturb', 'evaluate', 'visualize'])
    cfg = get_config(args.config)
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
        logger.info('Completed stage=%s elapsed=%.2fs', args.stage, time.monotonic() - start)
    except Exception:
        logger.exception('Experiment failed; completed artifacts are preserved for resume')
        raise


if __name__ == '__main__':
    main()
