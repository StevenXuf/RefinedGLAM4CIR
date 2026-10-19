"""Split-wise end-to-end inference, with independently callable pipeline stages."""
import argparse
from collections import Counter
import gc
import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import prompts
from feature_extractor import FeatureExtractor
from generation import TextGenerator, validate_modifier
from loading_data import GLAMGalleryDataset, GLAMQueryDataset, SPLITS, collate_samples, load_image
from metrics_computation import compute_metrics, metric_deltas, rank_queries
from perturbation import request as perturbation_request, sample_profile, validate_perturbation
from utils import (ROOT, batches, fingerprint, get_config, output_path, read_json,
                   save_json, seed_everything, setup_logging)


def load_data(cfg, logger=None):
    queries = GLAMQueryDataset(cfg, logger).records
    gallery = GLAMGalleryDataset(cfg, logger, queries)
    return queries, gallery


def signature(cfg, queries, gallery):
    return fingerprint({'queries': queries, 'gallery': gallery.records, 'config': cfg,
        'images': [(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in sorted(
            {g['path'] for g in gallery.records} | {q['reference_path'] for q in queries})],
        'sources': {name: (ROOT / name).read_text() for name in [
            'generation.py', 'prompts.py', 'perturbation.py', 'loading_data.py', 'main.py',
            'feature_extractor.py', 'metrics_computation.py', 'utils.py']}})


def load_modifiers(cfg, queries, gallery):
    path = output_path(cfg, 'modifiers.json')
    expected = signature(cfg, queries, gallery)
    if not path.exists():
        return {'signature': expected, 'split': cfg['split'], 'generation_model': cfg['generation']['models'][cfg['generation']['model']], 'records': []}
    data = read_json(path)
    if data['signature'] != expected:
        raise ValueError('Configuration, source, or image availability changed; choose a new paths.output_dir')
    by_id = {q['query_id']: q for q in queries}
    if len({r['query_id'] for r in data['records']}) != len(data['records']):
        raise ValueError('Duplicate modifier query IDs')
    for record in data['records']:
        if record['query_id'] not in by_id or any(record.get(k) != v for k, v in by_id[record['query_id']].items()):
            raise ValueError('Saved modifier does not match its original CSV triplet')
        validate_modifier({'modifier': record['modifier']}, cfg)
        for kind, text in record['perturbations'].items():
            validate_perturbation(text, record['modifier'], kind, cfg, record['query_id'])
    return data


def reuse_clean_modifiers(cfg, queries, data, source):
    """Explicitly import compatible clean queries, discarding every old perturbation."""
    source = Path(source).resolve()
    previous = read_json(source)
    if data['records']:
        raise ValueError('--reuse-clean requires an empty output checkpoint; omit it to resume')
    expected_model = cfg['generation']['models'][cfg['generation']['model']]
    if previous.get('generation_model') != expected_model or previous.get('split') != cfg['split']:
        raise ValueError('Clean source must use the same split and generation checkpoint/revision')
    records = previous['records']
    if len({r['query_id'] for r in records}) != len(records):
        raise ValueError('Duplicate query IDs in clean source')
    by_id = {r['query_id']: r for r in records}
    imported = []
    for query in queries:
        record = by_id.get(query['query_id'])
        if record is None or any(record.get(k) != v for k, v in query.items()):
            raise ValueError(f'Clean source does not match CSV triplet {query["query_id"]}')
        text = validate_modifier({'modifier': record['modifier']}, cfg)
        imported.append(dict(query, modifier=text, perturbations={}))
    data['clean_source'] = {'path': str(source), 'content_hash': fingerprint(previous),
                            'signature': previous.get('signature')}
    checkpoint(cfg, data, {r['query_id']: r for r in imported}, queries)
    logging.getLogger('glam').info('Reused %d exact clean CSV modifiers from %s; all perturbations will be regenerated', len(imported), source)


def checkpoint(cfg, data, records, queries):
    data['records'] = [records[q['query_id']] for q in queries if q['query_id'] in records]
    save_json(output_path(cfg, 'modifiers.json'), data)


def require_success(outputs):
    failures = [str(value) for value in outputs if isinstance(value, Exception)]
    if failures:
        raise ValueError(f'{len(failures)} generation items failed; successful modifiers were saved: {failures[0]}')


def generate_modifiers(cfg, queries, data, generator=None, logger=None):
    logger = logger or logging.getLogger('glam')
    records = {r['query_id']: r for r in data['records']}
    pending = [q for q in queries if q['query_id'] not in records]
    if not pending:
        return generator
    generator = generator or TextGenerator(cfg, logger)
    paths = {}
    for query in pending:
        paths[query['reference_id']] = query['reference_path']
        paths.update(zip(query['target_ids'], query['target_paths']))
    observations = {}
    def description(result):
        text = result.get('description')
        if not isinstance(text, str) or not text.strip() or len(text.split()) > 80:
            raise ValueError('description must be a nonempty sentence of at most 80 words')
        return text.strip()
    for batch in batches(sorted(paths.items()), cfg['generation']['image_batch_size']):
        tasks = [{'field': 'description', 'system': prompts.DESCRIPTION, 'prompt': 'Identify the main object, visible colour and shape in one short phrase. Return ONLY JSON with description, ideally 15-25 words. Finish the JSON object.', 'images': [load_image(path)]}
                 for _, path in batch]
        outputs = generator.request_json_many(tasks, [description] * len(tasks))
        require_success(outputs)
        observations.update((name, text) for (name, _), text in zip(batch, outputs))
        logger.info('%s: observed %d/%d CSV images', cfg['split'], len(observations), len(paths))
    for batch in batches(pending, cfg['generation']['batch_size']):
        tasks = [{'field': 'modifier', 'system': prompts.MODIFIER, 'prompt': 'BEFORE: ' + observations[q['reference_id']] +
                  '\nAFTER alternatives: ' + json.dumps([observations[name] for name in q['target_ids']]) +
                  f"\nUse {cfg['generation']['min_words']}-{cfg['generation']['max_words']} words."} for q in batch]
        outputs = generator.request_json_many(tasks, [lambda result: validate_modifier(result, cfg)] * len(tasks))
        for query, text in zip(batch, outputs):
            if not isinstance(text, Exception):
                records[query['query_id']] = dict(query, modifier=text, perturbations={})
        checkpoint(cfg, data, records, queries)
        require_success(outputs)
        logger.info('%s: generated %d/%d modifiers', cfg['split'], len(records), len(queries))
    return generator


def perturb_modifiers(cfg, queries, data, generator=None, logger=None):
    logger = logger or logging.getLogger('glam')
    if len(data['records']) != len(queries):
        raise ValueError('Generate all clean modifiers before perturbation')
    records = {r['query_id']: r for r in data['records']}
    for kind in cfg['perturbation']['types']:
        pending = [r for r in data['records'] if kind not in r['perturbations']]
        if pending and generator is None:
            generator = TextGenerator(cfg, logger)
        for batch in batches(pending, cfg['generation']['batch_size']):
            requests = [perturbation_request(r['modifier'], kind, cfg, r['query_id']) for r in batch]
            outputs = generator.request_json_many([r[0] for r in requests], [r[1] for r in requests])
            for record, text in zip(batch, outputs):
                if not isinstance(text, Exception):
                    record['perturbations'][kind] = text
            checkpoint(cfg, data, records, queries)
            require_success(outputs)
            logger.info('%s: %s %d/%d queries', cfg['split'], kind,
                        sum(kind in r['perturbations'] for r in data['records']), len(queries))
        severities = Counter(sample_profile(r['modifier'], kind, cfg, r['query_id'])['severity'] for r in data['records'])
        logger.info('%s: %s sampled severity distribution %s', cfg['split'], kind, dict(severities))
        logger.info('%s: %s complete for %d queries', cfg['split'], kind, len(queries))
    return generator


def encode_images(records, key, extractor, cfg):
    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)
        def __getitem__(self, index):
            return {'image': load_image(records[index][key])}
    loader = DataLoader(Images(), batch_size=cfg['retrieval']['batch_size'], shuffle=False,
                        num_workers=cfg['runtime']['num_workers'], collate_fn=collate_samples)
    features = []
    for batch in loader:
        features.append(extractor.get_image_features(batch['image']).cpu())
    return torch.cat(features).to(extractor.device)


def example_ids(cfg, records):
    return random.Random(cfg['seed']).sample([r['query_id'] for r in records],
        min(cfg['visualization']['num_examples'], len(records)))


def evaluate(cfg, records, gallery, extractor=None, features=None, conditions=None, examples_only=False):
    extractor = extractor or FeatureExtractor(cfg)
    if features is None:
        features = (encode_images(gallery.records, 'path', extractor, cfg),
                    encode_images(records, 'reference_path', extractor, cfg))
    gallery_features, reference_features = features
    conditions = conditions or ['unperturbed'] + cfg['perturbation']['types']
    gallery_ids = [g['image_id'] for g in gallery.records]
    gallery_index = {name: i for i, name in enumerate(gallery_ids)}
    selected = set(example_ids(cfg, records))
    metrics, examples = {}, {}
    for condition in conditions:
        rows = []
        examples[condition] = {}
        for start in range(0, len(records), cfg['evaluation']['query_batch_size']):
            batch = records[start:start+cfg['evaluation']['query_batch_size']]
            indexes = [i for i, r in enumerate(batch) if not examples_only or r['query_id'] in selected]
            if not indexes:
                continue
            chosen = [batch[i] for i in indexes]
            texts = [r['modifier'] if condition == 'unperturbed' else r['perturbations'][condition] for r in chosen]
            query_features = extractor.compose(reference_features[[start+i for i in indexes]], texts)
            rankings, scores = rank_queries(query_features, gallery_features, gallery_ids, chosen,
                                             cfg['evaluation']['exclude_reference'])
            _, details = compute_metrics(rankings, chosen, cfg['evaluation']['top_ks'])
            rows.extend(details)
            for i, record in enumerate(chosen):
                if record['query_id'] in selected:
                    examples[condition][record['query_id']] = [
                        {'image_id': name, 'score': float(scores[i, gallery_index[name]])}
                        for name in rankings[i][:cfg['visualization']['top_k']]]
        metrics[condition] = {k: sum(row[k] for row in rows)/len(rows) for k in rows[0] if k != 'query_id'}
    return metrics, examples, extractor, features


def print_metrics(cfg, metrics, count, gallery):
    columns = [f'Recall@{k}' for k in cfg['evaluation']['top_ks']] + ['AVG', 'mAP', 'MRR']
    print(f'\n{cfg["split"]}: {count} queries; {len(gallery)}/{gallery.declared_size} available gallery images (no sampling). Scores: %.')
    print('| Condition | ' + ' | '.join(columns) + ' |')
    print('|---|' + '---:|' * len(columns))
    for condition, values in metrics.items():
        print('| ' + condition + ' | ' + ' | '.join(f'{values[c]:.2f}' for c in columns) + ' |', flush=True)


def save_metrics(cfg, metrics, queries, gallery):
    result = {'split': cfg['split'], 'queries': len(queries), 'query_limit': cfg['subset']['query_limit'],
        'gallery_size': len(gallery), 'declared_gallery_size': gallery.declared_size,
        'missing_gallery_count': len(gallery.missing_ids), 'units': 'percent; deltas in percentage points',
        'generation_model': cfg['generation']['models'][cfg['generation']['model']], 'retrieval': cfg['retrieval'],
        'seed': cfg['seed'], 'perturbation': cfg['perturbation'],
        'metrics': metrics, 'delta_vs_unperturbed': {kind: metric_deltas(metrics['unperturbed'], values)
            for kind, values in metrics.items() if kind != 'unperturbed'}}
    save_json(output_path(cfg, 'metrics.json'), result)
    print_metrics(cfg, metrics, len(queries), gallery)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--stage', choices=['all', 'data', 'generate', 'perturb', 'evaluate', 'visualize'], default='all')
    parser.add_argument('--reuse-clean', metavar='MODIFIERS_JSON',
                        help='Import clean modifiers from the same model/split and exact CSV triplets; regenerate corruptions')
    args = parser.parse_args()
    cfg = get_config(args.config, args.split)
    logger = setup_logging()
    seed_everything(cfg)
    queries, gallery = load_data(cfg, logger)
    if args.stage == 'data':
        print(f"{cfg['split']}: {len(queries)} valid queries; {len(gallery)}/{gallery.declared_size} available gallery images")
        return
    data = load_modifiers(cfg, queries, gallery)
    if args.reuse_clean:
        reuse_clean_modifiers(cfg, queries, data, args.reuse_clean)
    generator, extractor, features = None, None, None
    if args.stage in {'all', 'generate'}:
        generator = generate_modifiers(cfg, queries, data, logger=logger)
    if args.stage == 'all':
        metrics, _, extractor, features = evaluate(cfg, data['records'], gallery, conditions=['unperturbed'])
        save_metrics(cfg, metrics, queries, gallery)
    if args.stage in {'all', 'perturb'}:
        generator = perturb_modifiers(cfg, queries, data, generator, logger)
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.stage in {'all', 'evaluate', 'visualize'}:
        if len(data['records']) != len(queries) or any(set(cfg['perturbation']['types']) - r['perturbations'].keys() for r in data['records']):
            raise ValueError('Generate and perturb every query before paired retrieval')
        metrics, examples, extractor, features = evaluate(cfg, data['records'], gallery, extractor, features,
                                                         examples_only=args.stage == 'visualize')
        if args.stage != 'visualize':
            save_metrics(cfg, metrics, queries, gallery)
        if args.stage in {'all', 'visualize'}:
            from figure import visualize
            visualize(cfg, data['records'], gallery.records, examples)


if __name__ == '__main__':
    main()
