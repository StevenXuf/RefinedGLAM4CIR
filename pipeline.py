"""Direct visual modifier generation, Python corruptions, and full-gallery retrieval."""

import argparse
import gc
import logging
import random
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader

import prompts
from feature_extractor import FeatureExtractor
from generation import DirectQwen
from loading_data import GLAMGalleryDataset, GLAMQueryDataset, ImageDataset, SPLITS, collate_samples
from metrics_computation import compute_metrics, metric_deltas, rank_queries
from perturbation import perturb_modifiers, perturbation_summary
from utils import (
    ROOT,
    batches,
    fingerprint,
    get_config,
    output_path,
    read_json,
    save_json,
    seed_everything,
    setup_logging,
)

BASELINES = ('reference_only', 'original_text_only', 'generated_text_only', 'reference_original')


def load_data(cfg, logger=None):
    """Keep existing available CSV triplets and the complete available split gallery."""
    queries = GLAMQueryDataset(cfg, logger).records
    return queries, GLAMGalleryDataset(cfg, logger, queries)


def generation_targets(cfg, query):
    """Sample targets only for generation; retain every available target for metrics."""
    ids = query['target_ids']
    limit = cfg['generation']['max_target_images']
    if len(ids) <= limit:
        return list(ids)
    seed = int(
        fingerprint({'seed': cfg['seed'], 'query_id': query['query_id'], 'targets': sorted(ids)}),
        16,
    )
    return random.Random(seed).sample(sorted(ids), limit)


def generate_modifiers(cfg, queries, model, logger):
    """Show Qwen the reference and up to three target images together, without descriptions."""
    records = []
    for batch in batches(queries, cfg['generation']['batch_size']):
        selected = [generation_targets(cfg, query) for query in batch]
        groups = [
            [query['reference_path']]
            + [
                path
                for name, path in zip(query['target_ids'], query['target_paths'])
                if name in target_ids
            ]
            for query, target_ids in zip(batch, selected)
        ]
        modifiers = model.generate(
            [prompts.MODIFIER.format(**cfg['generation'])] * len(batch), groups
        )
        records.extend(
            dict(query, generation_target_ids=target_ids, modifier=text)
            for query, target_ids, text in zip(batch, selected, modifiers)
        )
        logger.info('Direct visual generation: %d/%d queries', len(records), len(queries))
    save_json(output_path(cfg, 'modifiers.json', retrieval=False), records)
    return records


def encode_images(records, path_key, extractor, cfg):
    """Load RGB images through a shared dataset and encode them in record order."""
    loader = DataLoader(
        ImageDataset(records, path_key),
        batch_size=cfg['retrieval']['batch_size'],
        shuffle=False,
        num_workers=cfg['runtime']['num_workers'],
        collate_fn=collate_samples,
    )
    return torch.cat(
        [extractor.get_image_features(batch['pil_image']).cpu() for batch in loader]
    ).to(extractor.device)


def example_ids(cfg, records):
    return random.Random(cfg['seed']).sample(
        [r['query_id'] for r in records], min(cfg['visualization']['num_examples'], len(records))
    )


def evaluate(cfg, records, gallery, logger=None, examples_only=False):
    """Use unit image/text vectors, normalized element-wise sums, and full-gallery cosine ranks."""
    logger = logger or logging.getLogger('glam')
    selected = set(example_ids(cfg, records))
    if examples_only:
        records = [record for record in records if record['query_id'] in selected]
    extractor = FeatureExtractor(cfg)
    gallery_features = encode_images(gallery.records, 'path', extractor, cfg)
    references = encode_images(records, 'reference_path', extractor, cfg)
    gallery_ids = [item['image_id'] for item in gallery.records]
    gallery_index = {name: i for i, name in enumerate(gallery_ids)}
    metrics, examples = {}, {}
    conditions = [*BASELINES, 'unperturbed', *cfg['perturbation']['operations']]
    for condition in conditions:
        if condition == 'reference_only':
            features = references
        else:
            texts = [
                record['original_modifier']
                if condition in {'original_text_only', 'reference_original'}
                else record['modifier']
                if condition in {'generated_text_only', 'unperturbed'}
                else record['perturbations'][condition]
                for record in records
            ]
            text_features = torch.cat(
                [
                    extractor.get_text_features(batch)
                    for batch in batches(texts, cfg['retrieval']['batch_size'])
                ]
            )
            features = (
                text_features
                if condition in {'original_text_only', 'generated_text_only'}
                else extractor.compose(references, text_features)
            )
        rankings, scores = rank_queries(
            features, gallery_features, gallery_ids, records, cfg['evaluation']['exclude_reference']
        )
        examples[condition] = {
            record['query_id']: [
                dict(image_id=name, score=float(scores[row, gallery_index[name]]))
                for name in rankings[row][: cfg['visualization']['top_k']]
            ]
            for row, record in enumerate(records)
            if record['query_id'] in selected
        }
        if not examples_only:
            metrics[condition] = compute_metrics(
                rankings,
                records,
                cfg['evaluation']['top_ks'],
                gallery_ids,
                cfg['evaluation']['exclude_reference'],
            )
            logger.info(
                'Evaluated %s: %d queries, %d gallery images', condition, len(records), len(gallery)
            )
    return metrics, examples


def print_metrics(cfg, metrics, count, gallery):
    """Print all metric cutoffs and their arithmetic average in separate tables."""
    table = pd.DataFrame.from_dict(metrics, orient='index', dtype=float).rename_axis('Condition')
    logger = logging.getLogger('glam')
    logger.info(
        '%s, seed %d: %d queries; %d/%d available gallery images. Scores: %%.',
        cfg['split'],
        cfg['seed'],
        count,
        len(gallery),
        gallery.declared_size,
    )
    for family in ['Recall', 'MRR', 'mAP']:
        columns = [f'{family}@{k}' for k in cfg['evaluation']['top_ks']] + [f'{family}_AVG']
        logger.info('\n%s', table[columns].to_markdown(floatfmt='.2f'))


def save_metrics(cfg, records, gallery, metrics):
    """Store scores, coverage, settings, and applied corruption statistics."""
    statistics = perturbation_summary(records, cfg['perturbation']['operations'])
    save_json(
        output_path(cfg, 'metrics.json'),
        dict(
            seed=cfg['seed'],
            split=cfg['split'],
            queries=len(records),
            gallery_size=len(gallery),
            declared_gallery_size=gallery.declared_size,
            missing_gallery_count=len(gallery.missing_ids),
            units='percent',
            generation=cfg['generation'],
            retrieval=cfg['retrieval'],
            evaluation=cfg['evaluation'],
            perturbation=cfg['perturbation'],
            prompts=dict(modifier=prompts.MODIFIER),
            composition='L2-normalize(unit image embedding + unit text embedding)',
            perturbation_engine='NLPAug 1.1.11 and Python standard library; one output per type, no model-based corruption or verification.',
            severity_counts=dict(Counter(record['severity'] for record in records)),
            perturbation_statistics=statistics,
            metrics=metrics,
            delta_vs_unperturbed={
                kind: metric_deltas(metrics['unperturbed'], metrics[kind])
                for kind in cfg['perturbation']['operations']
            },
        ),
    )
    print_metrics(cfg, metrics, len(records), gallery)
    table = pd.DataFrame.from_dict(statistics, orient='index').rename_axis('Perturbation')
    logging.getLogger('glam').info(
        'Python corruption statistics:\n%s', table.to_markdown(floatfmt='.2f')
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument(
        '--stage',
        choices=['all', 'data', 'generate', 'perturb', 'evaluate', 'visualize'],
        default='all',
    )
    args = parser.parse_args()
    cfg = get_config(args.config, args.split)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.gpu is not None:
        cfg['runtime']['gpu_id'] = args.gpu
    seed_everything(cfg)
    logger = setup_logging(cfg)
    queries, gallery = load_data(cfg, logger)
    if args.stage == 'data':
        return
    if args.stage in {'all', 'generate'}:
        model = DirectQwen(cfg)
        generate_modifiers(cfg, queries, model, logger)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if args.stage in {'all', 'perturb'}:
        records = read_json(output_path(cfg, 'modifiers.json', retrieval=False))
        records = perturb_modifiers(cfg, records, logger)
        save_json(output_path(cfg, 'perturbations.json', retrieval=False), records)
    if args.stage in {'all', 'evaluate', 'visualize'}:
        records = read_json(output_path(cfg, 'perturbations.json', retrieval=False))
        metrics, examples = evaluate(
            cfg, records, gallery, logger, examples_only=args.stage == 'visualize'
        )
        if args.stage != 'visualize':
            save_metrics(cfg, records, gallery, metrics)
        if args.stage in {'all', 'visualize'}:
            from figure import visualize

            visualize(cfg, records, gallery.records, examples)


if __name__ == '__main__':
    main()
