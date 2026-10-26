"""Print independent-seed results and their mean ± sample standard deviation."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
from statistics import mean, stdev

from loading_data import SPLITS
from pipeline import generation_targets
from utils import ROOT, get_config, output_path, read_json


def load_runs(cfg, seeds):
    """Reject incomplete runs or changes to the cohort and scientific settings."""
    runs, triplets = [], None
    fields = (
        'split',
        'queries',
        'query_limit',
        'gallery_size',
        'declared_gallery_size',
        'missing_gallery_count',
        'units',
        'generation_model',
        'retrieval',
        'generation_policy',
        'generation_settings',
        'evaluation',
        'perturbation',
    )
    expected_settings = None
    for seed in seeds:
        settings = deepcopy(cfg)
        settings['seed'] = seed
        run = read_json(output_path(settings, 'metrics.json'))
        modifiers = read_json(output_path(settings, 'modifiers.json', retrieval=False))
        if run['seed'] != seed or run['split'] != cfg['split']:
            raise ValueError(f'Incorrect seed/split metadata for seed {seed}')
        scientific_settings = {key: deepcopy(run[key]) for key in fields}
        scientific_settings['generation_settings'].pop('gpu_ids', None)
        if expected_settings is None:
            expected_settings = scientific_settings
        elif scientific_settings != expected_settings:
            raise ValueError(
                f'Incompatible experiment settings or coverage for seed {seed}'
            )

        current_triplets = {}
        for record in modifiers['records']:
            query_id = record['query_id']
            if query_id in current_triplets:
                raise ValueError(f'Duplicate query {query_id} in seed {seed}')
            current_triplets[query_id] = {
                key: value
                for key, value in record.items()
                if key not in {'modifier', 'perturbations', 'generation_target_ids'}
            }
            if record['generation_target_ids'] != generation_targets(settings, record):
                raise ValueError(f'Incorrect seeded generation targets for {query_id}')
            if set(record['perturbations']) != set(cfg['perturbation']['types']):
                raise ValueError(f'Incomplete perturbations for {query_id}')
        if len(current_triplets) != run['queries']:
            raise ValueError(f'Incomplete modifier checkpoint for seed {seed}')
        if triplets is None:
            triplets = current_triplets
        elif current_triplets != triplets:
            raise ValueError(f'Query cohort or ground truths changed for seed {seed}')
        runs.append(run)

    conditions = set(runs[0]['metrics'])
    expected_conditions = {
        'reference_only',
        'original_text_only',
        'generated_text_only',
        'reference_original',
        'unperturbed',
        *cfg['perturbation']['types'],
    }
    columns = {
        f'{family}{suffix}'
        for family in ('Recall', 'MRR', 'mAP')
        for suffix in [*(f'@{k}' for k in cfg['evaluation']['top_ks']), '_AVG']
    }
    for run in runs:
        if set(run['metrics']) != conditions or conditions != expected_conditions:
            raise ValueError('Missing or unexpected retrieval conditions')
        for scores in run['metrics'].values():
            if set(scores) != columns or any(
                not math.isfinite(value) or not 0 <= value <= 100
                for value in scores.values()
            ):
                raise ValueError('Incomplete or invalid metric values')
            for family in ('Recall', 'MRR', 'mAP'):
                expected = mean(
                    scores[f'{family}@{k}'] for k in cfg['evaluation']['top_ks']
                )
                if not math.isclose(expected, scores[f'{family}_AVG'], abs_tol=1e-10):
                    raise ValueError(
                        'AVG must be computed within each seed before aggregation'
                    )
    return runs


def render_report(runs, top_ks):
    run = runs[0]
    seeds = ', '.join(str(item['seed']) for item in runs)
    lines = [
        f'# {run["split"]}: seeds {seeds}',
        '',
        f'{run["queries"]} queries; {run["gallery_size"]}/{run["declared_gallery_size"]} '
        'available gallery images, without gallery sampling.',
        '',
        'Scores and means are percentages; SD is the sample standard deviation '
        '(ddof=1), in percentage points. AVG is computed within each run first.',
    ]
    for family in ('Recall', 'MRR', 'mAP'):
        columns = [f'{family}@{k}' for k in top_ks] + [f'{family}_AVG']
        lines.extend(
            [
                '',
                f'## {family}: individual seeds',
                '',
                '| Condition | Seed | '
                + ' | '.join([*(f'@{k}' for k in top_ks), 'AVG'])
                + ' |',
                '|---|---:|' + '---:|' * len(columns),
            ]
        )
        for condition in run['metrics']:
            for item in runs:
                values = item['metrics'][condition]
                lines.append(
                    f'| {condition} | {item["seed"]} | '
                    + ' | '.join(f'{values[key]:.2f}' for key in columns)
                    + ' |'
                )
        lines.extend(
            [
                '',
                f'## {family}: mean ± SD',
                '',
                '| Condition | '
                + ' | '.join([*(f'@{k}' for k in top_ks), 'AVG'])
                + ' |',
                '|---|' + '---:|' * len(columns),
            ]
        )
        for condition in run['metrics']:
            cells = []
            for key in columns:
                values = [item['metrics'][condition][key] for item in runs]
                cells.append(f'{mean(values):.2f} ± {stdev(values):.2f}')
            lines.append('| ' + condition + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--split', choices=SPLITS, help='Default: report both splits')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Provide at least two distinct seeds')
    for split in [args.split] if args.split else SPLITS:
        cfg = get_config(args.config, split)
        runs = load_runs(cfg, args.seeds)
        report = render_report(runs, cfg['evaluation']['top_ks'])
        suffix = '_'.join(map(str, args.seeds))
        path = Path(cfg['paths']['output_dir']) / (
            f'metrics_summary_{cfg["generation"]["model"]}_'
            f'{cfg["retrieval"]["model"]}_seeds_{suffix}.md'
        )
        path.write_text(report)
        print(report, end='')
        print(f'\nSaved: {path}\n')


if __name__ == '__main__':
    main()
