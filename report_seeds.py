"""Print independent-seed results and their mean ± sample standard deviation."""

import argparse
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean

import pandas as pd

from loading_data import SPLITS
from pipeline import BASELINES, generation_targets
from utils import ROOT, get_config, output_path, read_json


def load_runs(cfg, seeds):
    """Reject incomplete runs or changes to the cohort and scientific settings."""
    runs, triplets = [], None
    fields = (
        'split queries query_limit gallery_size declared_gallery_size missing_gallery_count units '
        'generation_model retrieval generation_policy generation_settings evaluation perturbation'
    ).split()
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
            raise ValueError(f'Incompatible experiment settings or coverage for seed {seed}')

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

    expected_conditions = {*BASELINES, 'unperturbed', *cfg['perturbation']['types']}
    columns = {
        f'{family}{suffix}'
        for family in ('Recall', 'MRR', 'mAP')
        for suffix in [*(f'@{k}' for k in cfg['evaluation']['top_ks']), '_AVG']
    }
    for run in runs:
        if set(run['metrics']) != expected_conditions:
            raise ValueError('Missing or unexpected retrieval conditions')
        for scores in run['metrics'].values():
            if set(scores) != columns or any(
                not math.isfinite(value) or not 0 <= value <= 100 for value in scores.values()
            ):
                raise ValueError('Incomplete or invalid metric values')
            for family in ('Recall', 'MRR', 'mAP'):
                expected = mean(scores[f'{family}@{k}'] for k in cfg['evaluation']['top_ks'])
                if not math.isclose(expected, scores[f'{family}_AVG'], abs_tol=1e-10):
                    raise ValueError('AVG must be computed within each seed before aggregation')
    return runs


def render_report(runs, top_ks):
    """Use pandas for ordered tables and per-condition mean/sample SD across seeds."""
    run = runs[0]
    seeds = [item['seed'] for item in runs]
    scores = pd.concat(
        {
            item['seed']: pd.DataFrame.from_dict(item['metrics'], orient='index', dtype=float)
            for item in runs
        },
        names=['Seed', 'Condition'],
    )
    order = pd.MultiIndex.from_product([run['metrics'], seeds], names=['Condition', 'Seed'])
    individual = scores.swaplevel().reindex(order).reset_index()
    groups = scores.groupby(level='Condition', sort=False)
    summary = groups.mean().map('{:.2f}'.format) + ' ± ' + groups.std(ddof=1).map('{:.2f}'.format)
    lines = [
        f'# {run["split"]}: seeds {", ".join(map(str, seeds))}',
        '',
        f'{run["queries"]} queries; {run["gallery_size"]}/{run["declared_gallery_size"]} '
        'available gallery images, without gallery sampling.',
        '',
        'Scores and means are percentages; SD is the sample standard deviation '
        '(ddof=1), in percentage points. AVG is computed within each run first.',
    ]
    for family in ('Recall', 'MRR', 'mAP'):
        columns = [f'{family}@{k}' for k in top_ks] + [f'{family}_AVG']
        labels = dict(zip(columns, [*(f'@{k}' for k in top_ks), 'AVG']))
        lines.extend(
            [
                '',
                f'## {family}: individual seeds',
                '',
                individual[['Condition', 'Seed', *columns]]
                .rename(columns=labels)
                .to_markdown(index=False, floatfmt='.2f'),
                '',
                f'## {family}: mean ± SD',
                '',
                summary[columns].rename(columns=labels).to_markdown(disable_numparse=True),
            ]
        )
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
