"""Print per-seed metrics and mean/sample standard deviation without extra report files."""

import argparse
from copy import deepcopy

import pandas as pd

from loading_data import SPLITS
from utils import ROOT, get_config, output_path, read_json


def render_report(runs, top_ks):
    """Aggregate each metric and each run's cutoff average with pandas."""
    scores = pd.concat(
        {run['seed']: pd.DataFrame.from_dict(run['metrics'], orient='index') for run in runs},
        names=['Seed', 'Condition'],
    )
    groups = scores.groupby(level='Condition', sort=False)
    summary = groups.mean().map('{:.2f}'.format) + ' ± ' + groups.std(ddof=1).map('{:.2f}'.format)
    lines = [
        f'{runs[0]["split"]}, {runs[0]["retrieval"]["model"]}: '
        f'seeds {", ".join(str(run["seed"]) for run in runs)}; scores: %.'
    ]
    for family in ['Recall', 'MRR', 'mAP']:
        columns = [f'{family}@{k}' for k in top_ks] + [f'{family}_AVG']
        lines.extend(
            [
                family,
                scores[columns].reset_index().to_markdown(index=False, floatfmt='.2f'),
                'Mean ± sample SD',
                summary[columns].to_markdown(disable_numparse=True),
            ]
        )
    return '\n\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--split', choices=SPLITS)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--extractor', help='openclip, clip, siglip, or siglip2')
    args = parser.parse_args()
    for split in [args.split] if args.split else SPLITS:
        cfg = get_config(args.config, split, args.extractor)
        runs = []
        for seed in args.seeds:
            settings = deepcopy(cfg)
            settings['seed'] = seed
            runs.append(read_json(output_path(settings, 'metrics.json')))
        print(render_report(runs, cfg['evaluation']['top_ks']))


if __name__ == '__main__':
    main()
