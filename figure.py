"""Render paired clean/perturbed top-5 retrieval panels from saved artifacts."""
import random
from textwrap import fill

from loading_data import load_image
from utils import get_config, output_path, parse_args, read_json


def visualize(cfg, logger=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = read_json(output_path(cfg, 'manifest.json'))
    generated = read_json(output_path(cfg, 'modifiers.json'))
    results = read_json(output_path(cfg, 'retrieval.json'))
    by_id = {g['image_id']: g['path'] for g in manifest['gallery']}
    count = min(cfg['visualization']['num_examples'], len(generated))
    indices = random.Random(cfg['seed']).sample(range(len(generated)), count)
    directory = output_path(cfg, 'figures')
    directory.mkdir(parents=True, exist_ok=True)
    top_k = cfg['visualization']['top_k']
    for index in indices:
        item = generated[index]
        conditions = [('clean', item['modifier'])] + list(item['perturbations'].items())
        fig, axes = plt.subplots(len(conditions), top_k + 3,
                                 figsize=(3.2 * (top_k + 3), 3.8 * len(conditions)), squeeze=False)
        for row, (condition, modifier) in enumerate(conditions):
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[row, 0].imshow(load_image(item['reference_path']))
            axes[row, 0].set_title('Reference')
            shown = item['conditioning_target_ids']
            target_id = shown[0]
            axes[row, 1].imshow(load_image(by_id[target_id]))
            axes[row, 1].set_title(f'Target example ({len(item["target_ids"])} relevant)\n{target_id}', fontsize=9)
            axes[row, 2].text(0, 1, fill(f'{condition}: {modifier}', 33), va='top', fontsize=10)
            axes[row, 2].text(0, 0.08, fill('CSV: ' + item['original_modifier'], 33), fontsize=8)
            for rank, hit in enumerate(results[condition][index]['top_results'][:top_k]):
                ax = axes[row, rank + 3]
                ax.imshow(load_image(by_id[hit['image_id']]))
                ax.set_title(f'#{rank + 1} {hit["image_id"]}\ncosine={hit["score"]:.3f}', fontsize=9)
                if hit['image_id'] in item['target_ids']:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('green')
                        spine.set_linewidth(3)
        fig.suptitle(f'{item["query_id"]} | {manifest["scope"]} gallery | green = relevant', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = directory / f'query_{item["csv_row"]}.png'
        fig.savefig(path, dpi=cfg['visualization']['dpi'], bbox_inches='tight')
        plt.close(fig)
        if logger:
            logger.info('Saved paired visualization %s', path)


if __name__ == '__main__':
    args = parse_args(['visualize'])
    visualize(get_config(args.config, split=args.split))
