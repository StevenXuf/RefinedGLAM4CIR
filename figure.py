"""Three seeded CSV-triplet comparisons with equal square axes and centered wrapped text."""

import argparse
import logging
from textwrap import fill

from PIL import Image, ImageOps

from loading_data import SPLITS, load_image
from utils import ROOT, get_config, output_path, setup_logging


def display_image(path, size):
    return ImageOps.pad(
        load_image(path), (size, size), method=Image.Resampling.LANCZOS, color='white'
    )


def retrieval_caption(ax, rank, score):
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    caption = HPacker(
        children=[
            TextArea(f'Top-{rank} ', textprops={'color': 'black', 'fontsize': 9}),
            TextArea(f'({score:.3f})', textprops={'color': '#176b44', 'fontsize': 9}),
        ],
        align='baseline',
        pad=0,
        sep=0,
    )
    ax.add_artist(
        AnchoredOffsetbox(
            loc='lower center',
            child=caption,
            frameon=False,
            bbox_to_anchor=(0.5, 1),
            bbox_transform=ax.transAxes,
            pad=0,
            borderpad=0.8,
        )
    )


def figure_conditions(cfg, item):
    return [
        ('reference_original', item['original_modifier']),
        ('unperturbed', item['modifier']),
    ] + [(kind, item['perturbations'][kind]) for kind in cfg['perturbation']['operations']]


def visualize(cfg, records, gallery, retrieval):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from pipeline import example_ids

    by_id = {g['image_id']: g['path'] for g in gallery}
    selected = set(example_ids(cfg, records))
    directory = output_path(cfg, 'figures')
    directory.mkdir(parents=True, exist_ok=True)
    size, top_k = cfg['visualization']['image_size'], cfg['visualization']['top_k']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    for item in records:
        if item['query_id'] not in selected:
            continue
        conditions = figure_conditions(cfg, item)
        fig, axes = plt.subplots(
            len(conditions),
            top_k + 3,
            squeeze=False,
            figsize=(2.35 * (top_k + 3), 2.85 * len(conditions)),
        )
        for row, (condition, modifier) in enumerate(conditions):
            for ax in axes[row]:
                ax.set(box_aspect=1, facecolor='white', xticks=[], yticks=[])
                ax.spines[:].set_visible(False)
            cards = [
                (axes[row, 0], item['reference_path'], 'Reference'),
                (
                    axes[row, 1],
                    item['target_paths'][0],
                    f'Target (out of {len(item["target_ids"])})',
                ),
            ]
            for ax, path, title in cards:
                ax.imshow(display_image(path, size))
                ax.set_title(fill(title, 22), loc='center', fontsize=9, pad=8)
            text_ax = axes[row, 2]
            label = {
                'reference_original': 'Original modifier',
                'unperturbed': 'Generated modifier',
            }.get(condition, condition.replace('_', ' ').title())
            text_ax.text(
                0.5,
                0.78,
                fill(label, 24),
                ha='center',
                va='center',
                transform=text_ax.transAxes,
                fontsize=10,
                fontweight='bold',
                color='#174675',
            )
            modifier_text = text_ax.text(
                0.5,
                0.45,
                fill(modifier, cfg['visualization']['wrap_width']),
                ha='center',
                va='center',
                transform=text_ax.transAxes,
                multialignment='center',
                fontsize=9,
                linespacing=1.4,
            )
            modifier_text.set_in_layout(False)
            for rank, hit in enumerate(retrieval[condition][item['query_id']][:top_k], 1):
                ax = axes[row, rank + 2]
                ax.imshow(display_image(by_id[hit['image_id']], size))
                relevant = hit['image_id'] in item['target_ids']
                retrieval_caption(ax, rank, hit['score'])
                if relevant:
                    ax.spines[:].set(visible=True, color='#20945d', linewidth=2.5)
        fig.tight_layout(rect=(0.005, 0.005, 0.995, 0.995), pad=0.6, h_pad=2.0, w_pad=0.6)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, (_, modifier) in zip(axes[:, 2], conditions):
            text = ax.texts[1]
            width = cfg['visualization']['wrap_width']
            # Fit long modifier text within the fixed-size panel.
            while (
                width > 1
                and text.get_window_extent(renderer).width
                > 0.94 * ax.get_window_extent(renderer).width
            ):
                width -= 1
                text.set_text(fill(modifier, width))
        path = output_path(cfg, f'figures/example_{item["csv_row"]}.png')
        fig.savefig(path, dpi=cfg['visualization']['dpi'], facecolor='white', transparent=False)
        plt.close(fig)
        logging.getLogger('glam').info('Saved %s', path)


if __name__ == '__main__':
    from pipeline import evaluate, load_data
    from utils import read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--split', choices=SPLITS, required=True)
    args = parser.parse_args()
    cfg = get_config(args.config, args.split)
    setup_logging(cfg)
    queries, gallery = load_data(cfg)
    records = read_json(output_path(cfg, 'perturbations.json', retrieval=False))
    _, examples = evaluate(cfg, records, gallery, examples_only=True)
    visualize(cfg, records, gallery.records, examples)
