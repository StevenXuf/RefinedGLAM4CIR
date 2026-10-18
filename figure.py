"""Three seeded CSV-triplet comparisons with equal square axes and centered wrapped text."""
import argparse
from textwrap import fill

from PIL import Image, ImageOps

from loading_data import SPLITS, load_image
from utils import ROOT, get_config, output_path


def display_image(path, size):
    return ImageOps.pad(load_image(path), (size, size), method=Image.Resampling.LANCZOS, color='white')


def retrieval_caption(ax, rank, score):
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    caption = HPacker(children=[
        TextArea(f'Top-{rank} ', textprops={'color': 'black', 'fontsize': 9}),
        TextArea(f'({score:.3f})', textprops={'color': '#176b44', 'fontsize': 9}),
    ], align='baseline', pad=0, sep=0)
    ax.add_artist(AnchoredOffsetbox(loc='lower center', child=caption, frameon=False,
        bbox_to_anchor=(.5, 1), bbox_transform=ax.transAxes, pad=0, borderpad=.8))


def visualize(cfg, records, gallery, retrieval):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from main import example_ids
    by_id = {g['image_id']: g['path'] for g in gallery}
    selected = set(example_ids(cfg, records))
    directory = output_path(cfg, 'figures')
    directory.mkdir(parents=True, exist_ok=True)
    size, top_k = cfg['visualization']['image_size'], cfg['visualization']['top_k']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    for item in records:
        if item['query_id'] not in selected:
            continue
        conditions = [('unperturbed', item['modifier'])] + [(k, item['perturbations'][k]) for k in cfg['perturbation']['types']]
        fig, axes = plt.subplots(len(conditions), top_k+3, squeeze=False,
            figsize=(2.35*(top_k+3), 2.85*len(conditions)))
        fig.set_facecolor('#edf1f6')
        for row, (condition, modifier) in enumerate(conditions):
            for ax in axes[row]:
                ax.set_box_aspect(1)
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            cards = [(axes[row, 0], item['reference_path'], 'Reference'),
                     (axes[row, 1], item['target_paths'][0], f'Target example ({len(item["target_ids"])} relevant)')]
            for ax, path, title in cards:
                ax.imshow(display_image(path, size))
                ax.set_title(fill(title, 22), loc='center', fontsize=9, pad=8)
            text_ax = axes[row, 2]
            text_ax.text(.5, .78, fill(condition.replace('_', ' ').title(), 24), ha='center', va='center',
                         transform=text_ax.transAxes, fontsize=10, fontweight='bold', color='#174675')
            modifier_text = text_ax.text(.5, .45, fill(modifier, cfg['visualization']['wrap_width']), ha='center', va='center',
                         transform=text_ax.transAxes, multialignment='center', fontsize=9, linespacing=1.4)
            modifier_text.set_in_layout(False)
            for rank, hit in enumerate(retrieval[condition][item['query_id']][:top_k], 1):
                ax = axes[row, rank+2]
                ax.imshow(display_image(by_id[hit['image_id']], size))
                relevant = hit['image_id'] in item['target_ids']
                retrieval_caption(ax, rank, hit['score'])
                if relevant:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('#20945d')
                        spine.set_linewidth(2.5)
        title = f'{item["query_id"]} · CSV reference/target pair · green border = relevant retrieval'
        fig.suptitle(fill(title, 95), x=.5, ha='center', fontsize=12, fontweight='bold', color='#17324d', y=.99)
        fig.tight_layout(rect=(.005, .005, .995, .965), pad=.6, h_pad=2.0, w_pad=.6)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, (_, modifier) in zip(axes[:, 2], conditions):
            text = ax.texts[1]
            width = cfg['visualization']['wrap_width']
            # Uppercase corruptions need narrower wrapping at the same font size.
            while width > 1 and text.get_window_extent(renderer).width > .94 * ax.get_window_extent(renderer).width:
                width -= 1
                text.set_text(fill(modifier, width))
        path = directory / f'query_{item["csv_row"]}.png'
        fig.savefig(path, dpi=cfg['visualization']['dpi'], facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    from main import evaluate, load_data, load_modifiers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--split', choices=SPLITS, required=True)
    args = parser.parse_args()
    cfg = get_config(args.config, args.split)
    queries, gallery = load_data(cfg)
    data = load_modifiers(cfg, queries, gallery)
    _, examples, _, _ = evaluate(cfg, data['records'], gallery, examples_only=True)
    visualize(cfg, data['records'], gallery.records, examples)
