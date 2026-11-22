"""Qwen3-VL generation, outlier review, and saved-modifier analysis."""

import langid
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import prompts
import utils
from loading_data import GLAMQueryDataset, SPLITS, load_image


class DirectQwen:
    """Compare each reference/target image group directly in one Qwen request."""

    def __init__(self, cfg):
        spec = cfg['generation']
        common = dict(revision=spec['revision'], cache_dir=cfg['paths']['model_cache'])
        self.processor = AutoProcessor.from_pretrained(spec['model_id'], **common)
        self.processor.tokenizer.padding_side = 'left'
        self.processor.image_processor.size['longest_edge'] = cfg['generation']['max_pixels']
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec['model_id'],
            **common,
            dtype=getattr(torch, cfg['runtime']['dtype']),
            device_map=f'cuda:{cfg["runtime"]["gpu_id"]}',
            attn_implementation='sdpa',
        ).eval()
        self.max_tokens = spec['max_new_tokens']

    @torch.inference_mode()
    def generate(self, prompts, image_groups):
        messages, images = [], []

        for prompt, paths in zip(prompts, image_groups):
            content = [{'type': 'text', 'text': prompt}]
            for index, path in enumerate(paths):
                label = 'Reference image' if index == 0 else f'Target image {index}'
                content.extend([{'type': 'text', 'text': label}, {'type': 'image'}])
                images.append(load_image(path))
            messages.append([{'role': 'user', 'content': content}])

        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        inputs = self.processor(text=texts, images=images, return_tensors='pt', padding=True).to(
            self.model.device
        )

        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)

        tokens = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        outputs = self.processor.batch_decode(
            tokens[:, inputs['input_ids'].shape[1] :], skip_special_tokens=True
        )
        return [text.strip() for text in outputs]


def modifier_is_outlier(text, settings):
    """Screen language, length, and instruction form without an MLLM call."""
    words = text.split()
    return (
        not settings['min_words'] <= len(words) <= settings['max_words']
        or words[0].lower() not in settings['modifier_actions']
        or langid.classify(text)[0] != 'en'
    )


def verify_modifiers(model, cfg, modifiers, image_groups, logger):
    """Use one image-grounded review/correction call for outliers only."""
    settings = cfg['generation']
    modifiers = list(modifiers)
    outliers = [i for i, text in enumerate(modifiers) if modifier_is_outlier(text, settings)]
    logger.info('Qwen modifier review: %d/%d outliers', len(outliers), len(modifiers))

    if outliers:
        corrected = model.generate(
            [prompts.MODIFIER_REVIEW.format(modifier=modifiers[i], **settings) for i in outliers],
            [image_groups[i] for i in outliers],
        )
        for i, text in zip(outliers, corrected):
            if not settings['min_words'] <= len(text.split()) <= settings['max_words']:
                raise ValueError(
                    f'Modifier still fails length checks at batch position {i}: {text}'
                )
            modifiers[i] = text
    return modifiers, set(outliers)


def analyze_modifiers(cfg, seeds):
    """Report lengths and vocabulary for saved modifiers; plot per-corpus word clouds."""
    from pathlib import Path

    import pandas as pd

    directory, model = Path(cfg['paths']['output_dir']), cfg['generation']['model']
    corpora = {
        f'original_{model}_seed_{cfg["seed"]}': [
            q['original_modifier'] for q in GLAMQueryDataset(cfg).records
        ]
    }

    for seed in seeds:
        records = utils.read_json(
            utils.output_path(dict(cfg, seed=seed), 'modifiers.json', retrieval=False)
        )
        corpora[f'generated_{model}_seed_{seed}'] = [r['modifier'] for r in records]

    stats = {
        name: dict(utils.modifier_length_statistics(texts), unique_words=utils.count_unique_words(texts))
        for name, texts in corpora.items()
    }

    table = pd.json_normalize(stats.values()).set_axis(stats)
    averages = table.iloc[1:][['words.mean', 'characters.mean', 'unique_words']].agg(['mean', 'std'])
    pooled_unique_words = utils.count_unique_words(
        text for texts in list(corpora.values())[1:] for text in texts
    )

    summary = dict(
        split=cfg['split'],
        seeds=seeds,
        modifiers=stats,
        across_seeds=averages.replace({float('nan'): None}).to_dict(),
        generated_unique_words_all_seeds=pooled_unique_words,
    )
    tag = f'{model}_seeds_' + '_'.join(map(str, seeds))
    utils.save_json(directory / f'modifier_lengths_{tag}.json', summary)
    settings = cfg['modifier_analysis']['wordcloud']

    for name, texts in corpora.items():
        utils.plot_wordcloud(
            texts, directory / f'wordcloud_{name}.pdf', int(name.rsplit('_', 1)[1]), settings
        )
        
    print(cfg['split'], table.to_markdown(floatfmt='.4f'), sep='\n')
    print('Across seeds (mean and sample SD):', averages.to_markdown(floatfmt='.4f'), sep='\n')
    print('Generated unique words across all seeds:', pooled_unique_words)
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze saved modifiers; no model inference.')
    parser.add_argument('--split', choices=SPLITS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--extractor', help='openclip, clip, siglip, or siglip2')
    args = parser.parse_args()
    for split in [args.split] if args.split else SPLITS:
        cfg = utils.get_config(split=split, extractor=args.extractor)
        analyze_modifiers(cfg, args.seeds or cfg['modifier_analysis']['seeds'])
