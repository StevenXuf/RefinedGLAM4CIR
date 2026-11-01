"""Direct visual comparison and image-grounded review with Qwen3-VL."""

import langid
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import prompts
from loading_data import load_image


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


def modifier_is_outlier(text, settings, check_action=True):
    """Screen language, length, and instruction form without an MLLM call."""
    words = text.split()
    return (
        not settings['min_words'] <= len(words) <= settings['max_words']
        or (check_action and words[0].lower() not in settings['modifier_actions'])
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
            if modifier_is_outlier(text, settings, check_action=False):
                raise ValueError(
                    f'Modifier still fails English/length checks at batch position {i}: {text}'
                )
            modifiers[i] = text
    return modifiers, set(outliers)
