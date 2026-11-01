"""Direct visual comparison with Qwen3-VL; one response per request."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

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
