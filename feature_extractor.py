"""Frozen OpenCLIP, CLIP, SigLIP, and SigLIP2 retrieval embeddings."""

import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import AutoModel, AutoProcessor

from utils import get_device


class FeatureExtractor:
    """Use each model's native processor and aligned image/text output heads."""

    def __init__(self, cfg):
        self.device = get_device(cfg)
        settings = cfg['retrieval']
        model = settings['models'][settings['model']]
        self.is_openclip = settings['model'] == 'openclip'

        if self.is_openclip:
            self.model, _, self.processor = create_model_and_transforms(
                model['architecture'],
                pretrained=model['pretrained'],
                cache_dir=cfg['paths']['model_cache'],
            )
            self.tokenizer = get_tokenizer(model['architecture'])
        else:
            common = dict(cache_dir=cfg['paths']['model_cache'], revision=model['revision'])
            self.model = AutoModel.from_pretrained(model['model_id'], **common)
            self.processor = AutoProcessor.from_pretrained(model['model_id'], **common)
            self.max_length = self.model.config.text_config.max_position_embeddings

        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def get_image_features(self, images):
        if self.is_openclip:
            pixels = torch.stack([self.processor(image) for image in images]).to(self.device)
            features = self.model.encode_image(pixels)
        else:
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            features = self.model.get_image_features(**inputs, return_dict=True).pooler_output

        return F.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def get_text_features(self, texts):
        if self.is_openclip:
            tokens = self.tokenizer(texts, context_length=self.model.context_length).to(self.device)
            features = self.model.encode_text(tokens)
        else:
            inputs = self.processor(
                text=texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self.device)
            features = self.model.get_text_features(**inputs, return_dict=True).pooler_output

        return F.normalize(features.float(), dim=-1)

    def compose(self, reference_features, text_features):
        """Add individually normalized vectors element-wise, then L2-normalize."""
        return F.normalize(reference_features + text_features, dim=-1)
