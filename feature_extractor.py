"""Frozen OpenCLIP image/text encoders and L2-normalized additive composition."""

import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer

from utils import get_device


class FeatureExtractor:
    """Use OpenCLIP's aligned retrieval embeddings for both modalities."""

    def __init__(self, cfg):
        self.device = get_device(cfg)
        settings = cfg['retrieval']
        self.model, _, self.processor = create_model_and_transforms(
            settings['architecture'],
            pretrained=settings['pretrained'],
            cache_dir=cfg['paths']['model_cache'],
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer(settings['architecture'])

    @torch.inference_mode()
    def get_image_features(self, images):
        pixels = torch.stack([self.processor(image) for image in images]).to(self.device)
        return F.normalize(self.model.encode_image(pixels).float(), dim=-1)

    @torch.inference_mode()
    def get_text_features(self, texts):
        tokens = self.tokenizer(texts, context_length=self.model.context_length).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)

    def compose(self, reference_features, text_features):
        """Add individually normalized vectors element-wise, then L2-normalize."""
        return F.normalize(reference_features + text_features, dim=-1)
