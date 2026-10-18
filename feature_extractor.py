"""Image/text encoders and normalized reference + modifier composition."""
import logging

import torch
import torch.nn.functional as F

from utils import get_device


class FeatureExtractor:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.device = get_device(cfg)
        self.name = cfg['retrieval']['model']
        models = cfg['retrieval']['models']
        common = dict(cache_dir=cfg['paths']['model_cache'],
                      local_files_only=cfg['runtime']['local_files_only'])
        model_id = models[self.name]
        self.logger.info('Loading retrieval model=%s device=%s', model_id, self.device)
        if self.name in ['openclip', 'coca', 'openvision']:
            from open_clip import create_model_and_transforms, create_model_from_pretrained, get_tokenizer
            if self.name == 'openvision':
                self.model, self.processor = create_model_from_pretrained(model_id, cache_dir=common['cache_dir'])
            else:
                self.model, _, self.processor = create_model_and_transforms(
                    model_id, pretrained=models[self.name + '_pretrained'], cache_dir=common['cache_dir'])
            self.tokenizer = get_tokenizer(model_id, cache_dir=common['cache_dir'])
        else:
            from transformers import AutoProcessor, AutoModel, BlipForImageTextRetrieval
            cls = BlipForImageTextRetrieval if self.name == 'blip' else AutoModel
            self.model, loading = cls.from_pretrained(model_id, output_loading_info=True, **common)
            if loading['missing_keys'] or loading.get('mismatched_keys'):
                raise ValueError(f'Retrieval weights are incomplete: {loading}')
            self.processor = AutoProcessor.from_pretrained(model_id, **common)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def get_image_features(self, images):
        if self.name in ['openclip', 'coca', 'openvision']:
            tensor = torch.stack([self.processor(image) for image in images]).to(self.device)
            features = self.model.encode_image(tensor)
        else:
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            if self.name == 'blip':
                embeds = self.model.vision_model(**inputs).last_hidden_state[:, 0]
                features = self.model.vision_proj(embeds)
            else:
                features = self.model.get_image_features(**inputs)
        return normalized_features(features)

    @torch.inference_mode()
    def get_text_features(self, texts):
        if self.name in ['openclip', 'coca', 'openvision']:
            tokens = self.tokenizer(texts, context_length=self.model.context_length).to(self.device)
            features = self.model.encode_text(tokens)
        else:
            options = dict(padding='max_length' if self.name.startswith('siglip') else True,
                           truncation=True, return_tensors='pt')
            if self.name.startswith('siglip'):
                options['max_length'] = self.model.config.text_config.max_position_embeddings
            if self.name == 'blip':
                options['max_length'] = 35
            inputs = self.processor(text=texts, **options).to(self.device)
            if self.name == 'blip':
                # BLIP ITC text-only encoder: no cross-attention or LM/ITM classification head.
                inputs = {k: v for k, v in inputs.items() if k in {'input_ids', 'attention_mask'}}
                embeds = self.model.text_encoder(**inputs, return_dict=True).last_hidden_state[:, 0]
                features = self.model.text_proj(embeds)
            else:
                features = self.model.get_text_features(**inputs)
        return normalized_features(features)

    def compose(self, reference_features, modifiers):
        weight = self.cfg['retrieval']['image_weight']
        text = self.get_text_features(modifiers)
        if reference_features.shape != text.shape:
            raise ValueError('Image/text embeddings must have matching batch and shared projection dimensions')
        return F.normalize(weight * F.normalize(reference_features.float(), dim=-1) + (1 - weight) * text, dim=-1)


def normalized_features(features):
    # Transformers 5 returns projected pooler outputs from get_*_features; 4 returns tensors.
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    if features.ndim != 2 or not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
        raise ValueError('Expected finite, nonzero projected retrieval vectors [batch, dimension]')
    return F.normalize(features.float(), dim=-1)
