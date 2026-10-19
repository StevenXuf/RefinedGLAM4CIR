"""Run real data/model component checks without evaluating a sampled gallery."""
import argparse
import gc
import logging
import tempfile

import torch

from feature_extractor import FeatureExtractor
from generation import TextGenerator
from loading_data import load_image
from main import generate_modifiers, load_data, load_modifiers, perturb_modifiers
from perturbation import validate_perturbation
from utils import ROOT, get_config, seed_everything, setup_logging


def check_retrieval(cfg):
    queries, _ = load_data(cfg)
    images = [load_image(q['reference_path']) for q in queries[:2]]
    texts = ['a red ceramic vase', 'a blue round bowl']
    extractor = FeatureExtractor(cfg)
    image = extractor.get_image_features(images)
    text = extractor.get_text_features(texts)
    assert image.shape == text.shape and image.ndim == 2
    for vectors in [image, text]:
        assert torch.isfinite(vectors).all()
        torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2, device=vectors.device), atol=1e-5, rtol=1e-5)
    cosine = image @ text.T
    with torch.inference_mode():
        if extractor.name == 'blip':
            # This official ITC forward must equal cosine of our independent projected vectors.
            inputs = extractor.processor(images=images, text=texts, padding=True,
                                          truncation=True, max_length=35, return_tensors='pt').to(extractor.device)
            official = extractor.model(**inputs, use_itm_head=False).itm_score
        elif extractor.name in {'clip', 'siglip', 'siglip2'}:
            inputs = extractor.processor(images=images, text=texts, padding='max_length' if extractor.name.startswith('siglip') else True,
                                          truncation=True, return_tensors='pt').to(extractor.device)
            official = extractor.model(**inputs).logits_per_image
            official = official / extractor.model.logit_scale.exp() if extractor.name == 'clip' else (
                official - extractor.model.logit_bias) / extractor.model.logit_scale.exp()
        else:
            pixel = torch.stack([extractor.processor(im) for im in images]).to(extractor.device)
            tokens = extractor.tokenizer(texts, context_length=extractor.model.context_length).to(extractor.device)
            native_image = extractor.model.encode_image(pixel, normalize=True)
            native_text = extractor.model.encode_text(tokens, normalize=True)
            official = native_image @ native_text.T
    torch.testing.assert_close(cosine, official.float(), atol=3e-5, rtol=3e-5)
    composed = extractor.compose(image, texts)
    assert composed.shape == image.shape and torch.isfinite(composed).all()
    print(f'PASS {extractor.name}: shared projected embeddings {tuple(image.shape)}; cosine equals official retrieval scores', flush=True)


def check_generation(cfg):
    cfg['subset']['query_limit'] = None
    queries, gallery = load_data(cfg)
    queries = sorted(queries, key=lambda q: (len(q['target_ids']), q['csv_row']))[:3]
    cfg['subset']['query_limit'] = 3
    data = load_modifiers(cfg, queries, gallery)
    generator = TextGenerator(cfg)
    original_request = generator.request_json_many
    def inspected_request(tasks, validators):
        results = original_request(tasks, validators)
        for task, result in zip(tasks, results):
            if task.get('field') == 'description':
                print('Image observation:', result, flush=True)
        return results
    generator.request_json_many = inspected_request
    generate_modifiers(cfg, queries, data, generator, logging.getLogger('verify'))
    perturb_modifiers(cfg, queries, data, generator, logging.getLogger('verify'))
    for record in data['records']:
        assert len(record['modifier'].split()) <= 15
        for kind, text in record['perturbations'].items():
            validate_perturbation(text, record['modifier'], kind, cfg, record['query_id'])
        print(record['query_id'], record['modifier'], record['perturbations'], flush=True)
    print(f'PASS {cfg["generation"]["model"]}: {len(queries)} real CSV triplets, all targets, all corruption types', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--component', choices=['generation', 'retrieval'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', type=int, help='Optional diagnostic GPU override')
    parser.add_argument('--split', choices=['test_id', 'test_ood'], default='test_ood')
    args = parser.parse_args()
    cfg = get_config(args.config, args.split)
    if args.gpu is not None:
        cfg['runtime']['gpu_id'] = args.gpu
    cfg['subset']['query_limit'] = 3
    directory = ROOT / 'outputs' / 'checks'
    directory.mkdir(parents=True, exist_ok=True)
    cfg['paths']['output_dir'] = tempfile.mkdtemp(prefix=f'{args.component}-{args.model}-{args.split}-', dir=directory)
    cfg[args.component]['model'] = args.model
    setup_logging()
    seed_everything(cfg)
    if args.component == 'generation':
        check_generation(cfg)
    else:
        check_retrieval(cfg)
    gc.collect()
    torch.cuda.empty_cache()
