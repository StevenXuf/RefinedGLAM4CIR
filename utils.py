"""Configuration, atomic output, reproducibility, and GPU selection."""
import hashlib
import json
import logging
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def get_config(path=None, split=None):
    path = Path(path or ROOT / 'config.yaml').resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg['split'] = split or cfg['split']
    from loading_data import validate_split
    validate_split(cfg)
    for key, value in cfg['paths'].items():
        cfg['paths'][key] = str((path.parent / Path(value).expanduser()).resolve())
    cfg['paths']['output_dir'] = str(Path(cfg['paths']['output_dir']) / cfg['split'])
    if not 1 <= cfg['generation']['min_words'] <= cfg['generation']['max_words'] <= 15:
        raise ValueError('Generation requires 1 <= min_words <= max_words <= 15')
    if min(cfg['generation']['batch_size'], cfg['generation']['image_batch_size'], cfg['retrieval']['batch_size']) < 1:
        raise ValueError('Batch sizes must be positive')
    if cfg['generation']['target_review_attempts'] < 1 or cfg['generation']['shared_features_max_words'] < 1:
        raise ValueError('Target review attempts and shared-feature word limit must be positive')
    if type(cfg['generation']['max_target_images']) is not int or cfg['generation']['max_target_images'] < 1:
        raise ValueError('max_target_images must be a positive integer; it never limits metric ground truths')
    fallback = cfg['generation']['fallback_modifiers']
    if not fallback or any(not isinstance(text, str) or not cfg['generation']['min_words'] <= len(text.split()) <= cfg['generation']['max_words'] for text in fallback):
        raise ValueError('Fallback requests must respect the configured word limits')
    if not 0 <= cfg['retrieval']['image_weight'] <= 1:
        raise ValueError('image_weight must be between zero and one')
    limit = cfg['subset'].get('query_limit')
    if limit is not None and limit < 1:
        raise ValueError('query_limit must be positive or null')
    from perturbation import KINDS, validate_settings
    if not cfg['perturbation']['types'] or set(cfg['perturbation']['types']) - set(KINDS):
        raise ValueError(f'Choose perturbations from {list(KINDS)}')
    validate_settings(cfg['perturbation'])
    return cfg


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', force=True)
    return logging.getLogger('glam')


def seed_everything(cfg):
    import numpy as np
    import torch
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])


def get_device(cfg):
    import torch
    if cfg['runtime']['require_cuda'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA required but unavailable; use the GPU server')
    device = torch.device(f"cuda:{cfg['runtime']['gpu_id']}" if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def output_path(cfg, name):
    return Path(cfg['paths']['output_dir']) / name


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
