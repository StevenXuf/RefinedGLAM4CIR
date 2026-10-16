"""Configuration, reproducibility, logging, and atomic experiment artifacts."""
import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def get_config(path=None):
    path = Path(path or ROOT / 'config.yaml').resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    for key, value in cfg['paths'].items():
        p = Path(value).expanduser()
        cfg['paths'][key] = str(p if p.is_absolute() else path.parent / p)
    if cfg['generation']['batch_size'] != 1:
        raise ValueError('Multi-image generation currently requires generation.batch_size: 1')
    if cfg['generation']['max_target_images'] < 1:
        raise ValueError('max_target_images must be positive')
    if not 0 <= cfg['retrieval']['image_weight'] <= 1:
        raise ValueError('retrieval.image_weight must be between 0 and 1')
    for key in ['query_limit', 'gallery_limit', 'max_targets_per_query']:
        if cfg['subset'].get(key) is not None and cfg['subset'][key] < 1:
            raise ValueError(f'{key} must be positive or null')
    if cfg['visualization']['num_examples'] < 1 or cfg['visualization']['top_k'] < 1:
        raise ValueError('Visualization counts must be positive')
    if not all(isinstance(word, str) for word in cfg['perturbation']['editable_words']):
        raise ValueError('editable_words must be strings; quote YAML words such as "on"')
    if not 1 <= cfg['perturbation']['typo_words'] <= cfg['perturbation']['max_typo_words']:
        raise ValueError('typo_words must be positive and <= max_typo_words')
    return cfg


def parse_args(stages):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--stage', choices=stages, default=stages[0])
    return parser.parse_args()


def setup_logging(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.FileHandler(path), logging.StreamHandler()], force=True)
    return logging.getLogger('perturbedcir')


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
        raise RuntimeError('CUDA required but unavailable; run on the GPU server.')
    device = torch.device(f"cuda:{cfg['runtime']['gpu_id']}" if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_path(cfg, name):
    return Path(cfg['paths']['output_dir']) / name
