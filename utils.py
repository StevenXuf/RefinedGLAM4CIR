"""Configuration, atomic output, reproducibility, and GPU selection."""

import hashlib
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def get_config(path=None, split=None):
    """Load YAML, resolve paths, and reject settings that change the evaluation protocol."""
    path = Path(path or ROOT / 'config.yaml').resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg['split'] = split or cfg['split']
    from loading_data import validate_split

    validate_split(cfg)
    cfg['paths'] = {
        key: str((path.parent / Path(value).expanduser()).resolve())
        for key, value in cfg['paths'].items()
    }
    cfg['paths']['output_dir'] = str(Path(cfg['paths']['output_dir']) / cfg['split'])
    generation = cfg['generation']
    if not 1 <= generation['min_words'] <= generation['max_words'] <= 15:
        raise ValueError('Generation requires 1 <= min_words <= max_words <= 15')
    sizes = [
        generation['batch_size'],
        generation['image_batch_size'],
        cfg['retrieval']['batch_size'],
    ]
    if min(sizes) < 1:
        raise ValueError('Batch sizes must be positive')
    if generation['target_review_attempts'] < 1 or generation['shared_features_max_words'] < 1:
        raise ValueError('Target review attempts and shared-feature word limit must be positive')
    if type(generation['max_target_images']) is not int or generation['max_target_images'] < 1:
        raise ValueError(
            'max_target_images must be a positive integer; it never limits metric ground truths'
        )
    fallback = generation['fallback_modifiers']
    if not fallback or any(
        not isinstance(text, str)
        or not generation['min_words'] <= word_count(text) <= generation['max_words']
        for text in fallback
    ):
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


def setup_logging(cfg):
    """Write console messages and one fresh log per split, model pair, and seed."""
    path = output_path(cfg, 'run.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode='w')],
        force=True,
    )
    logging.captureWarnings(True)
    return logging.getLogger('glam')


def seed_everything(cfg):
    """Seed Python, NumPy, and PyTorch through Transformers' shared helper."""
    import torch
    from transformers import set_seed

    set_seed(cfg['seed'])
    torch.set_num_threads(cfg['runtime']['cpu_threads'])


def word_count(text):
    """Count whitespace-separated words, matching the generation limits."""
    return len(text.split())


def select_option(result, options, error='Choose one listed integer index'):
    """Resolve an MLLM's integer choice without accepting booleans or invalid indices."""
    index = result.get('choice')
    if type(index) is not int or not 0 <= index < len(options):
        raise ValueError(error)
    return options[index]


def get_device(cfg):
    import torch

    if cfg['runtime']['require_cuda'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA required but unavailable; use the GPU server')
    device = torch.device(
        f'cuda:{cfg["runtime"]["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
    )
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


def output_path(cfg, name, *, retrieval=True):
    """Name saved files by the models they use and the experiment seed."""
    path = Path(cfg['paths']['output_dir']) / name
    if not path.suffix:
        return path

    models = [cfg['generation']['model']]
    if retrieval:
        models.append(cfg['retrieval']['model'])
    suffix = '_'.join([*models, 'seed', str(cfg['seed'])])
    return path.with_stem(f'{path.stem}_{suffix}')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]
