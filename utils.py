"""Shared configuration, output, reproducibility, and modifier analysis."""

import hashlib
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def get_config(path=None, split=None):
    """Load YAML, resolve local paths, and select the requested inference split."""
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


def modifier_length_statistics(modifiers):
    """Describe whitespace-separated word counts and characters including spaces."""
    import pandas as pd

    texts = pd.Series(modifiers, dtype='string')
    lengths = pd.DataFrame({'words': texts.str.split().str.len(), 'characters': texts.str.len()})
    return dict(count=len(texts), **lengths.agg(['mean', 'median', 'min', 'max']).to_dict())


def plot_wordcloud(modifiers, path, seed, settings):
    """Save a reproducible unigram cloud using WordCloud's English stopwords."""
    from wordcloud import WordCloud

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    WordCloud(random_state=seed, **settings).generate(' '.join(modifiers)).to_file(str(path))
