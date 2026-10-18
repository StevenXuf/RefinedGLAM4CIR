"""Resolve the original GLAM CSV pipeline against existing local images only."""
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_image(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def search_image_path(image_names, gallery_folders, glam_root):
    paths = []
    for name in image_names:
        for folder in gallery_folders:
            path = Path(glam_root) / folder / f'{name}.png'
            if path.is_file():
                paths.append(path)
                break
    return paths


def split_ids(value):
    return list(dict.fromkeys(x.strip() for x in str(value).split(',') if x.strip()))


SPLITS = ('test_id', 'test_ood')


def validate_split(cfg):
    if cfg.get('missing_gallery_policy', 'available') not in {'available', 'strict'}:
        raise ValueError('missing_gallery_policy must be available or strict')
    if cfg['split'] not in SPLITS:
        raise ValueError('Supported inference splits: test_id, test_ood')
    if not cfg['image_storage_folders']:
        raise ValueError('At least one local image storage folder is required')
    if cfg['subset'].get('gallery_limit') is not None:
        raise ValueError('Gallery subsampling is forbidden; always use the entire split gallery')
    if cfg['subset'].get('max_targets_per_query') is not None:
        raise ValueError('Target-count filtering is forbidden; preserve all CSV triplets')


def process_df(cfg, logger):
    validate_split(cfg)
    source = cfg['paths']['cir_file']
    df = pd.read_csv(source, dtype=str).fillna('')
    required = {'id1', 'id2', 'query', 'partition'}
    if not required <= set(df.columns):
        raise ValueError(f'{source} must contain triplet columns {sorted(required)}; image metadata cannot define query pairs')
    if not (df.partition == cfg['split']).any():
        raise ValueError(f'No {cfg["split"]} triplets in {source}. Provide existing id1/id2/query/partition annotations. db_processed.txt only defines galleries.')
    df['csv_row'] = df.index
    df = df[df.partition == cfg['split']].copy()
    initial = len(df)
    cache = {}
    def resolve(value):
        names = split_ids(value)
        for name in names:
            if name not in cache:
                found = search_image_path([name], cfg['image_storage_folders'], cfg['paths']['glam_root'])
                cache[name] = str(found[0]) if found else None
        return [name for name in names if cache[name]]
    for column in ['id1', 'id2']:
        df[column] = df[column].map(resolve)
    df = df[df.id1.map(bool) & df.id2.map(bool)]
    valid = len(df)
    if cfg['remove_duplicates']:
        keys = df.apply(lambda row: (tuple(sorted(row.id1)), tuple(sorted(row.id2))), axis=1)
        df = df.loc[~keys.duplicated()]
    logger.info('Queries: split=%s initial=%d missing-image rows=%d duplicate rows=%d valid=%d',
                cfg['split'], initial, initial - valid, valid - len(df), len(df))
    logger.info('Missing image IDs: %d', sum(v is None for v in cache.values()))
    limit = cfg['subset'].get('query_limit')
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=cfg['seed'])
    return df.sort_values('csv_row').reset_index(drop=True)


class GLAMQueryDataset(Dataset):
    """One query per CSV row, first valid reference, all existing relevant targets."""
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        logger = logger or logging.getLogger(__name__)
        self.df = process_df(cfg, logger)
        self.records = []
        for row in self.df.to_dict('records'):
            reference = search_image_path(row['id1'][:1], cfg['image_storage_folders'], cfg['paths']['glam_root'])[0]
            targets = search_image_path(row['id2'], cfg['image_storage_folders'], cfg['paths']['glam_root'])
            # A reference cannot be both masked and relevant in the same evaluation.
            targets = [p for p in targets if p.stem != reference.stem]
            if not targets:
                raise ValueError(f"CSV row {row['csv_row']} has no non-reference targets")
            self.records.append({
                'query_id': f"{cfg['split']}:{row['csv_row']}", 'csv_row': row['csv_row'],
                'reference_id': reference.stem, 'reference_path': str(reference),
                'target_ids': [p.stem for p in targets], 'target_paths': [str(p) for p in targets],
                'original_modifier': row['query'],
            })
        if not self.records:
            raise ValueError('No valid queries; check split and local image paths')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        record['pil_image'] = load_image(record['reference_path'])
        return record



class GLAMGalleryDataset(Dataset):
    def __init__(self, cfg, logger=None, queries=None):
        logger = logger or logging.getLogger(__name__)
        validate_split(cfg)
        df = pd.read_csv(cfg['paths']['db_processed_name'], header=None, dtype=str)
        names = sorted(set(df.loc[df[1] == cfg['split'], 0]))
        paths = search_image_path(names, cfg['image_storage_folders'], cfg['paths']['glam_root'])
        by_id = {p.stem: str(p) for p in paths}
        required = {name for q in (queries or []) for name in q['target_ids']}
        missing = required - by_id.keys()
        if missing:
            raise ValueError(f'Relevant targets absent from split gallery: {sorted(missing)}')
        total = len(by_id)
        absent = sorted(set(names) - by_id.keys())
        self.declared_size = len(names)
        self.missing_ids = absent
        if absent and cfg.get('missing_gallery_policy', 'available') == 'strict':
            raise FileNotFoundError(f'Full {cfg["split"]} gallery has {len(absent)} missing images: {absent[:10]}')
        if absent:
            logger.warning('Using ALL %d available %s gallery images; %d/%d declared files are missing. No sampling.', total, cfg['split'], len(absent), len(names))
        self.records = [{'image_id': name, 'path': path} for name, path in sorted(by_id.items())]
        if not self.records:
            raise ValueError('No gallery images available')
        logger.info('Gallery: full available=%d selected=%d relevant=%d scope=%s', total, len(self),
                    len(required), 'full')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        record['pil_image'] = load_image(record['path'])
        return record


def collate_samples(batch):
    return {key: [item[key] for item in batch] for key in batch[0]}
