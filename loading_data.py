"""Resolve the original GLAM CSV pipeline against existing local PNG images only."""
import logging
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


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


def process_df(cfg, logger):
    df = pd.read_csv(cfg['paths']['cir_file'], dtype=str).fillna('')
    df['csv_row'] = df.index
    df = df[df.partition == cfg['split']].copy()
    initial = len(df)
    cache = {}
    def resolve(value):
        names = split_ids(value)
        for name in names:
            if name not in cache:
                found = search_image_path([name], cfg['image_gallery'], cfg['paths']['glam_root'])
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
    max_targets = cfg['subset'].get('max_targets_per_query')
    if max_targets is not None:
        before = len(df)
        df = df[df.id2.map(len) <= max_targets]
        logger.info('Target-count smoke budget <=%d: retained %d/%d queries', max_targets, len(df), before)
    limit = cfg['subset']['query_limit']
    if limit is not None and len(df) > limit:
        if cfg['subset'].get('balance_categories', False):
            groups = {}
            for index, row in df.iterrows():
                category = row.id1[0].rsplit('_', 1)[0]
                groups.setdefault(category, []).append(index)
            rng = random.Random(cfg['seed'])
            categories = sorted(groups)
            rng.shuffle(categories)
            for indices in groups.values():
                rng.shuffle(indices)
            selected = []
            while len(selected) < limit:
                for category in categories:
                    if groups[category] and len(selected) < limit:
                        selected.append(groups[category].pop())
            df = df.loc[selected]
        else:
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
            reference = search_image_path(row['id1'][:1], cfg['image_gallery'], cfg['paths']['glam_root'])[0]
            targets = search_image_path(row['id2'], cfg['image_gallery'], cfg['paths']['glam_root'])
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

    def load_dataset(self):
        return self

    def load_dataloader(self):
        return DataLoader(self, batch_size=self.cfg['retrieval']['batch_size'], shuffle=False,
                          num_workers=self.cfg['runtime']['num_workers'], collate_fn=collate_samples)


class GLAMGalleryDataset(Dataset):
    def __init__(self, cfg, logger=None, queries=None):
        logger = logger or logging.getLogger(__name__)
        df = pd.read_csv(cfg['paths']['db_processed_name'], header=None, dtype=str)
        names = sorted(set(df.loc[df[1] == cfg['split'], 0]))
        paths = search_image_path(names, cfg['image_gallery'], cfg['paths']['glam_root'])
        by_id = {p.stem: str(p) for p in paths}
        required = {name for q in (queries or []) for name in q['target_ids']}
        missing = required - by_id.keys()
        if missing:
            raise ValueError(f'Relevant targets absent from split gallery: {sorted(missing)}')
        total = len(by_id)
        limit = cfg['subset']['gallery_limit']
        if limit is not None and total > limit:
            if len(required) > limit:
                raise ValueError(f'gallery_limit={limit} cannot fit {len(required)} relevant targets')
            other = sorted(by_id.keys() - required)
            keep = required | set(random.Random(cfg['seed']).sample(other, limit - len(required)))
            by_id = {name: by_id[name] for name in sorted(keep)}
        self.records = [{'image_id': name, 'path': path} for name, path in sorted(by_id.items())]
        self.image_paths = [Path(item['path']) for item in self.records]
        self.full_size = total
        if not self.records:
            raise ValueError('No gallery images available')
        logger.info('Gallery: full available=%d selected=%d relevant=%d scope=%s', total, len(self),
                    len(required), 'full' if len(self) == total else 'subset (smoke metrics only)')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        record['pil_image'] = load_image(record['path'])
        return record


def collate_samples(batch):
    return {key: [item[key] for item in batch] for key in batch[0]}
