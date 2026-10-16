import logging

import pandas as pd
import pytest
import torch
from PIL import Image

from feature_extractor import FeatureExtractor
from generation import parse_json, validate_modifier
from loading_data import GLAMGalleryDataset, GLAMQueryDataset
from metrics_computation import compute_metrics, metric_deltas, rank_queries
from run import load_manifest, prepare
from utils import get_config


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg['runtime']['require_cuda'] = False
    cfg['runtime']['num_workers'] = 0
    cfg['paths'].update(glam_root=str(tmp_path / 'raw'), cir_file=str(tmp_path / 'cir.csv'),
                        db_processed_name=str(tmp_path / 'db.txt'), output_dir=str(tmp_path / 'out'))
    cfg['split'] = 'test_ood'
    cfg['image_storage_folders'] = ['test_ood']
    cfg['subset'] = {'query_limit': None}
    directory = tmp_path / 'raw' / 'test_ood'
    directory.mkdir(parents=True)
    for name in ['ref', 'a', 'b', 'c', 'd']:
        Image.new('RGB', (32, 40), color='white').save(directory / f'{name}.png')
    pd.DataFrame([
        dict(id1='ref', id2='a, b', query='look for a square', partition='test_ood'),
        dict(id1='ref', id2='b,a', query='duplicate pair', partition='test_ood'),
        dict(id1='missing', id2='a', query='missing ref', partition='test_ood'),
        dict(id1='ref', id2='missing', query='missing target', partition='test_ood'),
    ]).to_csv(cfg['paths']['cir_file'], index=False)
    (tmp_path / 'db.txt').write_text('ref,test_ood\na,test_ood\nb,test_ood\nc,test_ood\nd,test_ood\na,test_ood\n')
    return cfg


def test_csv_pairs_local_paths_and_deterministic_gallery(cfg):
    queries = GLAMQueryDataset(cfg)
    assert len(queries) == 1
    assert queries.records[0]['original_modifier'] == 'look for a square'
    assert queries.records[0]['target_ids'] == ['a', 'b']
    assert queries[0]['pil_image'].size == (32, 40)  # retain visual detail
    gallery = GLAMGalleryDataset(cfg, queries=queries.records)
    assert len(gallery) == 5
    assert {'a', 'b'} <= {r['image_id'] for r in gallery.records}
    assert gallery.records == GLAMGalleryDataset(cfg, queries=queries.records).records


def test_gallery_rejects_missing_targets_and_subsampling(cfg):
    queries = GLAMQueryDataset(cfg).records
    cfg['subset']['gallery_limit'] = 1
    with pytest.raises(ValueError, match='subsampling is forbidden'):
        GLAMGalleryDataset(cfg, queries=queries)
    del cfg['subset']['gallery_limit']
    queries[0]['target_ids'].append('not_in_gallery')
    with pytest.raises(ValueError, match='absent'):
        GLAMGalleryDataset(cfg, queries=queries)


def test_manifest_resume_and_config_change_guard(cfg):
    logger = logging.getLogger('test')
    manifest = prepare(cfg, logger)
    assert manifest['scope'] == 'full'
    assert manifest == prepare(cfg, logger)
    assert manifest['queries'][0]['conditioning_target_ids'] == ['a', 'b']
    cfg['seed'] += 1
    with pytest.raises(ValueError, match='Config/data changed'):
        load_manifest(cfg)


def test_multitarget_macro_recall_differs_from_hit_rate():
    queries = [dict(query_id='q1', target_ids=['a', 'b']), dict(query_id='q2', target_ids=['c'])]
    metrics, per_query = compute_metrics([['a', 'c', 'b'], ['a', 'c', 'b']], queries, [1, 2, 5])
    assert metrics['Recall@1'] == 25
    assert metrics['HitRate@1'] == 50
    assert metrics['Recall@2'] == 75
    assert metrics['Recall@5'] == 100
    assert metrics['MRR'] == 75
    assert metrics['mAP'] == pytest.approx(100 * ((1 + 2 / 3) / 2 + 1 / 2) / 2)
    assert metric_deltas({'a': 50}, {'a': 25}) == {'a': -25}


def test_reference_mask_and_missing_ground_truth():
    queries = [dict(query_id='q', reference_id='ref', target_ids=['target'])]
    query = torch.tensor([[1., 0.]])
    gallery = torch.tensor([[1., 0.], [0.8, 0.2], [0., 1.]])
    ranks, scores = rank_queries(query, gallery, ['ref', 'target', 'other'], queries)
    assert ranks == [['target', 'other']]
    assert scores[0, 0] == -torch.inf
    queries[0]['target_ids'] = ['missing']
    with pytest.raises(ValueError, match='Missing ground truth'):
        rank_queries(query, gallery, ['ref', 'target', 'other'], queries)


def test_normalize_each_modality_before_fusion(cfg):
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.cfg = cfg
    extractor.get_text_features = lambda _: torch.tensor([[0., 1.]])
    result = extractor.compose(torch.tensor([[1., 0.]]), ['modifier'])
    assert torch.allclose(result, torch.tensor([[2 ** -0.5, 2 ** -0.5]]))


def test_generation_json_and_invalid_perturbation(cfg):
    result = parse_json('```json\n{"modifier": "Change the round bowl to a square plate."}\n```')
    text = validate_modifier(result, cfg)
    with pytest.raises(ValueError, match='unchanged'):
        validate_modifier(result, cfg, original=text)
    with pytest.raises(ValueError, match='length'):
        validate_modifier({'modifier': ''}, cfg)
    with pytest.raises(ValueError, match='JSON'):
        parse_json('not JSON')


def test_duplicate_rankings_fail():
    with pytest.raises(ValueError, match='unique'):
        compute_metrics([['a', 'a']], [dict(query_id='q', target_ids=['a'])], [1])


def test_json_retry_rejects_unchanged_text_before_accepting_variant(cfg):
    from generation import TextGenerator
    generator = TextGenerator.__new__(TextGenerator)
    generator.logger = logging.getLogger('retry-test')
    original = 'Replace the red bowl with a blue vase.'
    responses = iter(['not JSON', '{"modifier": "' + original + '"}',
                      '{"modifier": "Repalce the red bowl with a blue vase."}'])
    generator.generate = lambda *args, **kwargs: next(responses)
    result = generator.request_json('system', 'prompt', attempts=3,
        validator=lambda r: validate_modifier(r, cfg, original))
    assert result == 'Repalce the red bowl with a blue vase.'


def test_perturbation_rejects_visual_drift_and_excessive_typos(cfg):
    from generation import validate_perturbation
    original = 'Replace the ornate bronze faucets with a black metal knocker.'
    with pytest.raises(ValueError, match='visual words'):
        validate_perturbation({'modifier': 'Swap the ornate brass faucets for a black metal knob.'}, cfg, original, 'informal')
    valid = 'Please swap the ornate bronze faucets for a black metal knocker.'
    assert validate_perturbation({'modifier': valid}, cfg, original, 'informal') == valid
    with pytest.raises(ValueError, match='changed'):
        validate_perturbation({'modifier': 'Repalce teh ornate bronz faucets with a black metal knocker.'}, cfg, original, 'typos')
    typo = 'Repalce the ornate bronze faucets wiht a black metal knocker.'
    assert validate_perturbation({'modifier': typo}, cfg, original, 'typos') == typo


def test_word_preserving_llm_rewrite_restores_visual_spans(cfg):
    from generation import generate_word_preserving
    class Generator:
        def request_json(self, *args, **kwargs):
            return kwargs['validator']({'modifier': 'Please swap the CONTENT0 for a CONTENT1.'})
    text = generate_word_preserving(Generator(), cfg,
        'Replace the ornate bronze faucets with a black metal knocker.', 'informal', 'Make it casual')
    assert text == 'Please swap the ornate bronze faucets for a black metal knocker.'


def test_llm_typo_is_applied_only_to_selected_spans(cfg):
    import re
    from generation import generate_typos, word_tokens
    class Generator:
        def request_json(self, system, prompt, **kwargs):
            word = re.search(r'Misspell ONLY this word: (\w+)', prompt).group(1)
            return kwargs['validator']({'typo': word[:-1] + 'x'})
    original = 'Replace the ornate bronze faucets with a black metal knocker.'
    text = generate_typos(Generator(), cfg, original, 'q1')
    assert len(word_tokens(original)) == len(word_tokens(text))
    assert sum(a != b for a, b in zip(word_tokens(original), word_tokens(text))) == cfg['perturbation']['typo_words']


def test_rephrasing_keeps_complex_target_alternatives_intact(cfg):
    from generation import generate_word_preserving
    class Generator:
        def request_json(self, *args, **kwargs):
            return kwargs['validator']({'modifier': 'Please swap the CONTENT0 for a CONTENT1.'})
    original = 'Replace the rings and bracelet with a gold-colored compact and a matching gold-colored comb, or a silver bangle with coiled details.'
    result = generate_word_preserving(Generator(), cfg, original, 'paraphrase', 'Rephrase')
    assert result == original.replace('Replace', 'Please swap').replace(' with a gold', ' for a gold')


def test_query_subset_never_reduces_gallery(cfg):
    frame = pd.read_csv(cfg['paths']['cir_file'])
    frame.loc[len(frame)] = dict(id1='c', id2='d', query='another pair', partition='test_ood')
    frame.to_csv(cfg['paths']['cir_file'], index=False)
    cfg['subset']['query_limit'] = 1
    queries = GLAMQueryDataset(cfg).records
    assert len(queries) == 1
    assert len(GLAMGalleryDataset(cfg, queries=queries)) == 5


@pytest.mark.parametrize('split', ['train', 'val', 'test_human'])
def test_only_test_id_and_test_ood_are_allowed(cfg, split):
    cfg['split'] = split
    with pytest.raises(ValueError, match='Only test_id and test_ood'):
        GLAMQueryDataset(cfg)
    with pytest.raises(ValueError, match='Only test_id and test_ood'):
        GLAMGalleryDataset(cfg)


def test_missing_gallery_distractor_cannot_silently_shrink_gallery(cfg):
    from pathlib import Path
    (Path(cfg['paths']['glam_root']) / 'test_ood/c.png').unlink()
    cfg['missing_gallery_policy'] = 'strict'
    with pytest.raises(FileNotFoundError, match='gallery has 1 missing'):
        GLAMGalleryDataset(cfg)


def test_batched_json_retries_only_failed_items_and_preserves_order():
    from generation import TextGenerator
    generator = TextGenerator.__new__(TextGenerator)
    generator.logger = logging.getLogger('batch-test')
    calls = []
    def generate(tasks, **kwargs):
        calls.append([task['prompt'] for task in tasks])
        return ['{"value": 1}', 'invalid'] if len(tasks) == 2 else ['{"value": 2}']
    generator.generate_many = generate
    result = generator.request_json_many([{'system': 's', 'prompt': 'one'}, {'system': 's', 'prompt': 'two'}],
                                         [lambda x: x['value']] * 2, attempts=2)
    assert result == [1, 2]
    assert len(calls[0]) == 2 and len(calls[1]) == 1 and calls[1][0].startswith('two')


def test_generation_uses_only_csv_images_and_every_listed_target(cfg, monkeypatch):
    import run
    import prompts
    caption_count = []
    class Generator:
        def __init__(self, *args):
            pass
        def request_json_many(self, tasks, validators, **kwargs):
            results = []
            for task, validator in zip(tasks, validators):
                if task['system'] == prompts.TARGET_DESCRIPTION_SYSTEM:
                    caption_count.append(1)
                    value = {'description': 'A ceramic bowl against a white background.'}
                else:
                    value = {'shared_traits': ['ceramic'], 'limitations': [],
                             'modifier': 'Replace the bowl with a blue vase.'}
                results.append(validator(value))
            return results
    cfg['generation']['quality_review'] = False
    monkeypatch.setattr(run, 'TextGenerator', Generator)
    manifest = prepare(cfg, logging.getLogger('pairs'))
    run.generate(cfg, logging.getLogger('pairs'))
    from utils import read_json
    observations = read_json(run.output_path(cfg, 'observations.json'))
    assert set(observations) == {'ref', 'a', 'b'}
    assert len(caption_count) == 3
    assert manifest['queries'][0]['conditioning_target_ids'] == ['a', 'b']
    assert len(manifest['gallery']) == 5


def test_test_membership_is_independent_of_physical_storage_folder(cfg):
    from pathlib import Path
    root = Path(cfg['paths']['glam_root'])
    (root / 'train').mkdir()
    (root / 'test_ood/a.png').rename(root / 'train/a.png')
    Image.new('RGB', (20, 20)).save(root / 'train/unrelated_training_image.png')
    with Path(cfg['paths']['db_processed_name']).open('a') as stream:
        stream.write('unrelated_training_image,train\n')
    cfg['image_storage_folders'] = ['test_ood', 'train']
    gallery = GLAMGalleryDataset(cfg, queries=GLAMQueryDataset(cfg).records)
    assert len(gallery) == 5
    assert 'unrelated_training_image' not in {r['image_id'] for r in gallery.records}


def test_all_csv_targets_are_conditioned_even_above_old_eight_image_limit(cfg):
    from pathlib import Path
    names = [f'z{i}' for i in range(9)]
    for name in names:
        Image.new('RGB', (20, 20)).save(Path(cfg['paths']['glam_root']) / f'test_ood/{name}.png')
    frame = pd.read_csv(cfg['paths']['cir_file'])
    frame.loc[len(frame)] = dict(id1='ref', id2=', '.join(names), query='many targets', partition='test_ood')
    frame.to_csv(cfg['paths']['cir_file'], index=False)
    with Path(cfg['paths']['db_processed_name']).open('a') as stream:
        stream.writelines(f'{name},test_ood\n' for name in names)
    manifest = prepare(cfg, logging.getLogger('all-targets'))
    row = next(r for r in manifest['queries'] if len(r['target_ids']) == 9)
    assert row['conditioning_target_ids'] == names
    assert len(manifest['gallery']) == 14


def test_safe_wording_candidates_preserve_visual_content(cfg):
    from generation import safe_rewrite_candidates, validate_perturbation
    original = 'Replace the red bowl with a blue vase.'
    candidates = safe_rewrite_candidates(cfg, original, 'informal')
    assert candidates
    for text in candidates:
        assert validate_perturbation({'modifier': text}, cfg, original, 'informal') == text


def test_original_availability_rule_uses_every_existing_gallery_image(cfg):
    from pathlib import Path
    (Path(cfg['paths']['glam_root']) / 'test_ood/c.png').unlink()
    cfg['missing_gallery_policy'] = 'available'
    gallery = GLAMGalleryDataset(cfg)
    assert len(gallery) == 4
    assert gallery.declared_size == 5
    assert gallery.missing_ids == ['c']


def test_restored_gallery_images_invalidate_an_incomplete_availability_snapshot(cfg):
    from pathlib import Path
    image = Path(cfg['paths']['glam_root']) / 'test_ood/c.png'
    image.unlink()
    cfg['missing_gallery_policy'] = 'available'
    prepare(cfg, logging.getLogger('restoration'))
    Image.new('RGB', (20, 20)).save(image)
    with pytest.raises(ValueError, match='Image availability changed'):
        load_manifest(cfg)


def test_parallel_merge_preserves_csv_order_and_rejects_changed_targets(cfg):
    import copy
    from parallel_perturbation import merge_records
    original = [dict(query_id=f'q{i}', reference_id='ref', target_ids=['a'],
                     modifier='Replace the red bowl with a blue vase.', perturbations={}, perturbation_reviews={}) for i in range(3)]
    completed = copy.deepcopy(original)
    for record in completed:
        record['perturbations'] = {'typos': 'Repalce the red bowl with a blue vase.',
                                  'informal': 'Please swap the red bowl for a blue vase.',
                                  'paraphrase': 'Substitute the red bowl with a blue vase.'}
    partitions = [completed[::2], completed[1::2]]
    assert [r['query_id'] for r in merge_records(original, partitions, cfg)] == ['q0', 'q1', 'q2']
    partitions[0][0]['target_ids'] = ['b']
    with pytest.raises(ValueError, match='changed'):
        merge_records(original, partitions, cfg)
