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
    cfg['image_gallery'] = ['test_ood']
    cfg['subset'] = {'query_limit': None, 'gallery_limit': 3}
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
    assert len(gallery) == 3
    assert {'a', 'b'} <= {r['image_id'] for r in gallery.records}
    assert gallery.records == GLAMGalleryDataset(cfg, queries=queries.records).records


def test_gallery_rejects_missing_targets_and_too_small_limit(cfg):
    queries = GLAMQueryDataset(cfg).records
    cfg['subset']['gallery_limit'] = 1
    with pytest.raises(ValueError, match='cannot fit'):
        GLAMGalleryDataset(cfg, queries=queries)
    cfg['subset']['gallery_limit'] = None
    queries[0]['target_ids'].append('not_in_gallery')
    with pytest.raises(ValueError, match='absent'):
        GLAMGalleryDataset(cfg, queries=queries)


def test_manifest_resume_and_config_change_guard(cfg):
    logger = logging.getLogger('test')
    manifest = prepare(cfg, logger)
    assert manifest['scope'] == 'subset'
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
