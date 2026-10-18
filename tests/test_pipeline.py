import logging
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from feature_extractor import FeatureExtractor, normalized_features
from generation import TextGenerator, parse_json, validate_modifier
from loading_data import GLAMGalleryDataset, GLAMQueryDataset
from main import (example_ids, generate_modifiers, load_data, load_modifiers,
                  perturb_modifiers, save_metrics)
from metrics_computation import compute_metrics, rank_queries
from perturbation import candidates, request, validate_perturbation
from utils import get_config, output_path


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg['runtime'].update(require_cuda=False, num_workers=0)
    cfg['paths'].update(glam_root=str(tmp_path/'raw'), cir_file=str(tmp_path/'cir.csv'),
        db_processed_name=str(tmp_path/'db.txt'), output_dir=str(tmp_path/'out'))
    cfg['image_storage_folders'] = ['test_ood', 'train']
    cfg['subset']['query_limit'] = None
    folder = tmp_path/'raw'/'test_ood'
    folder.mkdir(parents=True)
    for name in ['ref', 'a', 'b', 'c', 'd']:
        Image.new('RGB', (32, 40), 'white').save(folder/f'{name}.png')
    pd.DataFrame([
        dict(id1='ref', id2='a, b', query='old noisy query', partition='test_ood'),
        dict(id1='ref', id2='b,a', query='duplicate', partition='test_ood'),
        dict(id1='missing', id2='a', query='missing ref', partition='test_ood'),
    ]).to_csv(cfg['paths']['cir_file'], index=False)
    (tmp_path/'db.txt').write_text('ref,test_ood\na,test_ood\nb,test_ood\nc,test_ood\nd,test_ood\nz,train\n')
    return cfg


def test_csv_pairs_and_whole_gallery(cfg):
    q, g = load_data(cfg)
    assert len(q) == 1 and q[0]['target_ids'] == ['a', 'b']
    assert q[0]['original_modifier'] == 'old noisy query'
    assert len(g) == 5 and 'z' not in {r['image_id'] for r in g.records}
    assert GLAMQueryDataset(cfg)[0]['pil_image'].size == (32, 40)


@pytest.mark.parametrize('split', ['train', 'val'])
def test_absent_annotation_splits_rejected(cfg, split):
    cfg['split'] = split
    with pytest.raises(ValueError, match='Supported inference splits'):
        GLAMQueryDataset(cfg)


def test_missing_triplets_not_invented_from_gallery(cfg):
    cfg['split'] = 'test_id'
    with pytest.raises(ValueError, match='No test_id triplets'):
        load_data(cfg)


def test_image_metadata_is_not_query_annotations(cfg):
    pd.DataFrame([{'idInSource':'ref'}]).to_csv(cfg['paths']['cir_file'], index=False)
    with pytest.raises(ValueError, match='triplet columns'):
        load_data(cfg)


def test_query_limit_never_limits_gallery(cfg):
    cfg['subset']['query_limit'] = 1
    q, g = load_data(cfg)
    assert len(q) == 1 and len(g) == 5
    cfg['subset']['gallery_limit'] = 3
    with pytest.raises(ValueError, match='subsampling is forbidden'):
        load_data(cfg)


def test_all_csv_targets_including_large_groups(cfg):
    from pathlib import Path
    folder = Path(cfg['paths']['glam_root'])/'test_ood'
    ids = [f't{i}' for i in range(12)]
    for name in ids:
        Image.new('RGB', (20, 20)).save(folder/f'{name}.png')
    pd.DataFrame([dict(id1='ref', id2=','.join(ids), query='old', partition='test_ood')]).to_csv(cfg['paths']['cir_file'], index=False)
    with open(cfg['paths']['db_processed_name'], 'a') as f:
        f.write(''.join(f'{name},test_ood\n' for name in ids))
    q, g = load_data(cfg)
    assert q[0]['target_ids'] == ids and len(g) == 17


def test_storage_folder_is_not_annotation_partition(cfg):
    import pathlib
    source = pathlib.Path(cfg['paths']['glam_root'])/'test_ood'/'a.png'
    target = pathlib.Path(cfg['paths']['glam_root'])/'train'
    target.mkdir()
    source.rename(target/'a.png')
    q, g = load_data(cfg)
    assert q[0]['target_ids'] == ['a', 'b'] and len(g) == 5


def test_missing_gallery_available_and_strict(cfg):
    with open(cfg['paths']['db_processed_name'], 'a') as f:
        f.write('absent,test_ood\n')
    q, g = load_data(cfg)
    assert g.missing_ids == ['absent'] and g.declared_size == 6 and len(g) == 5
    cfg['missing_gallery_policy'] = 'strict'
    with pytest.raises(FileNotFoundError):
        GLAMGalleryDataset(cfg, queries=q)


def test_targets_must_be_in_complete_split_gallery(cfg):
    q = GLAMQueryDataset(cfg).records
    q[0]['target_ids'].append('other')
    with pytest.raises(ValueError, match='absent'):
        GLAMGalleryDataset(cfg, queries=q)


def test_modifier_hard_limit_and_json(cfg):
    text = 'Replace the red bowl with a tall blue vase.'
    assert validate_modifier(parse_json('```json\n{"modifier": "'+text+'"}\n```'), cfg) == text
    with pytest.raises(ValueError, match='words'):
        validate_modifier({'modifier': ' '.join(['vase']*16)}, cfg)
    with pytest.raises(ValueError, match='JSON'):
        parse_json('nothing')


@pytest.mark.parametrize('kind', ['deletion', 'capitalization', 'character_swap', 'keyboard_typo'])
def test_exact_corruption_operation_and_llm_choice(kind):
    original = 'Replace the blue bowl with a tall white vase.'
    task, check = request(original, kind)
    text = check({'choice': 0})
    assert text != original and text in candidates(original, kind)
    with pytest.raises(ValueError):
        validate_perturbation('Replace the bowl with a black vase.', original, kind)
    with pytest.raises(ValueError):
        check({'choice': True})


def test_deletion_is_one_internal_letter_and_preserves_negation():
    original = 'Replace the red bowl with a blue vase without handles.'
    for option in candidates(original, 'deletion'):
        assert len(option) == len(original)-1
        assert 'without' in option and len(option.split()) == len(original.split())
        assert any(original[:i] + original[i+1:] == option for i in range(1, len(original)-1))
    assert candidates('Add inscriptions and crests to coins.', 'deletion')


def test_batched_retry_preserves_successes(cfg):
    generator = TextGenerator.__new__(TextGenerator)
    generator.cfg = cfg
    generator.logger = logging.getLogger('test')
    calls = []
    def generate(tasks):
        calls.append(len(tasks))
        return ['{"modifier":"ok"}', 'not JSON'] if len(calls) == 1 else ['{"modifier":"fixed"}']
    generator.generate_many = generate
    result = generator.request_json_many([dict(system='',prompt='')]*2, [lambda r:r['modifier']]*2)
    assert result == ['ok', 'fixed'] and calls == [2, 1]


def test_projected_vectors_and_modality_normalization(cfg):
    vectors = torch.tensor([[3., 4.]])
    torch.testing.assert_close(normalized_features(SimpleNamespace(pooler_output=vectors)), torch.tensor([[.6, .8]]))
    with pytest.raises(ValueError):
        normalized_features(torch.zeros(2, 4))
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.cfg = cfg
    extractor.get_text_features = lambda _: torch.tensor([[0., 1.]])
    torch.testing.assert_close(extractor.compose(torch.tensor([[20., 0.]]), ['x']), torch.tensor([[2**-.5, 2**-.5]]))


def test_multitarget_metrics_and_reference_mask():
    q = [dict(query_id='q', reference_id='ref', target_ids=['a', 'b'])]
    ranks, _ = rank_queries(torch.tensor([[1., 0.]]), torch.tensor([[1., 0.], [.8, .2], [0., 1.]]), ['ref', 'a', 'b'], q)
    metrics, _ = compute_metrics(ranks, q, [1, 5])
    assert ranks == [['a', 'b']] and metrics['Recall@1'] == 50 and metrics['HitRate@1'] == 100
    assert metrics['Recall@5'] == 100


def test_composed_query_cosines_are_unit_and_scale_invariant(cfg):
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.cfg = cfg
    extractor.get_text_features = lambda _: torch.tensor([[0., 1., 0.]])
    query = extractor.compose(torch.tensor([[30., 0., 0.]]), ['modifier'])
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(1))
    gallery = torch.tensor([[10., 10., 0.], [0., 40., 0.], [-12., -12., 0.]])
    queries = [dict(query_id='q', reference_id='ref', target_ids=['a'])]
    ranks, scores = rank_queries(query, gallery, ['a', 'b', 'c'], queries)
    torch.testing.assert_close(scores, torch.tensor([[1., 2**-.5, -1.]]))
    scaled_ranks, scaled_scores = rank_queries(query * 9, gallery * torch.tensor([[3.], [.25], [7.]]),
                                              ['a', 'b', 'c'], queries)
    assert ranks == scaled_ranks == [['a', 'b', 'c']]
    torch.testing.assert_close(scores, scaled_scores)


def test_reduced_outputs_and_all_components(cfg, capsys):
    q, g = load_data(cfg)
    data = load_modifiers(cfg, q, g)
    class Generator:
        def request_json_many(self, tasks, validators):
            outputs = []
            for task, check in zip(tasks, validators):
                if 'Options:' in task['prompt']:
                    result = {'choice':0}
                elif 'BEFORE:' in task['prompt']:
                    assert task['prompt'].count('a white vase') == 3  # reference plus both exact CSV targets
                    result = {'modifier':'Replace the white bowl with a tall blue vase.'}
                else:
                    result = {'description':'a white vase'}
                outputs.append(check(result))
            return outputs
    generator = Generator()
    generate_modifiers(cfg, q, data, generator)
    perturb_modifiers(cfg, q, data, generator)
    assert set(data['records'][0]['perturbations']) == set(cfg['perturbation']['types'])
    metrics = {'unperturbed':{k:0 for k in ['Recall@1','Recall@5','Recall@10','Recall@50','AVG','mAP','MRR']}}
    save_metrics(cfg, metrics, q, g)
    assert {p.name for p in output_path(cfg, '').iterdir()} == {'modifiers.json', 'metrics.json'}
    assert '| unperturbed |' in capsys.readouterr().out
    assert load_modifiers(cfg, q, g)['records'] == data['records']
    cfg['seed'] += 1
    with pytest.raises(ValueError, match='changed'):
        load_modifiers(cfg, q, g)


def test_seeded_examples_and_image_size(cfg):
    from figure import display_image
    q, _ = load_data(cfg)
    assert example_ids(cfg, q) == example_ids(cfg, q)
    assert display_image(q[0]['reference_path'], 224).size == (224, 224)


def test_concise_clauses_preserve_target_and_logic():
    from generation import concise_candidates
    original = 'Replace the old reference object with a blue vase without handles standing on a wooden table.'
    options = concise_candidates(original, 15)
    assert options
    for text in options:
        assert len(text.split()) <= 15
        assert 'blue vase without handles' in text
        assert text.endswith('.') and not text.endswith('standing.')
    # Never lose an alternative target hidden in a later clause.
    original = 'Replace the old reference object with a woman in a floral dress or a man in a black jacket.'
    for text in concise_candidates(original, 15):
        assert ' or ' in text


def test_overlong_modifier_uses_llm_clause_choice(cfg):
    from generation import concise_candidates
    generator = TextGenerator.__new__(TextGenerator)
    generator.cfg, generator.logger = cfg, logging.getLogger('concise')
    text = 'Replace the round ceramic bowl with a tall blue vase with a narrow neck standing on a wooden table.'
    calls = []
    def generate(tasks):
        calls.append(tasks[0]['field'])
        return ['{"modifier": "' + text + '"}'] if tasks[0]['field'] == 'modifier' else ['{"choice": 0}']
    generator.generate_many = generate
    result = generator.request_json_many([{'field':'modifier','system':'','prompt':''}], [lambda r:validate_modifier(r,cfg)])
    assert calls == ['modifier', 'choice']
    assert result[0] in concise_candidates(text, 15)
    assert 'blue vase' in result[0] and len(result[0].split()) <= 15
