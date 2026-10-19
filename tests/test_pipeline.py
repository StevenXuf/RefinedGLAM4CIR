import logging
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from feature_extractor import FeatureExtractor, normalized_features
from generation import (TextGenerator, generate_grounded_modifiers, parse_json,
                        validate_modifier)
from loading_data import GLAMGalleryDataset, GLAMQueryDataset
from main import (example_ids, generate_modifiers, generation_targets, load_data, load_modifiers,
                  perturb_modifiers, query_features, reuse_modifiers, save_metrics)
from metrics_computation import compute_metrics, rank_queries
from perturbation import (KINDS, NEIGHBOURS, candidates, request, sample_profile,
                          validate_perturbation, validate_settings)
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


@pytest.mark.parametrize('kind', list(KINDS))
def test_exact_corruption_operation_and_llm_choice(cfg, kind):
    original = 'Replace the blue bowl with a tall white vase.'
    task, check = request(original, kind, cfg, 'query')
    text = check({'choice': 0})
    assert text != original and text in candidates(original, kind, cfg, 'query')
    with pytest.raises(ValueError):
        validate_perturbation('Replace the bowl with a black vase.', original, kind, cfg, 'query')
    with pytest.raises(ValueError):
        check({'choice': True})


def test_deletion_budget_and_preserved_negation(cfg):
    original = 'Replace the red bowl with a blue vase without handles.'
    budget = sample_profile(original, 'character_deletion', cfg)['edits']
    for option in candidates(original, 'character_deletion', cfg):
        assert len(option) == len(original)-budget
        assert 'without' in option and len(option.split()) == len(original.split())
    assert candidates('Add inscriptions and crests to coins.', 'character_deletion', cfg)


@pytest.mark.parametrize('kind', list(KINDS))
def test_severities_are_seeded_and_edits_stay_in_the_named_type(cfg, kind):
    original = 'Replace the red carved bowl with a taller blue vase without handles.'
    from collections import Counter
    choices = candidates(original, kind, cfg, 'q17')
    assert choices == candidates(original, kind, cfg, 'q17')
    budget = sample_profile(original, kind, cfg, 'q17')['edits']
    for text in choices:
        if kind == 'capitalization':
            assert text.lower() == original.lower()
            assert sum(c.isupper() for c in text) == budget
        elif kind in {'character_insertion', 'character_repetition', 'space_insertion'}:
            assert len(text) == len(original) + budget
            if kind == 'space_insertion':
                assert text.replace(' ', '') == original.replace(' ', '')
            elif kind == 'character_repetition':
                import re
                assert re.sub(r'(.)\1+', r'\1', text) == re.sub(r'(.)\1+', r'\1', original)
        elif kind in {'character_deletion', 'space_deletion'}:
            assert len(text) == len(original) - budget
            if kind == 'space_deletion':
                assert text.replace(' ', '') == original.replace(' ', '')
        elif kind == 'adjacent_character_transposition':
            assert Counter(text) == Counter(original)
            assert sum(a != b for a, b in zip(text, original)) == 2 * budget
        elif kind == 'character_substitution':
            assert len(text) == len(original)
            changed = [(a, b) for a, b in zip(original, text) if a != b]
            assert len(changed) == budget
            assert all(b.lower() in NEIGHBOURS[a.lower()] for a, b in changed)


def test_distribution_has_mostly_mild_cases_and_rare_extremes(cfg):
    from collections import Counter
    original = 'Replace the carved wooden bowl with a blue ceramic vase without handles.'
    profiles = [sample_profile(original, 'character_deletion', cfg, str(i)) for i in range(10000)]
    counts = Counter(p['severity'] for p in profiles)
    assert .72 < counts['mild']/10000 < .78
    assert .005 < counts['extreme']/10000 < .015
    assert max(p['edits'] for p in profiles) > 10
    profiles = [sample_profile(original, 'capitalization', cfg, str(i)) for i in range(10000)]
    counts = Counter(p['severity'] for p in profiles)
    assert .97 < counts['mixed']/10000 < .99
    assert .005 < counts['lowercase']/10000 < .015
    assert .005 < counts['uppercase']/10000 < .015
    letters = sum(c.isalpha() for c in original)
    mixed = [p['edits']/letters for p in profiles if p['severity'] == 'mixed']
    assert .08 < sum(mixed)/len(mixed) < .13


def test_extreme_profiles_remain_valid_and_have_exact_budgets(cfg):
    cfg['perturbation']['severity_probabilities'] = dict(mild=0., moderate=0., severe=0., extreme=1.)
    cfg['perturbation']['edit_fraction']['extreme'] = [1., 1.]
    for kind in KINDS:
        for original in ['Replace the blue bowl with a wooden vase.', 'Add inscriptions and crests to coins.']:
            task, check = request(original, kind, cfg, 'extreme')
            assert check({'choice': 0}) != original
    cfg['perturbation']['capitalization']['mode_probabilities'] = dict(mixed=0., lowercase=0., uppercase=1.)
    assert candidates(original, 'capitalization', cfg) == [original.upper()]
    cfg['perturbation']['capitalization']['mode_probabilities'] = dict(mixed=0., lowercase=1., uppercase=0.)
    assert candidates(original, 'capitalization', cfg) == [original.lower()]


def test_keyboard_neighbours_include_adjacent_rows():
    assert 'e' in NEIGHBOURS['d'] and 'c' in NEIGHBOURS['d']
    assert 'f' in NEIGHBOURS['d'] and 'p' not in NEIGHBOURS['d']


def test_candidate_locations_are_not_biased_toward_sentence_start(cfg):
    cfg['perturbation']['candidate_count'] = 1
    cfg['perturbation']['severity_probabilities'] = dict(mild=1., moderate=0., severe=0., extreme=0.)
    cfg['perturbation']['edit_fraction']['mild'] = [.001, .001]
    original = 'AbcdefghijkL MnopqrstuvwX Yz0123456789'
    late = 0
    for i in range(1000):
        text = candidates(original, 'character_deletion', cfg, str(i))[0]
        late += text.startswith('AbcdefghijkL MnopqrstuvwX ')
    assert .25 < late/1000 < .42


def test_bad_distribution_configuration_rejected(cfg):
    cfg['perturbation']['severity_probabilities']['extreme'] = .5
    with pytest.raises(ValueError, match='sum to 1'):
        validate_settings(cfg['perturbation'])


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
    metrics, _ = compute_metrics(ranks, q, [1, 5], ['ref', 'a', 'b'])
    assert ranks == [['a', 'b']] and metrics['Recall@1'] == 50 and metrics['HitRate@1'] == 100
    assert metrics['Recall@5'] == 100


def test_torchmetrics_multi_positive_macro_and_missing_targets():
    queries = [dict(query_id='q1', reference_id='ref', target_ids=['a', 'b', 'not_downloaded']),
               dict(query_id='q2', reference_id='ref', target_ids=['c'])]
    metrics, rows = compute_metrics([['c', 'a', 'd', 'b'], ['c', 'a', 'b', 'd']], queries, [1, 2, 4], ['a', 'b', 'c', 'd'])
    assert len(rows) == 2
    assert metrics == pytest.approx({'MRR':75, 'mAP':75, 'Recall@1':50, 'Recall@2':75, 'Recall@4':100,
                                     'HitRate@1':50, 'HitRate@2':100, 'HitRate@4':100, 'AVG':75})


def test_missing_all_targets_skipped_but_truncated_rankings_rejected():
    q = [dict(query_id='missing', reference_id='ref', target_ids=['absent']),
         dict(query_id='valid', reference_id='ref', target_ids=['a'])]
    metrics, rows = compute_metrics([['b', 'a'], ['b', 'a']], q, [1, 2], ['a', 'b'])
    assert [r['query_id'] for r in rows] == ['valid'] and metrics['mAP'] == 50
    with pytest.raises(ValueError, match='complete available gallery'):
        compute_metrics([['b'], ['b']], q, [1], ['a', 'b'])
    with pytest.raises(ValueError, match='No queries'):
        compute_metrics([['a', 'b']], q[:1], [1], ['a', 'b'])


def test_negative_cosines_and_stable_ties_keep_relevant_candidates():
    q = [dict(query_id='q', reference_id='ref', target_ids=['a'])]
    rankings, scores = rank_queries(torch.tensor([[1., 0.]]), torch.tensor([[-1., 0.], [-.8, -.6]]), ['a','b'], q)
    assert (scores < 0).all()
    metrics, _ = compute_metrics(rankings, q, [1, 2], ['a', 'b'])
    assert metrics['mAP'] == 50 and metrics['MRR'] == 50 and metrics['Recall@2'] == 100
    ranks, _ = rank_queries(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.], [0., 1.]]), ['a','b'], q)
    assert ranks == [['a', 'b']]
    metrics, _ = compute_metrics(ranks, q, [1], ['a', 'b'])
    assert metrics['mAP'] == 100 and metrics['Recall@1'] == 100


def test_partially_missing_target_group_keeps_remaining_target(cfg):
    from pathlib import Path
    (Path(cfg['paths']['glam_root'])/'test_ood'/'b.png').unlink()
    q, g = load_data(cfg)
    assert len(q) == 1 and q[0]['target_ids'] == ['a']
    assert len(g) == 4
    ranks = [[r['image_id'] for r in g.records if r['image_id'] != 'ref']]
    metrics, _ = compute_metrics(ranks, q, [1, 5], [r['image_id'] for r in g.records])
    assert metrics['Recall@1'] == 100 and metrics['mAP'] == 100


def test_baseline_queries_use_the_correct_modalities(cfg):
    import torch.nn.functional as F
    calls = []
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.cfg = cfg
    def text_features(texts):
        calls.append(texts)
        return torch.tensor([[0., 1., 0.]]) if texts == ['original'] else torch.tensor([[0., 0., 1.]])
    extractor.get_text_features = text_features
    reference = torch.tensor([[20., 0., 0.]])
    records = [dict(original_modifier='original', modifier='generated', perturbations={'character_deletion':'perturbed'})]
    torch.testing.assert_close(query_features(extractor, reference, records, 'reference_only'), torch.tensor([[1., 0., 0.]]))
    assert not calls
    torch.testing.assert_close(query_features(extractor, reference, records, 'original_text_only'), torch.tensor([[0., 1., 0.]]))
    torch.testing.assert_close(query_features(extractor, reference, records, 'generated_text_only'), torch.tensor([[0., 0., 1.]]))
    torch.testing.assert_close(query_features(extractor, reference, records, 'reference_original'), F.normalize(torch.tensor([[1., 1., 0.]]), dim=-1))
    torch.testing.assert_close(query_features(extractor, reference, records, 'unperturbed'), F.normalize(torch.tensor([[1., 0., 1.]]), dim=-1))
    assert calls == [['original'], ['generated'], ['original'], ['generated']]


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
                elif task.get('field') == 'shared_features':
                    result = {'shared_features':'a white vase'}
                elif task.get('field') == 'coverage':
                    result = {'coverage':'Y', 'direction_correct':True}
                elif 'BEFORE:' in task['prompt']:
                    assert task['prompt'].count('a white vase') == 2  # reference and all-target shared phrase
                    result = {'modifier':'Replace the white bowl with a tall blue vase.'}
                else:
                    result = {'description':'a white vase'}
                outputs.append(check(result))
            return outputs
    generator = Generator()
    generate_modifiers(cfg, q, data, generator)
    perturb_modifiers(cfg, q, data, generator)
    assert set(data['records'][0]['perturbations']) == set(cfg['perturbation']['types'])
    metrics = {'unperturbed':{k:0 for k in ['Recall@1','Recall@5','Recall@10','Recall@50',
                                           'HitRate@1','HitRate@5','HitRate@10','HitRate@50','AVG','mAP','MRR']}}
    save_metrics(cfg, metrics, q, g)
    assert {p.name for p in output_path(cfg, '').iterdir()} == {'modifiers.json', 'metrics.json'}
    output = capsys.readouterr().out
    assert '| unperturbed |' in output and all(key in output for key in metrics['unperturbed'])
    assert load_modifiers(cfg, q, g)['records'] == data['records']
    cfg['seed'] += 1
    with pytest.raises(ValueError, match='changed'):
        load_modifiers(cfg, q, g)


def test_clean_reuse_requires_same_model_and_exact_csv_triplets(cfg, tmp_path):
    from utils import save_json
    q, g = load_data(cfg)
    old = load_modifiers(cfg, q, g)
    modifier = 'Replace the white bowl with a tall blue vase.'
    old['records'] = [dict(q[0], generation_target_ids=generation_targets(cfg, q[0]), modifier=modifier, perturbations={'keyboard_typo': 'obsolete'})]
    source = tmp_path/'previous.json'
    save_json(source, old)
    data = load_modifiers(cfg, q, g)
    reuse_modifiers(cfg, q, data, source)
    assert data['records'] == [dict(q[0], generation_target_ids=generation_targets(cfg, q[0]), modifier=modifier, perturbations={})]
    assert load_modifiers(cfg, q, g)['clean_source']['content_hash']
    with pytest.raises(ValueError, match='empty'):
        reuse_modifiers(cfg, q, data, source)
    old['records'][0]['target_ids'] = ['c']
    save_json(source, old)
    with pytest.raises(ValueError, match='CSV triplet'):
        reuse_modifiers(cfg, q, dict(records=[]), source)
    old['generation_model'] = dict(id='other')
    save_json(source, old)
    with pytest.raises(ValueError, match='same split and generation'):
        reuse_modifiers(cfg, q, dict(records=[]), source)


def test_seeded_examples_and_image_size(cfg):
    from figure import display_image
    q, _ = load_data(cfg)
    assert example_ids(cfg, q) == example_ids(cfg, q)
    assert display_image(q[0]['reference_path'], 224).size == (224, 224)


def test_partial_checkpoint_import_keeps_the_full_query_cohort(cfg, tmp_path):
    from utils import save_json
    cfg['remove_duplicates'] = False
    queries, gallery = load_data(cfg)
    assert len(queries) == 2
    data = load_modifiers(cfg, queries, gallery)
    old = dict(data, records=[dict(queries[0], generation_target_ids=generation_targets(cfg, queries[0]),
                                  modifier='Replace it with a ceramic vase.', perturbations={})])
    source = tmp_path / 'partial.json'
    save_json(source, old)
    reuse_modifiers(cfg, queries, data, source)
    assert len(data['records']) == 1
    assert len(load_data(cfg)[0]) == 2
    assert load_modifiers(cfg, queries, gallery)['records'] == data['records']


def test_original_baseline_is_first_and_capitalization_is_not_plotted(cfg):
    from figure import figure_conditions
    item = dict(original_modifier='original', modifier='generated',
                perturbations={k:k for k in cfg['perturbation']['types']})
    conditions = figure_conditions(cfg, item)
    assert conditions[:2] == [('reference_original','original'), ('unperturbed','generated')]
    assert 'capitalization' not in {c for c, _ in conditions}
    assert len(conditions) == 9


def test_generation_samples_three_without_changing_metric_ground_truth(cfg):
    import copy
    record = dict(query_id='q', target_ids=[f'image_{i}' for i in range(20)])
    before = copy.deepcopy(record)
    selected = generation_targets(cfg, record)
    assert len(selected) == len(set(selected)) == 3
    assert set(selected) <= set(record['target_ids'])
    assert selected == generation_targets(cfg, record)
    assert record == before
    assert len(record['target_ids']) == 20
    record['target_ids'] = ['a', 'b']
    assert generation_targets(cfg, record) == ['a', 'b']


def test_figure_highlights_ground_truth_outside_generation_sample(cfg, monkeypatch):
    from figure import figure_conditions, visualize
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    queries, gallery = load_data(cfg)
    record = dict(queries[0], generation_target_ids=['a'], modifier='Replace it with a ceramic vase.',
                  perturbations={kind:'Replace it with a ceramic vase.' for kind in cfg['perturbation']['types']})
    cfg['visualization']['top_k'] = 2
    retrieval = {condition: {record['query_id']: [dict(image_id='b', score=.8), dict(image_id='c', score=.6)]}
                 for condition, _ in figure_conditions(cfg, record)}
    saved = []
    def inspect(fig, *args, **kwargs):
        saved.append(True)
        assert fig._suptitle is None
        sizes = [(ax.get_window_extent().width, ax.get_window_extent().height) for ax in fig.axes]
        for width, height in sizes:
            assert width == pytest.approx(height)
            assert width == pytest.approx(sizes[0][0])
        for offset in range(0, len(fig.axes), 5):
            assert all(spine.get_visible() and spine.get_edgecolor() == to_rgba('#20945d')
                       for spine in fig.axes[offset+3].spines.values())
            assert not any(spine.get_visible() for spine in fig.axes[offset+4].spines.values())
    monkeypatch.setattr(Figure, 'savefig', inspect)
    visualize(cfg, [record], gallery.records, retrieval)
    assert saved == [True]


def test_every_target_is_reviewed_and_minority_mismatch_forces_broader_modifier(cfg):
    seen = []
    class Generator:
        attempts = 0
        def request_json_many(self, tasks, validators):
            outputs = []
            for task, validate in zip(tasks, validators):
                seen.append(task)
                if task['field'] == 'modifier':
                    self.attempts += 1
                    result = {'modifier': 'Replace the bowl with a tall blue vase.' if self.attempts == 1 else
                              'Replace the bowl with a tall ceramic vase.'}
                else:
                    result = {'coverage':'N' if self.attempts == 1 and 'green vase' in task['prompt'] else 'Y',
                              'direction_correct':True, 'reason':'Blue excludes the green target'}
                outputs.append(validate(result))
            return outputs
    result = generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['a blue vase','a green vase']], ['ceramic vases'])
    assert result == ['Replace the bowl with a tall ceramic vase.']
    reviews = [task for task in seen if task['field'] == 'coverage' and 'SINGLE TARGET:' in task['prompt']]
    assert len(reviews) == 4 and any('green vase' in task['prompt'] for task in reviews)
    assert all('BEFORE:' not in task['prompt'] for task in reviews)
    assert 'Failed target reviews' in [t for t in seen if t['field'] == 'modifier'][1]['prompt']
    with pytest.raises(ValueError, match='shared by every target'):
        validate_modifier({'modifier':'Replace the bowl with a blue or green vase.'}, cfg)


def test_rejected_target_review_never_returns_an_unchecked_modifier(cfg):
    class Generator:
        def request_json_many(self, tasks, validators):
            return [check({'modifier':'Replace the white bowl with a tall blue vase.'} if task['field'] == 'modifier'
                          else {'choice':0} if task['field'] == 'choice' else {'coverage':'N', 'reason':'Wrong category'})
                    for task, check in zip(tasks, validators)]
    result = generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['a blue vase', 'a green vase']], ['ceramic vases'])
    assert isinstance(result[0], ValueError)


def test_target_review_excludes_the_replacement_source():
    from generation import target_requirements
    assert target_requirements('Replace dragon heads with metal animal heads on stone walls') == 'metal animal heads on stone walls'
    assert target_requirements('Change the bowl into a ceramic vase.') == 'a ceramic vase.'
    assert target_requirements('Make it blue.') == 'Make it blue.'


def test_repeated_rejection_uses_a_reviewed_common_category(cfg):
    import prompts
    jobs = []
    class Generator:
        broad = False
        def request_json_many(self, tasks, validators):
            results = []
            for task, check in zip(tasks, validators):
                jobs.append(task)
                if task['field'] == 'modifier':
                    self.broad = task['system'] == prompts.BROAD_MODIFIER
                    value = {'modifier':'Find an image of an artifact.' if self.broad else 'Find an image of a blue vase.'}
                else:
                    value = {'coverage':'Y' if self.broad else 'N', 'reason':'Blue is not common'}
                results.append(check(value))
            return results
    result = generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['a blue vase', 'a green statue']], ['blue vases'])
    assert result == ['Find an image of an artifact.']
    modifier_jobs = [task for task in jobs if task['field'] == 'modifier']
    assert len(modifier_jobs) == 3
    assert 'blue vases' not in modifier_jobs[-1]['prompt']
    assert all(observation in modifier_jobs[-1]['prompt'] for observation in ['a blue vase', 'a green statue'])
    assert len([task for task in jobs if task['field'] == 'coverage']) == 6


def test_format_failure_also_discards_the_incorrect_shared_summary(cfg):
    import prompts
    class Generator:
        def request_json_many(self, tasks, validators):
            results = []
            for task, check in zip(tasks, validators):
                if task['field'] == 'modifier':
                    if task['system'] != prompts.BROAD_MODIFIER:
                        results.append(ValueError('Do not enumerate alternatives using or'))
                        continue
                    assert 'INDIVIDUAL TARGET IMAGES' in task['prompt']
                    assert 'incorrect summary' not in task['prompt']
                    results.append(check({'modifier':'Find an image of an artifact.'}))
                else:
                    results.append(check({'coverage':'Y'}))
            return results
    assert generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['a vase', 'a statue']], ['incorrect summary']) == ['Find an image of an artifact.']


def test_failed_shared_summary_uses_individual_observations(cfg):
    seen = []
    class Generator:
        def request_json_many(self, tasks, validators):
            for task in tasks:
                if task['field'] == 'modifier':
                    seen.append(task['prompt'])
            return [check({'modifier':'Find an image of an artifact.'} if task['field'] == 'modifier' else {'coverage':'Y'})
                    for task, check in zip(tasks, validators)]
    assert generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['a vase', 'a statue']], [None]) == ['Find an image of an artifact.']
    assert all(text in seen[0] for text in ['INDIVIDUAL TARGET IMAGES', 'a vase', 'a statue'])
    assert 'None' not in seen[0]


def test_visual_review_sees_actual_targets_without_incomplete_captions(cfg):
    import prompts
    seen = []
    queries, _ = load_data(cfg)
    class Generator:
        def request_json_many(self, tasks, validators):
            seen.extend(tasks)
            return [check({'modifier':'Find an image of a figurine.'} if task['field'] == 'modifier' else {'coverage':'Y'})
                    for task, check in zip(tasks, validators)]
    result = generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['unknown triangular object', 'an elongated object']],
                                        ['figurines'], target_paths=[queries[0]['target_paths']])
    assert result == ['Find an image of a figurine.']
    reviews = [task for task in seen if task['field'] == 'coverage']
    assert len(reviews) == 2
    assert all(task['system'] == prompts.VISUAL_TARGET_REVIEW and len(task['images']) == 1 for task in reviews)
    assert all('triangular' not in task['prompt'] and 'elongated' not in task['prompt'] for task in reviews)
    assert all(task['images'][0].size == (32, 40) for task in reviews)


def test_final_fallback_applies_the_exact_chosen_request_and_reviews_it(cfg):
    class Generator:
        fallback = False
        def request_json_many(self, tasks, validators):
            results = []
            for task, check in zip(tasks, validators):
                if task['field'] == 'choice':
                    self.fallback = True
                    value = {'choice':1}
                elif task['field'] == 'modifier':
                    value = {'modifier':'Find an image of a golden chair.'}
                else:
                    if self.fallback:
                        assert cfg['generation']['fallback_modifiers'][1] in task['prompt']
                    value = {'coverage':'Y' if self.fallback else 'N', 'reason':'No gold visible'}
                results.append(check(value))
            return results
    result = generate_grounded_modifiers(Generator(), cfg, ['a bowl'], [['a wooden chair']], ['chairs'])
    assert result == [cfg['generation']['fallback_modifiers'][1]]


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
