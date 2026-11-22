"""One seeded Python corruption per request and type, with fixed severity quotas."""

import math
import random
from collections import Counter

import nlpaug.augmenter.char as nac
import numpy as np

from utils import fingerprint

LIBRARY_ACTIONS = {'character_deletion': 'delete', 'character_insertion': 'insert'}


def assign_severities(cfg, count):
    """Round quotas by largest remainder, then shuffle their assignment to queries."""
    probabilities = cfg['perturbation']['severity_probabilities']
    expected = count * np.array(list(probabilities.values()))
    quotas = np.floor(expected).astype(int)
    order = np.argsort(-(expected - quotas), kind='stable')
    quotas[order[: count - quotas.sum()]] += 1
    levels = np.repeat(list(probabilities), quotas).tolist()
    random.Random(fingerprint([cfg['seed'], cfg['split'], 'severity'])).shuffle(levels)
    return levels


def edit_count(fraction, capacity):
    return min(capacity, max(1, math.ceil(fraction * capacity)))


def library_corruption(text, kind, fraction, rng, settings):
    """Use character tokens so NLPAug applies an exact whole-request edit budget."""
    options = dict(
        tokenizer=list,
        reverse_tokenizer=''.join,
        min_char=1,
        aug_word_p=0,
        aug_char_min=1,
        aug_char_max=1,
        aug_char_p=0,
    )

    if kind == 'character_substitution':
        augmenter = nac.KeyboardAug(
            **options,
            lang=settings['keyboard_language'],
            include_special_char=False,
            include_upper_case=False,
            include_numeric=True,
        )
        excluded = [char for char in set(text) if char not in augmenter.model.model]
    else:
        augmenter = nac.RandomCharAug(
            action=LIBRARY_ACTIONS[kind],
            **options,
            candidates=list(settings['insertion_characters']),
        )
        excluded = [char for char in set(text) if char.isspace()]
        
    capacity = sum(char not in excluded for char in text)
    count = edit_count(fraction, capacity)
    augmenter.stopwords = excluded
    augmenter.aug_word_min = augmenter.aug_word_max = count
    augmenter.sample = rng.sample  # Inject the query's RNG instead of a global sampler.
    return augmenter.augment(text, n=1)[0], capacity, count


def swap_sites(text, rng):
    """Choose disjoint adjacent, unequal letter/digit pairs within words."""
    candidates = [
        i for i in range(len(text) - 1) if text[i : i + 2].isalnum() and text[i] != text[i + 1]
    ]
    rng.shuffle(candidates)

    sites, occupied = [], set()
    for index in candidates:
        if occupied.isdisjoint([index, index + 1]):
            sites.append(index)
            occupied.update([index, index + 1])
    return sites


def simple_corruption(text, kind, fraction, rng):
    """Apply swaps, repetition, or spaces once using standard-library operations."""
    if kind == 'adjacent_character_transposition':
        sites = swap_sites(text, rng)
    elif kind == 'space_insertion':
        sites = [i for i in range(1, len(text)) if text[i - 1 : i + 1].isalnum()]
    elif kind == 'space_deletion':
        sites = [i for i, char in enumerate(text) if char == ' ']
    else:
        sites = [i for i, char in enumerate(text) if char.isalnum()]
    count = edit_count(fraction, len(sites))
    selected = set(rng.sample(sites, count))

    if kind == 'adjacent_character_transposition':
        chars = list(text)
        for index in selected:
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return ''.join(chars), len(sites), count
    
    if kind == 'space_deletion':
        replacements = dict.fromkeys(selected, '')
    elif kind == 'space_insertion':
        replacements = {i: ' ' + text[i] for i in selected}
    else:
        replacements = {i: text[i] * 2 for i in selected}
        
    return ''.join(replacements.get(i, char) for i, char in enumerate(text)), len(sites), count


def perturb_modifiers(cfg, records, logger):
    """Keep clean requests and targets; create one independently seeded error per type."""
    levels = assign_severities(cfg, len(records))
    records = [
        dict(record, severity=level, perturbations={}, perturbation_details={})
        for record, level in zip(records, levels)
    ]
    settings = cfg['perturbation']
    for record in records:
        for kind in settings['operations']:
            rng = random.Random(
                fingerprint([cfg['seed'], record['query_id'], kind, record['modifier']])
            )
            fraction = rng.uniform(*settings['edit_fraction'][record['severity']])
            if kind in LIBRARY_ACTIONS or kind == 'character_substitution':
                text, capacity, count = library_corruption(
                    record['modifier'], kind, fraction, rng, settings
                )
            else:
                text, capacity, count = simple_corruption(record['modifier'], kind, fraction, rng)
            record['perturbations'][kind] = text
            record['perturbation_details'][kind] = dict(
                requested_fraction=fraction, eligible_positions=capacity, edits=count
            )
    logger.info('Severity quotas: %s', dict(Counter(levels)))
    return records


def perturbation_summary(records, operations):
    statistics = {}
    for kind in operations:
        changed = sum(record['modifier'] != record['perturbations'][kind] for record in records)
        edits = [record['perturbation_details'][kind]['edits'] for record in records]
        statistics[kind] = dict(
            changed=changed, unchanged=len(records) - changed, mean_edits=float(np.mean(edits))
        )
    return statistics
