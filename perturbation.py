"""Seeded severity sampling, MLLM-selected corruptions, and exact validation."""
import bisect
import hashlib
import math
import random
import re

from prompts import PERTURBATION

KINDS = {
    'character_deletion': 'Delete characters inside words, simulating missed keystrokes.',
    'character_insertion': 'Insert extra characters inside words.',
    'character_substitution': 'Substitute characters with neighbouring English QWERTY keys.',
    'adjacent_character_transposition': 'Transpose adjacent characters inside words.',
    'character_repetition': 'Repeat characters, simulating held or repeated keystrokes.',
    'space_insertion': 'Insert spaces inside words.',
    'space_deletion': 'Delete spaces between words.',
}
PROTECTED = {'no', 'not', 'without', 'or', 'and', 'nor'}
# Staggered QWERTY rows include horizontal, vertical and diagonal neighbours.
KEY_ROWS = [('1234567890', 0.), ('qwertyuiop', .25), ('asdfghjkl', .5), ('zxcvbnm', 1.)]
KEYS = {char: (i + offset, row) for row, (letters, offset) in enumerate(KEY_ROWS) for i, char in enumerate(letters)}
NEIGHBOURS = {char: ''.join(sorted(other for other, (u, v) in KEYS.items()
    if other != char and (x-u)**2 + (y-v)**2 <= 1.45**2)) for char, (x, y) in KEYS.items()}


def validate_settings(settings):
    probabilities = settings['severity_probabilities']
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1.):
        raise ValueError('Perturbation probabilities must be finite, nonnegative, and sum to 1')
    if set(settings['severity_probabilities']) != set(settings['edit_fraction']):
        raise ValueError('Each severity needs an edit_fraction range')
    if any(len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] <= 1 for bounds in settings['edit_fraction'].values()):
        raise ValueError('Edit fraction ranges must satisfy 0 < minimum <= maximum <= 1')
    if type(settings['candidate_count']) is not int or not 1 <= settings['candidate_count'] <= 64:
        raise ValueError('candidate_count must be an integer between 1 and 64')
    alphabet = settings['insertion_characters']
    if not isinstance(alphabet, str) or not re.fullmatch(r'[A-Za-z0-9]+', alphabet) or len(set(alphabet)) != len(alphabet):
        raise ValueError('insertion_characters must be distinct ASCII letters/digits')


def edit_sites(original, kind, alphabet):
    """Original-string spans and allowed replacements; sites never edit logical words."""
    words = list(re.finditer(r'[A-Za-z0-9]+', original))
    if kind == 'space_deletion':
        protected = [m.span() for m in words if m.group().lower() in PROTECTED]
        return [(m.start(), m.end(), ['']) for m in re.finditer(r' +', original)
                if not any(end == m.start() or start == m.end() for start, end in protected)]
    sites = []
    for match in words:
        if match.group().lower() in PROTECTED or len(match.group()) < 4:
            continue
        start, end = match.span()
        if kind in {'character_insertion', 'space_insertion'}:
            sites.extend((i, i, list(alphabet) if kind == 'character_insertion' else [' '])
                         for i in range(start+1, end))
            continue
        for i in range(start+1, end-1):
            char = original[i]
            if kind == 'character_deletion':
                replacements = ['']
            elif kind == 'character_substitution':
                replacements = [c.upper() if char.isupper() else c for c in NEIGHBOURS.get(char.lower(), '')]
            elif kind == 'character_repetition':
                replacements = [char * 2]
            elif kind == 'adjacent_character_transposition' and i+1 < end-1 and char != original[i+1]:
                sites.append((i, i+2, [original[i:i+2][::-1]]))
                continue
            else:
                continue
            if replacements:
                sites.append((i, i+1, replacements))
    return sites


def applicable(original, kind):
    if kind not in KINDS:
        raise ValueError(f'Unknown perturbation: {kind}')
    return bool(edit_sites(original, kind, 'a'))


def site_capacity(sites):
    """Maximum nonoverlapping edits in every suffix, including adjacent swaps."""
    starts = [s[0] for s in sites]
    following = [bisect.bisect_left(starts, end, lo=i+1) for i, (_, end, _) in enumerate(sites)]
    capacity = [0] * (len(sites)+1)
    for i in range(len(sites)-1, -1, -1):
        capacity[i] = max(capacity[i+1], 1+capacity[following[i]])
    return following, capacity


def sample_profile(original, kind, cfg, context=''):
    """Fix severity before the MLLM chooses edit locations; independent of batch order."""
    settings = cfg['perturbation']
    digest = hashlib.sha256(f'{cfg["seed"]}\0{context}\0{kind}\0{original}'.encode()).digest()
    seed = int.from_bytes(digest, 'big')
    rng = random.Random(seed)
    levels = settings['severity_probabilities']
    severity = rng.choices(list(levels), weights=list(levels.values()))[0]
    rate = rng.uniform(*settings['edit_fraction'][severity])
    sites = edit_sites(original, kind, settings['insertion_characters'])
    _, capacity = site_capacity(sites)
    count = min(capacity[0], max(1, int(len(sites) * rate + .5)))
    return dict(seed=seed, severity=severity, rate=rate, edits=count)


def candidates(original, kind, cfg, context=''):
    if not applicable(original, kind):
        return []
    settings = cfg['perturbation']
    profile = sample_profile(original, kind, cfg, context)
    rng = random.Random(profile['seed'] ^ 0xCAFE)
    options = []
    sites = edit_sites(original, kind, settings['insertion_characters'])
    following, capacity = site_capacity(sites)
    # Count feasible plans so edit locations are uniform across the whole text,
    # rather than favouring the first few characters of the first word.
    plans = [[1] + [0] * profile['edits'] for _ in range(len(sites)+1)]
    for i in range(len(sites)-1, -1, -1):
        for count in range(1, min(profile['edits'], capacity[i])+1):
            plans[i][count] = plans[i+1][count] + plans[following[i]][count-1]
    for _ in range(settings['candidate_count'] * 24):
        plan, i, remaining = [], 0, profile['edits']
        while remaining:
            take = plans[following[i]][remaining-1]
            skip = plans[i+1][remaining]
            if rng.randrange(take + skip) < take:
                start, end, replacements = sites[i]
                plan.append((start, end, rng.choice(replacements)))
                remaining -= 1
                i = following[i]
            else:
                i += 1
        text = original
        for start, end, replacement in reversed(plan):
            text = text[:start] + replacement + text[end:]
        if text != original and text not in options:
            options.append(text)
        if len(options) == settings['candidate_count']:
            break
    return options


def validate_perturbation(text, original, kind, cfg, context=''):
    if text not in candidates(original, kind, cfg, context):
        raise ValueError(f'Not a valid seeded {kind} corruption at the sampled severity')
    return text


def request(original, kind, cfg, context=''):
    options = candidates(original, kind, cfg, context)
    if not options:
        raise ValueError(f'No {kind} operation possible for this modifier')
    profile = sample_profile(original, kind, cfg, context)
    task = {'field': 'choice', 'system': PERTURBATION, 'prompt':
            f'Operation: {KINDS[kind]}\nSampled severity: {profile["severity"]}; '
            f'{profile["edits"]} edit operations. Do not reduce or change this severity.\nOriginal: {original}\nOptions:\n' +
            '\n'.join(f'{i}: {text}' for i, text in enumerate(options))}
    def validate(result):
        index = result.get('choice')
        if type(index) is not int or not 0 <= index < len(options):
            raise ValueError(f'choice must be an integer in 0..{len(options)-1}')
        return validate_perturbation(options[index], original, kind, cfg, context)
    return task, validate
