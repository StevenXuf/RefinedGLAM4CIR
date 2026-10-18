"""LLM-selected, explicitly checked everyday typing/editing corruptions."""
import re

from prompts import PERTURBATION

KINDS = {
    'deletion': 'Omit one internal letter from one word, simulating a missed keystroke.',
    'character_swap': 'Transpose two adjacent internal letters in one word, like blue -> bule.',
    'capitalization': 'Change letter case only, like accidental CAPS LOCK.',
    'keyboard_typo': 'Replace one letter by a neighbouring key on an English QWERTY keyboard.',
}
# Leave logical operators intact and keep every word boundary.
PROTECTED = {'no', 'not', 'without', 'or', 'and', 'nor'}
ROWS = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
NEIGHBOURS = {letter: set(row[max(0, i-1):i] + row[i+1:i+2]) for row in ROWS for i, letter in enumerate(row)}


def candidates(original, kind):
    if kind not in KINDS:
        raise ValueError(f'Unknown perturbation: {kind}')
    options = []
    words = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", original))
    if kind == 'capitalization':
        options = [original.upper(), original.lower(), original.swapcase()]
    else:
        for match in words:
            word = match.group()
            if len(word) < 4 or word.lower() in PROTECTED:
                continue
            # Internal edits preserve recognisable word boundaries.
            for i in range(1, len(word)-1):
                if kind == 'deletion':
                    replacement = word[:i] + word[i+1:]
                    options.append(original[:match.start()] + replacement + original[match.end():])
                elif kind == 'character_swap' and i + 1 < len(word)-1 and word[i] != word[i+1]:
                    replacement = word[:i] + word[i+1] + word[i] + word[i+2:]
                    options.append(original[:match.start()] + replacement + original[match.end():])
                elif kind == 'keyboard_typo':
                    for letter in sorted(NEIGHBOURS.get(word[i].lower(), ())):
                        replacement = word[:i] + letter + word[i+1:]
                        options.append(original[:match.start()] + replacement + original[match.end():])
    return list(dict.fromkeys(option for option in options if option != original))


def validate_perturbation(text, original, kind):
    if text not in candidates(original, kind):
        raise ValueError(f'Not a valid single-operation {kind} corruption')
    return text


def request(original, kind):
    options = candidates(original, kind)
    if not options:
        raise ValueError(f'No {kind} operation possible; use a word with editable internal letters')
    # Bounded prompt length; candidates remain deterministic and independently verifiable.
    options = options[:24]
    task = {'field': 'choice', 'system': PERTURBATION, 'prompt': f'Operation: {KINDS[kind]}\nOriginal: {original}\nOptions:\n' +
            '\n'.join(f'{i}: {text}' for i, text in enumerate(options))}
    def validate(result):
        index = result.get('choice')
        if type(index) is not int or not 0 <= index < len(options):
            raise ValueError(f'choice must be an integer in 0..{len(options)-1}')
        return validate_perturbation(options[index], original, kind)
    return task, validate
