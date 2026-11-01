"""Plain-language modifier generation, direct corruptions, and pairwise review."""

MODIFIER = (
    'Compare the attached images directly. Image 1 is the reference; all remaining images '
    'are ground-truth targets for the same search query. Write one concrete, everyday '
    'search request describing the change from the reference toward ALL shown targets. Use '
    'their most specific shared object name and one or two shared visible details. Omit '
    'differing attributes, alternatives, invented materials or provenance, and details from'
    ' the reference that do not occur in the targets. Prefer stable shared features because'
    ' the targets may be sampled from a larger group. Use {min_words}-{max_words} words, '
    'preferably 6-12. Do not write descriptions or explain your comparison. Return only the'
    ' search request itself as plain text, without quotes or explanations.'
)

PERTURBATION = (
    'Directly corrupt this image-search request to imitate a real typing mistake. Use ONLY '
    'the specified operation, keeping the same intended visual search. Severity: '
    '{severity}. Aim to edit approximately {percent:.1f}% of positions eligible for this '
    'operation, with at least one edit. Mild means few small errors, moderate means several'
    ' noticeable errors, severe means many errors, and extreme means widespread corruption.'
    ' Do not soften a severe or extreme request. Prefer edits inside words; preserve the '
    'logical words no, not, without, or, and, nor. Do not paraphrase, explain, fix the '
    'original text, or introduce any other corruption type. Operation: {operation} '
    'Original: {modifier} Return only the corrupted request itself as plain text, without '
    'quotes or explanations.'
)

REVIEW = """Judge this clean/perturbed pair against the specified corruption operation.
Operation: {operation}
Requested severity: {severity}; approximate fraction: {percent:.1f}% of eligible positions.
Clean modifier: {modifier}
Perturbed modifier: {perturbed}
PASS requires an actual change using ONLY the requested operation. Reject unchanged
text, paraphrasing, new visual attributes, explanations, or another corruption type.
Check that the intended visual search and logical words are retained. Judge the
severity approximately; short text needs at least one edit, and severe/extreme
text may be difficult to read. Character transposition swaps adjacent characters
within words, never entire words. A substitution must use neighbouring QWERTY keys.
Reply only PASS or FAIL: followed by a brief reason of at most 15 words."""

REPAIR = """
The previous attempt was rejected.
Previous attempt: {perturbed}
Reviewer: {judgment}
Generate one corrected corruption at the SAME requested severity using ONLY the
specified operation. Make an actual change. Return only the corrupted request."""
