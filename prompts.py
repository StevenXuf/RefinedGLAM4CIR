"""Plain-language modifier generation from direct visual comparisons."""

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
