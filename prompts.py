"""Everyday English change instructions, with review reserved for outliers."""

MODIFIER = (
    'Write one short modifier in plain, everyday English, using simple, familiar words. '
    'Image 1 is the reference; the remaining images are targets for the same query. '
    'Tell how to change the reference toward ALL shown targets. Begin with an action '
    'such as {modifier_actions}. Aim for {target_words} words, within '
    '{min_words}-{max_words} words. '
    'For example, "Make the vase taller and add blue flowers" is a modifier; '
    '"A tall vase with blue flowers" is only a caption. Name the shared target object '
    'and one or two shared visible details; omit differing attributes, academic wording, '
    'vague phrases, and invented details. Return only the English modifier, without '
    'quotes or explanations.'
)

MODIFIER_REVIEW = (
    'Review this draft against image 1 (reference) and the remaining target images: '
    '{modifier}\nRewrite it as a short change instruction in everyday English, '
    'not a target caption. Aim for {target_words} words; use {min_words}-{max_words} '
    'words total. Begin with Make, Change, Replace, or Find. Keep only the essential '
    'visual change shared by the targets. Remove extra details and formal wording. '
    'Return only the corrected English instruction, without quotes or explanations.'
)
