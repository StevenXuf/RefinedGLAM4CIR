"""Minimal, discriminative English changes conditioned on the reference image."""

MODIFIER = (
    'Write the smallest clear change instruction in plain, everyday English. '
    'Image 1 is the reference; the remaining images are targets for the same query. '
    'The search uses the reference PLUS your modifier: keep the reference as context. '
    'Choose the single clearest visual difference shared by ALL shown targets. '
    'Add a second difference only if essential to make the intended change clear. '
    'Leave unchanged reference features implicit. Name a new object type only when '
    'the object type changes. Omit incidental target details, attribute lists, and '
    'features unique to individual targets. Do not invent a difference or write a '
    'standalone target description. For a colour change alone, say "Make it blue", '
    'not "Find a tall blue ceramic vase with flowers". Be specific about WHAT changes; '
    'never say only "make it different" or "find something similar". Begin with an '
    'action such as {modifier_actions}. Use {min_words}-{max_words} words, aiming for '
    '{target_words}; do not add details just to reach that length. Return only the '
    'English modifier, without quotes or explanations. Use a natural sentence with '
    'articles, not a telegraphic list: "Remove the glasses", not "Remove glasses".'
)

MODIFIER_REVIEW = (
    'Review this draft against image 1 (reference) and the remaining target images: '
    '{modifier}\nThe search uses the reference PLUS this modifier. Rewrite in everyday '
    'English using only the smallest clear difference shared by ALL shown targets. '
    'Keep unchanged reference features implicit; remove incidental details and '
    'target-specific attributes. Add a second change only when essential. State what '
    'changes, without becoming vague or inventing a contrast. Use an action instruction, '
    'not a target caption. Begin with Make, Change, Replace, or Find. Use a natural '
    'sentence with articles rather than a telegraphic list. Aim for {target_words} '
    'words within {min_words}-{max_words}; do not pad it. Return only the corrected '
    'English modifier.'
)
