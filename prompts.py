"""Prompts shared by the Qwen, LLaVA, and DeepSeek adapters."""

REFINE_SYSTEM = '''You are a visual expert writing English composed-image retrieval modifiers.
The first image is a reference. The remaining images are relevant targets for the SAME query.
Each target is a SEPARATE ALTERNATIVE acceptable result, never a collection to be combined.
Inspect the images; the old instruction is noisy and may be wrong. Write one concise natural
modification request that transforms the reference's salient visual content toward the target
images. Describe concrete visible changes (object, shape, colour, material if visible, composition).
Use appearance terms such as gold-coloured instead of asserting a precious metal.
For multiple targets use shared visible traits, not details unique to one target. Do not assume
that backgrounds or other attributes remain unchanged if targets differ. Do not invent provenance,
artist, location, date, function, or invisible material. Avoid image IDs and retrieval terminology.
Include at least two useful visible target attributes where supported, rather than only naming
an object replacement. For a group of people do not assume they all have the same gender.
Return ONLY a JSON object with one key "modifier" containing 1-2 short English sentences.
Do not return analysis, markdown, a target caption, or a list of alternatives.'''

TARGET_DESCRIPTION_SYSTEM = '''Describe this SINGLE image in one concise sentence of at most
35 words. Identify the main object(s), visible colour, shape and background. Do not guess
provenance, date, location, function or hidden materials. Use gold-coloured/silver-coloured
instead of assuming metals. Return ONLY JSON: {"description": "one short sentence"}.'''

SYNTHESIS_SYSTEM = '''Write a composed-image retrieval modifier from image observations.
You receive a BEFORE reference description and a list of ALTERNATIVE acceptable AFTER targets.
The modifier must fit ALL targets, not just the first. First identify their shared visual traits.
If targets differ in subtype, use their broader common category. Omit colours, shapes and
backgrounds that are not shared. If targets have incompatible subjects, explicitly allow
alternatives with "or"; never combine separate target images into a single collection.
For example, a cat and a dog share "animal", not cat; red and blue vases share "vase", not red.
Describe how the reference should change toward that common target content, in 1-2 short
English sentences. Avoid invented details, image IDs, and claims about hidden materials.
Return ONLY JSON with "shared_traits" (list), "limitations" (list), and "modifier" (string).'''

PERTURB_SYSTEM = '''You simulate realistic variation in a person's image-search language.
You MUST change the wording of the supplied modifier according to the requested perturbation.
Preserve the visual intent, object identity, attributes, relations, and direction of each change.
Do not add or remove visual constraints. Returning the unchanged input is invalid.
Return ONLY JSON: {"modifier": "..."}.'''

TYPO_SYSTEM = '''You deliberately simulate keyboard spelling mistakes.
Return ONLY JSON: {"typo": "misspelled word"}.'''

WORDING_SYSTEM = '''Rewrite a search request while preserving protected CONTENT tokens.
Every CONTENT token must occur exactly once, unchanged, in its original order.
They represent intact visual phrases; never turn adjectives into a list of separate objects.
Change only surrounding function words, never add visual words. Keep the direction of replacement.
For example, "Replace the CONTENT0 with a CONTENT1" can become
"Please swap the CONTENT0 for a CONTENT1". Return ONLY JSON: {"modifier": "..."}.'''

REVIEW_SYSTEM = '''Review the supplied composed-image retrieval modifier against the images.
The first image is BEFORE (reference), and every later image is AFTER (a relevant target).
An instruction describes a CHANGE: the target object need not exist in the BEFORE image.
For example, "replace a red bowl with a blue vase" is correct when BEFORE shows a red bowl
and AFTER shows a blue vase. Do not reject it because BEFORE has no vase or AFTER has no bowl.
Check whether its changes
are visible, directionally correct, coherent, and shared across the targets shown. Flag invented
or overly specific claims. Return ONLY JSON with boolean keys "grounded", "direction_correct",
"coherent", "shared_across_targets", and string key "reason". Be critical.'''

PERTURB_REVIEW_SYSTEM = '''Compare an original modifier and its perturbed version.
Allow spelling errors, abbreviations, and paraphrases, but require the same intended objects,
visual attributes, relationships, and direction of change. Return ONLY JSON with boolean key
"meaning_preserved" and string key "reason". Be critical about added or lost constraints.
Spelling errors are INTENTIONAL: infer the intended word before comparing meanings. For example,
"blue vase" -> "bule vase" and "bishop" -> "bisho" preserve intended meaning and must pass.
Reject changes like "blue vase" -> "red vase", which change an actual visual constraint.'''


def refinement_prompt(original, target_count):
    hint = f'Noisy old instruction (only a hint): {original}\n' if original else ''
    return f'{hint}Relevant target images shown: {target_count}. Compare BEFORE to AFTER and write the modifier.'


def perturbation_prompt(modifier, instruction):
    return f'Original modifier: {modifier}\nPerturbation: {instruction}'
