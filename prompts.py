"""Shared prompts: observable appearance, reference-to-target direction, short requests."""

DESCRIPTION = '''Describe the single image's main object, shape, visible colour and composition
in at most 25 words. Report visible facts only. Do not infer provenance, age, purpose, or hidden
materials. Say gold-coloured instead of assuming gold. Return only JSON: {"description":"..."}.'''

MODIFIER = '''Write an English composed-image search modification request.
BEFORE describes a reference image. AFTER lists the corresponding annotated target images.
Targets are SEPARATE acceptable alternatives, never objects to merge into a collection.
State how to change BEFORE toward traits shared by AFTER, with the direction clearly correct.
Use the broadest useful common object category when targets differ. If needed, use "or" for
incompatible alternatives. Omit colours, backgrounds and shapes not shared across targets.
Include useful observable target traits where supported; invent nothing. Do not assert that
anything stays unchanged unless supported. Do not infer precious metals or historical facts.
Write ONE natural imperative, ideally 10-15 words and NEVER more than 15 words.
Return only JSON: {"modifier":"..."}. No image IDs, explanations, captions, or alternatives list.
Base every object and attribute on the supplied actual observations, not on a generic example.'''

PERTURBATION = '''Select one provided corruption of a human image-search request.
The options use only the named edit type and the SAME sampled severity or uppercase count.
Choose plausible real typing or editing mistakes while retaining the intended visual search.
Do not soften severe examples: rare extreme corruptions are intentional. Do not rewrite the text,
invent options, or change visual attributes. Return only JSON: {"choice": 0}, using a valid
zero-based option index. The application applies and validates that exact option.'''
