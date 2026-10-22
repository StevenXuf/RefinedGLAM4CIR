"""Shared prompts: observable appearance, reference-to-target direction, short requests."""

DESCRIPTION = """Describe the single image's main object, shape, visible colour and composition
in at most 25 words. Report visible facts only. Do not infer provenance, age, purpose, or hidden
materials. Say gold-coloured instead of assuming gold. Return only JSON: {"description":"..."}."""

SHARED_FEATURES = """Find the intersection of the visible features of ALL supplied target images.
Each entry describes ONE target or the shared traits of a subgroup. Keep every entry binding;
the result must fit every individual target, never a scene assembled from different targets.
Return a VERY SHORT noun phrase, ideally 3-10 words: one visible category and at most two
useful traits shared by EVERY target. Drop colour, material, pose, and display arrangement
entirely whenever they vary; never summarize the different values.
Use a broader visual category when individual subjects differ. Keep only colours, shapes,
composition, and object properties supported for ALL targets, including minority examples.
Do not list alternatives, use "or", merge objects, or invent shared details or provenance.
Return only JSON: {"shared_features":"..."}."""

MODIFIER = """Write an English composed-image search modification request.
BEFORE or REFERENCE IMAGE describes a reference image. COMMON FEATURES summarizes provided targets;
INDIVIDUAL TARGET IMAGES, when supplied, lists their actual observations instead of a summary.
The provided targets may be a random sample from a larger ground-truth group.
Prefer broad categories and stable traits to avoid excluding unseen ground truths with overly specific details.
Every existing target must satisfy the request after applying it to BEFORE.
State how to change BEFORE toward COMMON FEATURES, using only traits shared by ALL targets.
The change must describe the intersection of their visible traits, not a union of alternatives.
Use a broader visible category when subjects differ. Never enumerate different target categories,
use "or", combine targets into a scene, or include a trait true of only some targets.
Check EVERY target, including minority examples, before writing the request.
Omit colours, backgrounds, shapes, and materials not supported across ALL targets.
If a target review rejects a supplied common feature, remove it and use a broader category;
the individual target review takes precedence over a possibly over-specific shared summary.
Include useful observable target traits where supported; invent nothing. Do not assert that
anything stays unchanged unless supported. Do not infer precious metals or historical facts.
Write ONE natural imperative, ideally 10-15 words and NEVER more than 15 words.
Return only JSON: {"modifier":"..."}. No image IDs, explanations, captions, or alternatives list.
Base every object and attribute on the supplied actual observations, not on a generic example."""

BROAD_MODIFIER = """Write a broad image-search request that fits EVERY described target.
Use only ONE common object category. Omit colour, shape, material, count, background,
arrangement, and other properties. Never combine different objects into one scene.
If a previous category was rejected, choose its broader parent category. An object need
not depict a human to count as an artifact, and a tool is not necessarily machinery.
When no narrow category includes all targets, use a broad category such as physical objects.
Use the form "Find an image of ..." followed by the common category, in 5-15 words.
Return only JSON: {"modifier":"..."}."""

TARGET_REVIEW = """Audit ONLY the explicit desired-result requirements of a search request
against the ONE target observation supplied here. Output the single character Y if they fit
this target, otherwise the single character N. For explicit replacement requests the source
phrase has already been removed; judge the supplied desired result only.
Broader category names and ordinary synonyms are allowed if compatible with the observation.
A generic plural category name is not an exact count requirement; reject count differences
only when the request states a number or explicit quantity such as a pair or a row.
Unmentioned colour, exact shape, age, newness, material, background, and other attributes are
unconstrained. Never invent an unstated requirement: raised designs do NOT imply new or unworn,
and "coins" does NOT imply gold-coloured coins. Reject a claimed property only if unsupported
or contradicted by the target observation, not because of an attribute the request never stated.
Do not excuse a contradictory colour, object category, shape, or composition because it fits other
targets. Prefer rejecting an unsupported descriptor to assuming all targets share it.
Return only JSON: {"coverage":"Y", "reason":""}.
If rejected, give a short reason identifying the explicit unsupported requirement.
Do not rewrite the modifier. The coverage value must be exactly Y or N, never YYY or NNN."""

VISUAL_TARGET_REVIEW = (
    TARGET_REVIEW.replace('observation', 'image')
    + """
The attached target IMAGE is authoritative. Inspect its visible object category and properties
directly. No text caption or source image is supplied. Do not reject a category because a caption
failed to mention it; decide from the actual target image. Generic category plurals allow one
or several instances. A generic category such as figurine includes abstract/stylized figures."""
)

CATEGORY_FALLBACK = """Narrower descriptions failed compatibility checks. Select the most
natural broad image-search request from the listed options. Do not add any attribute or
rewrite the request. Return ONLY JSON: {"choice": 0}, with a listed zero-based integer index."""

PERTURBATION = """Select one provided corruption of a human image-search request.
The options use only the named edit type and the SAME sampled severity and edit count.
Choose plausible real typing or editing mistakes while retaining the intended visual search.
Do not soften severe examples: rare extreme corruptions are intentional. Do not rewrite the text,
invent options, or change visual attributes. Return only JSON: {"choice": 0}, using a valid
zero-based option index. The application applies and validates that exact option."""
