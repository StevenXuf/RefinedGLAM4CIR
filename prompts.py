"""Grounded image observations and short, everyday search requests."""

DESCRIPTION = """Describe what you can actually see, using plain everyday words.
Name the main object first, then its two or three most distinctive visible features:
shape, parts such as handles or a lid, decoration, colour, or the kind of picture.
Prefer "a bowl with two handles" to "an ornate artifact". Be precise about what is
pictured: a drawing of a chair is not a photograph of a chair. Use at most 25 words.
Do not guess dates, makers, purpose, precious metals, or hidden materials.
Return only JSON: {"description":"..."}."""

SHARED_FEATURES = """Describe the most specific visible features shared by ALL the targets.
Each entry describes one target or a subgroup. Name the actual common object or picture
category and, when available, one or two distinctive shared details. Use everyday nouns.
Keep a useful shared shape, part, decoration, or picture type even when colours differ.
For example, differing vase colours do not require dropping their shared handles.
Drop only details that differ or lack support. Broaden the object name only when a
narrower name really excludes a target. Never default to "objects" or "artifacts" when
a more useful shared name or visible description fits. Do not join different targets
into one scene, list alternatives, use "or", or invent common details.
Return a short noun phrase, ideally 3-12 words, as JSON: {"shared_features":"..."}."""

MODIFIER = """Write a short search request as a person would type when browsing images.
BEFORE describes the reference. COMMON FEATURES and INDIVIDUAL TARGET IMAGES describe
the desired results. Ask for the visible change from the reference toward those targets.
The reference is NOT a target. Never copy its object name into the desired result unless
that object is also present in every target.
Name the target object clearly and keep one or two useful details that ALL shown targets
share. Prefer the most specific supported noun: "a vase with two handles", not "an object".
If the object type stays the same, ask for its changed appearance without inventing a swap.
Use plain verbs such as "Find", "Show me", or "Swap this for". This is image search,
not an instruction to draw or generate a new image. Avoid catalogue and academic wording
such as "depicting", "attire", "exhibiting", "composition", and "characterized by".
Write a complete, natural request in 5-15 words; prefer 6-12 when enough. Do not pad it
with "an image of", "a picture containing", or other filler just to reach a length.
For style only: "Swap this bowl for a blue vase with two handles." Use those details
ONLY if the supplied observations support them; never copy an example's objects or traits.
Different targets may have different colours or layouts. Omit those differences, but keep
other shared details. Do not describe the union of the targets, list alternatives with
"or", or turn them into one scene. Do not guess provenance, age, purpose, or materials.
The shown targets may be a sample of a larger group: prefer stable identifying features
over exact counts, backgrounds, and tiny details. Do not become vague solely because
there are unseen targets. Each shown target must fit the final request.
Return only JSON: {"modifier":"..."}. No image IDs, explanations, or captions."""

BROAD_MODIFIER = """Write ONE everyday search request that fits EVERY target description.
Do not describe just one target. First find their nearest useful COMMON category, then
keep only a visible detail that all entries share. Check your noun against each entry.
Examples of the reasoning: a jug and a goblet are containers; a plough and a wrench are
tools; a bust and a full-body statue are sculptures. A photo and a sketch of horses
can share "horses", but not "photo" or "sketch". These are examples, not target details.
A machine and a small hand tool can both be tools or equipment. Never use a specific
object name that excludes another target, however prominent it is in one description.
If one image has arrows and another does not, omit arrows; do not add "without arrows".
Keep the narrowest accurate category and any genuinely shared shape, parts, or decoration.
Do not invent details, list alternatives, or combine separate targets into one scene.
Use "Show me ..." or "Find ..." and 5-15 plain words. Avoid vague pronouns, academic
wording, and "physical objects" when a useful category such as tools or clothing fits.
Return only JSON: {"modifier":"..."}."""

TARGET_REVIEW = """Check only the explicit desired-result requirements of this search request
against the ONE supplied target observation. For replacement requests, the source phrase
has already been removed. Judge the desired target, not how formal the wording sounds.
Accept ordinary object names, everyday synonyms, and broader categories when compatible.
A plural noun alone does not specify a number. Attributes not requested are unconstrained.
A drawing of an object is different from a photograph of that object when the request
explicitly specifies the picture type. Do not infer extra requirements from a noun:
"coins" does not mean gold coins, and "carved" does not mean new or unworn.
Reject an explicit category or detail only if unsupported or contradicted by this target.
If rejected, name the exact unsupported words and what the target visibly shows instead;
do not ask to remove correct words or all useful details. Do not rewrite the request.
Return only JSON: {"coverage":"Y", "reason":""} when supported, or
{"coverage":"N", "reason":"short unsupported requirement", "visible_features":"plain description of what this target actually shows"}.
For a rejection, visible_features must describe the actual target without quoting or negating
the rejected request. Name its subject and distinctive visible details in at most 20 words.
The coverage must be exactly Y or N."""

VISUAL_TARGET_REVIEW = (
    TARGET_REVIEW.replace('observation', 'image')
    + """
The attached target IMAGE is authoritative. Inspect its pixels directly. No source image
or caption is supplied. Do not reject a visible feature just because a caption omitted it.
Ordinary category names include stylized examples when visually compatible, but do not
excuse a genuinely different subject or an explicit contradictory feature."""
)

CATEGORY_FALLBACK = """Choose the most specific listed search request that fits EVERY target
description. A request describing only one target is invalid. Prefer a meaningful category
such as tools or clothing over objects whenever it covers all targets. Use visible detail
only if every target supports it. Do not invent a new option or change its words.
Return only JSON: {"choice": 0}, with a listed zero-based integer index."""

PERTURBATION = """Select one provided corruption of a human image-search request.
The options use only the named edit type and the SAME sampled severity and edit count.
Choose plausible real typing or editing mistakes while retaining the intended visual search.
Do not soften severe examples: rare extreme corruptions are intentional. Do not rewrite the text,
invent options, or change visual attributes. Return only JSON: {"choice": 0}, using a valid
zero-based option index. The application applies and validates that exact option."""
