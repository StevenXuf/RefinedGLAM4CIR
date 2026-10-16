# Smoke modifier quality inspection

This is the assistant's visual inspection of the six reference images and every conditioning target in the smoke subset. It is not independent human annotation. Generated outputs were retained unchanged, including weak cases, so evaluation does not select samples based on their performance or reviewer judgments.

The same-model automated reviewer accepted grounding/direction/coherence for all six queries and flagged shared-target consistency for `test_ood:684` and `test_ood:1893`. Visual inspection found additional limitations. Those automated acceptance rates must not be interpreted as verified caption accuracy.

| Query | Visual inspection and limitation |
|---|---|
| `test_ood:384` | The target is a dark ring-shaped door knocker on stone. The reference shows a row of wall-mounted door hardware. The generated modifier identifies the target but calls the reference objects “faucets”; this reference identification needs correction. |
| `test_ood:684` | Targets include several figurines and fragments with different shapes and colours. One specific material/colour change cannot describe them all. Claims about stone/terracotta are inferences from appearance, not verified material labels. |
| `test_ood:1556` | The shift from a painted mask to illustrations of a group in a medical scene is visibly supported. This improves on an instruction restricted to men's clothing, since the scenes contain multiple people. |
| `test_ood:1893` | The targets are a decorated gold-coloured item photographed with a comb and a separate bangle. The modifier explicitly allows alternatives instead of merging them. The exact identity “compact” is uncertain and needs specialist review. |
| `test_ood:1895` | The reference ring and target bangle on red are visibly distinct. The object/background change is useful, but apparent gold/silver colours do not establish metal composition. |
| `test_ood:1972` | Both sides show coins with inscriptions and symbols. The modifier's wording is redundant and its background alternatives are weak discriminators. The material/colour/design differences deserve more careful annotation. |

All selected targets fit the eight-image conditioning budget, so this inspection covers the full relevant target set for these six queries. For larger runs that sample targets, quality conclusions apply only to the images shown.

The initial unrestricted perturbation test demonstrated that the same LLM can accept a semantic change such as “knocker” to “knob.” The final pipeline therefore uses LLM-generated word-level spelling edits and protected visual phrases for rephrasing. It validates word-edit budgets, visual vocabulary, protected phrase order and model-assisted semantic checks. This intentionally tests conservative wording perturbations. Broader synonym perturbations require further review.

Inspect `smoke_modifiers.json`, `conditioning_review.png`, and `example_retrieval.png` alongside these notes. The six-query subset is a functional experiment test, not a validated new language annotation benchmark or a statistically reliable robustness conclusion.
