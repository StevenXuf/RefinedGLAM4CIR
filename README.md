# PerturbedCIR4GLAM

This project measures how everyday typing and editing mistakes affect composed image retrieval on GLAM. A multimodal language model observes a reference image and its annotated targets, generates a short modification request, and selects explicit textual corruptions. A retrieval encoder combines the reference image and text and ranks the same complete available gallery for every condition.

The project performs inference only. It derives from [CIR4DH](https://github.com/StevenXuf/CIR4DH) and retains its MIT license.

## Run one split

Install a CUDA-compatible PyTorch/torchvision build, then:

```bash
python -m pip install -r requirements.txt
python main.py --split test_ood
python main.py --split test_id
```

On the current server, the isolated project interpreter is `.venv/bin/python`. All experiment settings, model IDs/revisions, GPU selection, data paths, word limits, and output settings are in `config.yaml`. The default Qwen3-VL-8B fits one A6000 and uses `generation.gpu_ids: [1]`. Other generation models and retrieval use `runtime.gpu_id`. GPU inference is required by default.

To run the complete pipeline separately for both supported splits:

```bash
bash run_splits.sh
# With another interpreter or configuration:
PYTHON=/path/to/python bash run_splits.sh /path/to/config.yaml
```

The script runs `main.py` once per split and returns a nonzero exit status if either run fails. Each invocation prints its own baseline/unperturbed/perturbed metric table and saves the console output to that split's `run.log`. For an additional combined log, use `bash run_splits.sh > outputs/run.log 2>&1` after creating `outputs/`.

## Data protocol

Only `test_id` and `test_ood` have CIR triplets in the provided annotations. Training and validation image metadata do not define reference–target relationships, so those partitions are excluded. No model is trained.

- `cir_db.csv` is the actual file name in the source repository; there is no `db_cir.csv`. Its `partition`, `id1`, `id2`, and `query` columns define existing triplets. Evaluation uses the first available reference and **all existing non-reference targets from the same row**. Generation samples up to three of those targets. The original modifier supplies the text baselines but is not a generation hint.
- `db_processed.txt` defines gallery partition membership only. Every unique available image in the selected partition is searched. It never supplies or invents query pairs.
- Files are resolved under `/data/data_fxu/GLAM/raw`; no images are downloaded. Physical storage folders are independent of annotation partitions, so a selected test ID may resolve under a folder named `train` or `val` without evaluating unrelated training/validation annotations.
- Missing files are skipped using the original loader's availability rule. Counts are logged and included in final metrics. `missing_gallery_policy: strict` instead requires every declared gallery file.
- Pair deduplication follows the original loader. All retained queries are evaluated by default. `subset.query_limit` is an optional query-only diagnostic limit and **never reduces the gallery or the target group**. Gallery subsampling and target-count filters are rejected.

The current local data yields 81 valid test_ood query pairs with all 22,373 declared gallery images, and 1,429 valid test_id pairs with all 2,363 available images out of 2,631 declared. The other 268 test_id gallery files are absent locally; these results do not cover that missing portion.

The published CIR4DH protocol retains 1,435 test_id queries. This project removes repeated image IDs inside each target list before deduplicating reference/target pairs, which merges six additional rows with different original modifiers. These six rows are not missing-image exclusions. For exact paper reproduction, preserve the original loader's list multiplicities during pair deduplication.

## Generation and perturbations

For each query, generation uses all available targets when there are at most three, otherwise a random sample of **three targets** (`generation.max_target_images`). Sampling uses the configured seed and query ID, is repeatable, and is recorded as `generation_target_ids`. The pipeline describes each unique reference/selected target image once in memory, then extracts a short shared-feature phrase for each selected multi-target group. A modifier describes the reference-to-target change using features shared by the selected targets. It cannot list alternative categories with "or" or merge targets into one scene. When only a sample is observed, the prompt asks for broad categories and stable structural traits, avoiding exact colour, count, material, pose, and layout requirements that could exclude unseen targets.

If a shared summary fails format checks, generation uses the individual sampled observations directly. This does not bypass the modifier checks or target compatibility reviews.

After generation, the MLLM reviews the desired result against **each selected target image**, inspecting actual pixels rather than relying on a short caption that might omit important features. Every selected target must pass. If any fails, the application replaces the shared summary with individual observations and asks for a broader rewrite. After two rejected requests, it asks for a search request containing only their common object category. Every rewrite receives the same individual visual reviews, up to `generation.target_review_attempts`. Explicit replacement source phrases are excluded from the target compatibility check. Exhausted reviews stop the run after checkpointing accepted modifiers. Observations, shared phrases, and review responses remain in memory. These checks are model judgements, and unsampled targets are not visually reviewed. **All available targets remain ground truths for metrics and figure highlights; neither the gallery nor evaluation positives are sampled.**

Modifiers must contain **5–15 whitespace-delimited words**, with 10–15 preferred. Overlong responses are rejected and replaced by a validated concise clause selected by the MLLM, rather than truncated mid-sentence. An editable word is required so every configured corruption can be applied. Target-conditioned generation is not a target-blind benchmark, and prompt checks cannot certify human-level visual accuracy.

If narrower categories repeatedly fail review, the final attempt asks the MLLM to select one of the broad "physical objects" requests in `generation.fallback_modifiers`, then visually reviews that exact request against each sampled target. A constrained choice prevents the model from reintroducing rejected attributes. These fallback templates can provide little retrieval information and remain visible in the saved modifiers. Shared-feature generation does not guarantee an improvement over the original text.

The corruption types are:

| Type | Exact edit |
|---|---|
| `character_deletion` | Delete internal characters, leaving word endpoints intact |
| `character_insertion` | Insert configured letters/digits inside words |
| `character_substitution` | Replace internal characters with horizontal, vertical, or diagonal QWERTY neighbours |
| `adjacent_character_transposition` | Transpose nonoverlapping adjacent internal character pairs, such as `blue` → `bule` |
| `character_repetition` | Double selected internal characters, simulating repeated keystrokes |
| `space_insertion` | Insert spaces inside words |
| `space_deletion` | Remove spaces between words |
| `capitalization` | Randomize which characters are uppercase, usually retaining mostly lowercase text |

For character and space edits, each query/type independently samples a severity: mild 75%, moderate 20%, severe 4%, extreme 1%. The corresponding edit fractions are 1–3%, 4–8%, 10–20%, and 30–100% of eligible original-string sites. Budgets are rounded to the nearest integer, require at least one edit, and are capped at the number of nonoverlapping operations possible. Short texts can therefore share the same edit count across severity levels. No edit changes punctuation or edits inside logical words (`no`, `not`, `without`, `or`, `and`, `nor`); space deletion also preserves their surrounding boundaries.

Capitalization is mixed-case 98% of the time. Its uppercase probability is drawn from Beta(2, 18), with mean 10%, then used to sample an uppercase-character count; mixed cases always contain both cases. Characters assigned uppercase are randomized rather than capitalizing whole words. All-lowercase and all-uppercase modes each have probability 1%; an extreme that would leave the original unchanged falls back to mixed case. These distributions model varying corruption severity; they are simulation assumptions, not empirical measurements of human error rates. All probabilities, ranges, beta parameters, candidate count, and insertion characters are in `config.yaml`.

The application samples severity reproducibly from the seed, CSV query ID, clean modifier, and perturbation type **before** asking the MLLM to choose among candidates with that fixed budget. This prevents the MLLM from preferentially choosing milder cases. It chooses edit locations/replacements; the application validates and applies the exact seeded candidate. Transpositions never exchange words or visual attributes. Space edits can change the perturbed word count; the 5–15-word constraint applies to clean modifiers. Capitalization may leave retrieval unchanged when the encoder lowercases text. Per-split severity counts are logged, and final metrics retain the full perturbation configuration.

To reuse previously inspected clean modifiers with a new corruption/retrieval configuration, use an empty output directory and explicitly import the old checkpoint once:

```bash
python main.py --split test_ood --stage generate --reuse-clean /path/to/old/test_ood/modifiers.json
bash run_splits.sh
```

Import the corresponding `test_id` checkpoint in the same way when reusing both splits. The import checks the generation model/revision, shared-target generation policy, split, original CSV metadata for each imported record, sampled target IDs, and modifier validity; it discards old corruptions and records the source checkpoint hash. Partial checkpoints are supported: the remaining queries are generated normally and evaluation still requires the whole cohort. Legacy checkpoints created before the shared-target policy must be regenerated. For evaluation-only changes, `--reuse-modifiers` imports compatible clean modifiers **and** individually validates saved corruptions against their seeded candidates. Without an import option, generation runs normally. Omit import options when resuming a populated output directory.

Current model families are configurable through `generation.model`:

| Setting | Checkpoint | Implementation |
|---|---|---|
| `qwen` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Native Transformers; thinking disabled for concise JSON |
| `llava` | [LLaVA-OneVision-2-8B-Instruct](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct) | Revision-pinned upstream custom code |
| `deepseek` | [Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B) | Native HF conversion of DeepSeek's model, explicitly in understanding/text mode |

The selected models are in the requested comparable 7–8B size range. They are recent available releases in that range, rather than a mixture of small models and a 16B/27B model. DeepSeek uses Hugging Face's native conversion of Janus-Pro; image generation is disabled. All three backends use the same environment and validated prompts. Model revisions are pinned in YAML. Qwen batches requests; LLaVA and Janus use serial requests for processor compatibility.

When a model proposes an overlong request, the application constructs complete concise reference-to-target clauses by omitting secondary clauses or replacing the source description with "it". The same MLLM selects a concise candidate, which must pass all word-count and corruption checks. This never cuts a request at an arbitrary word boundary. Alternative-target and negation tokens are retained during shortening; logical operators are protected during character edits. Main objects can still be misidentified, so inspect the generated modifiers and figures; length/format checks are not human visual validation.

The full split runs use the YAML-selected model, Qwen by default; changing `generation.model` selects another backend. Choose a new output directory for each model comparison.

<!--
Local verification notes; omitted from the rendered project documentation.
Check each backend on three actual CSV triplets: one single-target query, one small multi-target query, and a query from the largest target group, sampling at most three targets per query and checking every configured corruption.
Run the component checks below before adopting another backend for a full experiment. A model can load successfully and still fail compatibility review on heterogeneous target groups; the application rejects those outputs instead of silently accepting them.

On the current server, Qwen3-VL-8B, LLaVA-OneVision-2-8B, and Janus-Pro-7B passed the three-triplet GPU generation/corruption checks under this policy. These small checks verify execution and validation, not universal semantic accuracy.
-->

## Retrieval heads and metrics

The default retrieval checkpoint is OpenCLIP `ViT-B-16` pretrained on `laion2b_s34b_b88k`. Every extractor returns finite, L2-normalized image/text vectors in a shared embedding space. The query is:

```text
normalize(w * normalize(reference_image) + (1-w) * normalize(modifier_text))
```

`w` is `retrieval.image_weight`. With the default `w=0.5`, the query is the L2-normalized sum of the image and text unit vectors. Gallery vectors are also L2-normalized, so their dot product with the normalized combined query is cosine similarity. Ranking uses stable ties, and the query reference is excluded by default.

| Encoder | Features used |
|---|---|
| BLIP | `BlipForImageTextRetrieval` with image CLS → `vision_proj` and text-only CLS → `text_proj`, matching the **ITC retrieval head** (`use_itm_head=False`) |
| CLIP | Trained image/text projection heads via `get_image_features` / `get_text_features` |
| SigLIP / SigLIP2 | Trained pooled image/text representations from their retrieval feature APIs |
| OpenCLIP / OpenVision | `encode_image` / `encode_text` in the trained contrastive space |
| CoCa | Contrastive image/text encoders; caption-decoder states are never used |

BLIP's binary image-text matching classifier (`itm_head`) requires joint cross-attention and does not produce independent cosine-search embeddings. Captioning, VQA, raw vision hidden states, and unprojected text states are not used. Transformers 4 tensor and Transformers 5 projected pooler return formats are handled explicitly. Incomplete pretrained retrieval weights are rejected.

Baseline scores are not a reproduction of Table 4 in the CIR4DH paper. That experiment uses OpenAI-pretrained ViT-B-16, forcibly resizes each image to 224×224 before the model transform, replaces `for` with `to` in modifiers, keeps reference images eligible for retrieval, and adds raw image/text embeddings for its combined baseline. This project's configured LAION checkpoint uses the model transform on the original image, preserves original modifiers, excludes the reference candidate, and normalizes each modality before fusion. The paper and this project both define Recall@K as macro multi-target recall.

Every run evaluates these baselines on the same query cohort and complete available gallery:

| Condition | Query representation |
|---|---|
| `reference_only` | Reference image only |
| `original_text_only` | Original modifier from `cir_db.csv` only |
| `generated_text_only` | Reviewed generated modifier only |
| `reference_original` | Normalized reference image + original modifier |
| `unperturbed` | Normalized reference image + generated modifier |
| Each corruption type | Normalized reference image + perturbed generated modifier |

TorchMetrics computes Recall@K, HitRate@K, average precision, and reciprocal rank for each query; query values are macro-averaged into Recall@K, HitRate@K, mAP, and MRR. Recall@K divides retrieved positives by **all available relevant targets for that query**; HitRate@K measures whether at least one was retrieved. Missing target files are removed individually before generation and evaluation. Metrics also intersect target IDs with the actual gallery; a query with no available positives is skipped. Complete gallery rankings are required, so a target below a displayed top-5 list is never mistaken for a missing target.

Real cosine scores determine rankings, with deterministic ties and the same reference-image exclusion in every condition, including image-only retrieval. Metrics receive positive rank-equivalent scores to preserve that ordering without TorchMetrics ignoring relevant candidates with zero/negative cosine scores. Figures continue to display actual cosine similarity. AVG is the mean of Recall@1/5/10/50. **Every computed metric, including all HitRate@K columns, is printed and saved.** All scores are percentages; deltas are measured against `unperturbed` in percentage points.

## Outputs and figures

Only these experiment outputs are saved per split under `outputs/current/<split>/`:

- `modifiers.json`: original CSV identities/text, generated modifiers, and their perturbed versions; incrementally saved for resumption.
- `metrics.json`: final aggregate scores, deltas, model settings, query count, and gallery availability.
- `run.log`: split execution messages and every printed metric table, written by `run_splits.sh`.
- `figures/query_*.png`: three seeded examples with original, generated, and selected perturbed modifiers, reference/target images, and top-5 retrievals.

The first figure row shows **reference + original modifier**, followed by reference + generated modifier and the corruption conditions. Capitalization remains evaluated in the metrics but is excluded from every figure through `visualization.exclude_conditions`. Figures use equal 224×224 letterboxed images, equal square subplot axes (including text panels), wrapped centered text, tight layout, and green borders for **any available ground-truth target**, including targets outside the generation sample. Retrieval captions show `Top-1 (0.786)`, with the rank in black and cosine similarity in green. Sampling selects existing retained CSV triplets and uses the same triplets for every condition. `python figure.py --split test_ood` regenerates figures from saved modifiers; it recomputes retrieval against the entire gallery rather than storing intermediate rankings.

Observations, feature tensors, full rankings, manifests, environment dumps, and diagnostic reports are not exported. GPU/model weights remain in the ignored `.model_cache/`. Intermediate computation stays in memory. A failed generation retains completed modifiers; restarting resumes them, but unfinished image observations are recomputed. Configuration/source/image-availability signatures reject incompatible modifier checkpoints; use a new `paths.output_dir` when changing an experiment.

All outputs, old local reports, logs, environments, and model caches are ignored by Git. Local inspection files are preserved outside the tracked source tree.

## Components

| File | Responsibility |
|---|---|
| `main.py` | Assemble data loading → generation → clean retrieval → perturbation → paired retrieval → figures |
| `loading_data.py` | CSV triplets, local paths, and complete available split galleries |
| `generation.py` / `prompts.py` | Model adapters, batched requests, prompts, JSON and word-length checks |
| `perturbation.py` | Explicit corruption candidates and exact operation validation |
| `feature_extractor.py` | Projected retrieval embeddings and reference/text composition |
| `metrics_computation.py` | Stable ranking, multi-positive metrics, and paired deltas |
| `figure.py` | Seeded paired visualizations |
| `utils.py` | YAML configuration, seeding, GPU selection, atomic output |
| `run_splits.sh` | End-to-end invocation for each supported split |

Every production Python entry point accepts `--config`. Individual pipeline stages are available through `main.py --stage`.

<!--
Local verification instructions; all associated code is kept in the Git-ignored test/ directory.
Stages can be checked independently, in order:

```bash
python main.py --split test_ood --stage data
python main.py --split test_ood --stage generate
python main.py --split test_ood --stage perturb
python main.py --split test_ood --stage evaluate
python main.py --split test_ood --stage visualize
python test/verify.py --component generation --model qwen
python test/verify.py --component generation --model llava
python test/verify.py --component generation --model deepseek
python test/verify.py --component retrieval --model blip
python test/verify.py --component retrieval --model openclip
python -m pip install -r test/requirements.txt
python -m pytest -c test/pytest.ini --basetemp test/.tmp
```

Every Python entry point accepts `--config`. Retrieval checks also accept `clip`, `siglip`, `siglip2`, `coca`, and `openvision`. The checks compare extracted cosine scores numerically against the backend's own retrieval calculation; they check head correctness, not dataset retrieval accuracy. Generation checks include single-target, small multi-target, and large multi-target queries, using the configured generation sample and validating all corruption conditions. Full experiments retain every available ground truth regardless of group size.
-->
