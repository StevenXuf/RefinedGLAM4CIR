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

The script runs `main.py` once per split and returns a nonzero exit status if either run fails. Each invocation prints its own unperturbed/perturbed metric table. For an optional console log, use `bash run_splits.sh > outputs/run.log 2>&1` after creating `outputs/`.

## Data protocol

Only `test_id` and `test_ood` have CIR triplets in the provided annotations. Training and validation image metadata do not define reference–target relationships, so those partitions are excluded. No model is trained.

- `cir_db.csv` is the actual file name in the source repository; there is no `db_cir.csv`. Its `partition`, `id1`, `id2`, and `query` columns define existing triplets. The first available reference and **all existing non-reference targets from the same row** are used. The old modifier is preserved for inspection but is not supplied as a generation hint.
- `db_processed.txt` defines gallery partition membership only. Every unique available image in the selected partition is searched. It never supplies or invents query pairs.
- Files are resolved under `/data/data_fxu/GLAM/raw`; no images are downloaded. Physical storage folders are independent of annotation partitions, so a selected test ID may resolve under a folder named `train` or `val` without evaluating unrelated training/validation annotations.
- Missing files are skipped using the original loader's availability rule. Counts are logged and included in final metrics. `missing_gallery_policy: strict` instead requires every declared gallery file.
- Pair deduplication follows the original loader. All retained queries are evaluated by default. `subset.query_limit` is an optional query-only diagnostic limit and **never reduces the gallery or the target group**. Gallery subsampling and target-count filters are rejected.

The current local data yields 81 valid test_ood query pairs with all 22,373 declared gallery images, and 1,429 valid test_id pairs with all 2,363 available images out of 2,631 declared. The other 268 test_id gallery files are absent locally; these results do not cover that missing portion.

## Generation and perturbations

The pipeline describes each unique CSV reference/target image once in memory, then synthesizes a modifier using the reference observation and every corresponding target observation. Targets are alternative acceptable results; prompts explicitly forbid merging them into one scene. Prompts require the correct reference-to-target direction, shared observable traits, and no invented provenance, material, or unchanged background claims.

Modifiers must contain **5–15 whitespace-delimited words**, with 10–15 preferred. Overlong responses are rejected and replaced by a validated concise clause selected by the MLLM, rather than truncated mid-sentence. An editable word is required so every configured corruption can be applied. Target-conditioned generation is not a target-blind benchmark, and prompt checks cannot certify human-level visual accuracy.

Concrete corruption types replace the former ambiguous informal style:

| Type | Exact edit |
|---|---|
| `deletion` | Omit one internal letter from one word, simulating a missed keystroke |
| `character_swap` | Transpose adjacent internal letters in one word, such as `blue` → `bule` |
| `capitalization` | Change letter case only, simulating accidental Caps Lock or inconsistent case |
| `keyboard_typo` | Replace one internal letter with a neighbouring key in the same QWERTY row |

The MLLM selects among candidates produced by the specified operation; the application validates and applies that exact candidate. This keeps edit severity controlled, makes the perturbations independently checkable, and avoids semantic rewrites disguised as typos. Character swapping refers to letters within a word, not swapping subjects or attributes. Capitalization can leave retrieval unchanged when the encoder lowercases text.

Current model families are configurable through `generation.model`:

| Setting | Checkpoint | Implementation |
|---|---|---|
| `qwen` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Native Transformers; thinking disabled for concise JSON |
| `llava` | [LLaVA-OneVision-2-8B-Instruct](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct) | Revision-pinned upstream custom code |
| `deepseek` | [Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B) | Native HF conversion of DeepSeek's model, explicitly in understanding/text mode |

The selected models are in the requested comparable 7–8B size range. They are recent available releases in that range, rather than a mixture of small models and a 16B/27B model. DeepSeek uses Hugging Face's native conversion of Janus-Pro; image generation is disabled. All three backends use the same environment and validated prompts. Model revisions are pinned in YAML. Qwen batches requests; LLaVA and Janus use serial requests for processor compatibility.

When a model proposes an overlong request, the application constructs complete concise reference-to-target clauses by omitting secondary clauses or replacing the source description with "it". The same MLLM selects a concise candidate, which must pass all word-count and corruption checks. This never cuts a request at an arbitrary word boundary. Alternative-target and negation tokens are retained during shortening; logical operators are protected during character edits. Main objects can still be misidentified, so inspect the generated modifiers and figures; length/format checks are not human visual validation.

All three selected backends were checked on actual CSV triplets, including every target in those triplets and every configured corruption. The full split runs use the YAML-selected model, Qwen by default; changing `generation.model` selects another backend. Choose a new output directory for each model comparison.

## Retrieval heads and metrics

Every extractor returns finite, L2-normalized image/text vectors in a shared embedding space. The query is:

```text
normalize(w * normalize(reference_image) + (1-w) * normalize(modifier_text))
```

`w` is `retrieval.image_weight`. Ranking uses cosine similarity with stable ties, and the query reference is excluded by default.

| Encoder | Features used |
|---|---|
| BLIP | `BlipForImageTextRetrieval` with image CLS → `vision_proj` and text-only CLS → `text_proj`, matching the **ITC retrieval head** (`use_itm_head=False`) |
| CLIP | Trained image/text projection heads via `get_image_features` / `get_text_features` |
| SigLIP / SigLIP2 | Trained pooled image/text representations from their retrieval feature APIs |
| OpenCLIP / OpenVision | `encode_image` / `encode_text` in the trained contrastive space |
| CoCa | Contrastive image/text encoders; caption-decoder states are never used |

BLIP's binary image-text matching classifier (`itm_head`) requires joint cross-attention and does not produce independent cosine-search embeddings. Captioning, VQA, raw vision hidden states, and unprojected text states are not used. Transformers 4 tensor and Transformers 5 projected pooler return formats are handled explicitly. Incomplete pretrained retrieval weights are rejected.

Recall@K is macro multi-positive recall: recovered relevant targets divided by all relevant targets, averaged across queries. HitRate@K, mAP, and MRR are also computed. AVG is the mean of Recall@1/5/10/50. All scores are percentages; deltas are perturbed minus unperturbed in percentage points. Gallery, positives, reference, fusion, and ranking settings stay fixed across conditions.

## Outputs and figures

Only these experiment outputs are saved per split under `outputs/current/<split>/`:

- `modifiers.json`: original CSV identities/text, generated modifiers, and their perturbed versions; incrementally saved for resumption.
- `metrics.json`: final aggregate scores, deltas, model settings, query count, and gallery availability.
- `figures/query_*.png`: three seeded examples with unperturbed and every perturbed modifier, reference/target images, and top-5 retrievals.

Figures use equal 224×224 letterboxed images, wrapped centered text, a consistent layout, and green borders for relevant results. Sampling selects existing retained CSV triplets and uses the same triplets for every condition. `python figure.py --split test_ood` regenerates figures from saved modifiers; it recomputes retrieval against the entire gallery rather than storing intermediate rankings.

Observations, feature tensors, full rankings, manifests, environment dumps, and diagnostic reports are not exported. GPU/model weights remain in the ignored `.model_cache/`. Intermediate computation stays in memory. A failed generation retains completed modifiers; restarting resumes them, but unfinished image observations are recomputed. Configuration/source/image-availability signatures reject incompatible modifier checkpoints; use a new `paths.output_dir` when changing an experiment.

All outputs, old local reports, logs, environments, and model caches are ignored by Git. Local inspection files are preserved outside the tracked source tree.

## Components and verification

| File | Responsibility |
|---|---|
| `main.py` | Assemble data loading → generation → clean retrieval → perturbation → paired retrieval → figures |
| `loading_data.py` | CSV triplets, local paths, and complete available split galleries |
| `generation.py` / `prompts.py` | Model adapters, batched requests, prompts, JSON and word-length checks |
| `perturbation.py` | Explicit corruption candidates and exact operation validation |
| `feature_extractor.py` | Projected retrieval embeddings and reference/text composition |
| `metrics_computation.py` | Stable ranking, multi-positive metrics, and paired deltas |
| `figure.py` | Seeded paired visualizations |
| `verify.py` | Actual GPU generation and native retrieval-head equivalence checks |
| `utils.py` | YAML configuration, seeding, GPU selection, atomic output |
| `run_splits.sh` | End-to-end invocation for each supported split |

Stages can be checked independently, in order:

```bash
python main.py --split test_ood --stage data
python main.py --split test_ood --stage generate
python main.py --split test_ood --stage perturb
python main.py --split test_ood --stage evaluate
python main.py --split test_ood --stage visualize
python verify.py --component generation --model qwen
python verify.py --component generation --model llava
python verify.py --component generation --model deepseek
python verify.py --component retrieval --model blip
python verify.py --component retrieval --model openclip
python -m pip install pytest
python -m pytest -q
```

Every Python entry point accepts `--config`. Retrieval checks also accept `clip`, `siglip`, `siglip2`, `coca`, and `openvision`. The checks compare extracted cosine scores numerically against the backend's own retrieval calculation; they check head correctness, not dataset retrieval accuracy. Generation checks use three small intact CSV target groups and validate all corruption conditions; full experiments do not filter target-group sizes.
