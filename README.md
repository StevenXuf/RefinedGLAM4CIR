# PerturbedCIR4GLAM

A modular experiment pipeline derived from [StevenXuf/CIR4DH](https://github.com/StevenXuf/CIR4DH), source commit `d37998f`. It builds visually grounded modifiers from GLAM reference/target pairs, retrieves with reference images plus those modifiers, generates realistic textual perturbations, and measures the change in retrieval performance.

See the [GPU smoke report](reports/smoke_report.md) and [modifier quality inspection](reports/quality_inspection.md) for measured results and remaining annotation issues.

The original repository has been disconnected: this checkout's `origin` points only to `StevenXuf/PerturbedCIR4GLAM`. The original history and MIT license are retained.

## Setup and run

Use Python 3.10 or newer with a CUDA-compatible PyTorch/torchvision installation, then install `requirements.txt`. CUDA is required by default. On the existing server, `/home/fxu/my-env/bin/python` is the tested interpreter. Choose a free GPU using `runtime.gpu_id` in `config.yaml`.

```bash
python -m pip install -r requirements.txt
python run.py --stage all
```

All experiment settings live in `config.yaml`. Paths are resolved relative to the configuration file, so scripts also work from other working directories. The pipeline reads existing PNGs from `/data/data_fxu/GLAM/raw/{test_id,test_ood,train,val}`. It never downloads GLAM images. Model weights are downloaded into the ignored `.model_cache/` directory when needed.

The default smoke experiment selects six CSV queries across reference-image categories, with at most eight relevant targets per query, and a fixed 512-image gallery. The target-count bound makes it possible to inspect every selected target during the smoke test. Every selected relevant target is retained in the gallery; the other images are seeded distractors. These are subset metrics, not full-benchmark results. To use all valid queries and the full split gallery, edit:

```yaml
subset:
  balance_categories: true
  query_limit: null
  max_targets_per_query: null
  gallery_limit: null
paths:
  output_dir: outputs/full_test_ood
```

Keep the other `paths` entries when editing. Each configuration/data/source signature needs a separate output directory. Changing settings in an existing run produces an error instead of mixing results.

## Pipeline and stages

`run.py --stage all` executes the following sequence. Individual stages can be rerun after a failure; completed generation/perturbation records and image features are reused.

| Stage | Behavior |
|---|---|
| `prepare` | Read `cir_db.csv`, select the configured partition, remove rows without a valid reference/target, deduplicate ID pairs, resolve existing image paths, and freeze query/gallery manifests. |
| `generate` | Use Qwen, LLaVA, or DeepSeek to describe the reference and each conditioning target separately, then synthesize a modifier from these observations and their shared traits. Save model-assisted quality checks. |
| `retrieve-clean` | Compose normalized reference-image and clean-text embeddings, rank the fixed gallery, and save clean metrics before perturbation. |
| `perturb` | Use the same LLM to create typo, informal, and paraphrase variants. Reject empty/unchanged outputs, excessive typo edits, visual-word substitutions, and variants the semantic reviewer rejects. |
| `evaluate` | Evaluate clean and every perturbed condition against identical queries, reference embeddings, gallery embeddings, reference exclusions, and metrics. Save paired differences. |
| `visualize` | Render randomly selected paired panels with reference, target example, clean modifier and top-5, and each perturbed modifier and its top-5. Green borders mark relevant targets. |

```bash
python run.py --stage prepare
python run.py --stage generate
python run.py --stage retrieve-clean
python run.py --stage perturb
python run.py --stage evaluate
python figure.py  # standalone visualization from saved artifacts
```

Use `--config /path/to/config.yaml` with either entry point. `prepare` and visualization do not require a GPU. Other stages fail clearly if CUDA is required and unavailable.

## Experimental protocol

- `GLAMQueryDataset` follows the original CSV `id1`/`id2` pairing. A row uses its first available reference, retains every existing non-reference target, and preserves the original instruction verbatim for inspection. Missing targets are logged and removed before gallery selection. No images are resized before generation; each model applies its own preprocessing.
- `GLAMGalleryDataset` reads the split membership from `db_processed.txt`, uses sorted unique IDs, and checks that every relevant target is present. It does not silently add out-of-split positives.
- A seeded sample of up to `generation.max_target_images` targets conditions the LLM. The manifest records exactly which targets it saw. All existing targets remain retrieval ground truth, including targets not shown during generation.
- Target observations and synthesis prompts treat targets as alternatives, not objects to combine into one scene. Noisy CSV text is excluded from generation by default (`use_original_hint: false`). The text is still saved for comparison.
- The experiment is **target-conditioned modifier construction**. Because the generator sees relevant target images, results are not a target-blind retrieval benchmark and should not be presented as such.
- By default, the LLM misspells one selected word (`typo_words`), with at most two character edits per word and a configurable maximum word count. Only the approved word spans are changed. Informal/paraphrase perturbations give the LLM a request with protected visual tokens, restore those tokens after rewriting, and validate that all visual-content words remain exact. Surrounding syntax uses the configured function-word vocabulary. These are conservative wording perturbations; set `preserve_content_words: false` to explore broader lexical changes, with additional manual review.
- Textual perturbation sees only the clean modifier and its perturbation instruction. It receives neither target IDs, retrieval rankings, nor performance feedback. The clean modifier is never overwritten.
- Every condition uses `normalize(w * normalize(reference) + (1-w) * normalize(text))`, where `w = retrieval.image_weight`. Gallery image embeddings are normalized too; ranking uses cosine similarity. The reference is excluded by default, and tie ordering is deterministic.
- `Recall@K` matches the original multi-positive `RetrievalRecall` definition: for each query, fraction of all relevant images recovered in the first K, then average across queries. `HitRate@K` is the fraction of queries with any positive in the first K. `mAP` and `MRR` use the complete ranking. `AVG` is the mean of the configured recalls. All values are percentages; perturbation deltas are **perturbed minus clean in percentage points**. Positive deltas mean improvement.
- Quality reviews use another prompt to the same model. They are diagnostic, not independent human validation; false positives and false negatives are possible. Flagged clean queries remain in evaluation to avoid selecting examples based on model judgments or retrieval outcomes. Inspect the saved figures and observations before using generated text in a benchmark. Heterogeneous target groups may not admit one specific modifier that fits every image.

## Models

`generation.model` selects the original `qwen`, `llava`, or `deepseek` family. Qwen2.5-VL-7B-Instruct is the default and is exercised by the smoke run. LLaVA and DeepSeek adapters are available but are not GPU-validated by that run. DeepSeek requires its optional upstream `deepseek_vl` package and explicit `runtime.trust_remote_code: true`.

`retrieval.model` selects `clip`, `openclip`, `blip`, `siglip`, `siglip2`, `coca`, or `openvision`. Checkpoint IDs are all configured in YAML. OpenCLIP ViT-B-16 with OpenAI weights and matching QuickGELU activation is the tested default. BLIP uses its image/text contrastive projection heads through Transformers, removing the original hardcoded external repository dependency. OpenVision uses the upstream OpenCLIP Hugging Face adapter. Other retrieval backends need their own model/environment validation before comparative experiments. Keep text short enough for the chosen encoder's context length; text encoders truncate to their model limits.

## Files and artifacts

| Module | Responsibility |
|---|---|
| `loading_data.py` | Existing-image resolution, CSV filtering, query and gallery datasets |
| `prompts.py` | Target observation, modifier synthesis, perturbation, and review prompts |
| `generation.py` | Lazy GPU model adapters, multi-image prompting, JSON validation and retries |
| `feature_extractor.py` | Image/text embeddings and weighted composition |
| `metrics_computation.py` | Reference masking, deterministic ranking, metrics and paired deltas |
| `run.py` | Stage orchestration, checkpoints, feature caching and reporting |
| `figure.py` | Standalone retrieval visualization |
| `utils.py` | Configuration, seeding, device selection, logging and atomic JSON saves |

Each run saves these files under `paths.output_dir`:

- `manifest.json`, `config.resolved.json`, `environment.json`: frozen query/gallery membership, target conditioning IDs, input signatures, source hashes, resolved settings, library and GPU information.
- `observations.json`, `modifiers.json`: resumable per-image observations, original text, clean text, synthesis rationale/limitations, every perturbation and model-assisted review.
- `quality_report.json`: word counts, unique outputs, review pass counts and flagged query IDs.
- `image_features.pt`: reusable gallery/reference tensors; cache identity includes encoder settings, source hash and image size/mtime.
- `metrics_clean.json`, `retrieval_clean.json`: pre-perturbation evaluation.
- `metrics.json`, `metrics.csv`, `retrieval.json`: all conditions, per-query scores/target ranks, top results and paired aggregate deltas.
- `run.log`: parameters, progress, all LLM responses/retries, metrics and exceptions printed to the terminal and a persistent file.
- `figures/query_*.png`: paired retrieval panels selected reproducibly with the configured seed.

Large weights, caches, generated images and run directories are ignored by Git. A compact smoke-test report is kept in `reports/`.

## Verification

```bash
python -m pip install pytest
python -m pytest -q
python -m compileall -q *.py tests
```

Thirteen tests cover retry behavior, visual-word drift, typo severity, multi-positive recall versus hit rate, ranking/reference masking, missing ground truth, deterministic local-image loading, duplicate CSV pairs, gallery coverage, config mismatch protection, normalized fusion and invalid generated text. The GPU smoke experiment verifies the actual default generation and retrieval backends separately from these synthetic correctness tests.
