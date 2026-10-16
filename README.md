# PerturbedCIR4GLAM

PerturbedCIR4GLAM measures how realistic changes in search wording affect composed image retrieval on the GLAM collection. A vision-language model inspects a reference image and its annotated target images to write an improved textual modifier. A retrieval model combines the reference image and modifier to rank the gallery. The language model then introduces spelling errors or changes the wording, and retrieval is repeated against the same complete gallery.

This project performs inference only. It does not train or fine-tune any model. It is derived from [StevenXuf/CIR4DH](https://github.com/StevenXuf/CIR4DH) and retains its MIT license.

## Data protocol

Only `test_id` and `test_ood` are supported. Training, validation, and other splits are rejected.

- **`cir_db.csv` defines the queries.** Each row contains a comma-separated `id1` reference field, comma-separated `id2` target field, the original `query` modifier, and a `partition`. Reference/target relationships come exclusively from that row. The first existing reference is used, matching the original loader. Every existing non-reference `id2` target is retained and used for modifier generation and retrieval ground truth. No targets are sampled and no rows are removed because they have many targets.
- **`db_processed.txt` defines gallery partition membership.** Its first column is an image ID and its second column is the partition. The loader selects every existing unique image in the requested test partition, matching the original availability filtering. It never uses this file to invent reference/target pairings.
- **The gallery always contains every available image, including during query-subset checks.** There is no gallery-size setting. A legacy non-null `gallery_limit` is rejected. Missing local files are logged and listed in the manifest. The default `missing_gallery_policy: available` matches the original loader and uses every existing image. Set this policy to `strict` to require every declared file and fail if any is absent. For each query, its own reference can be masked from retrieval according to `evaluation.exclude_reference`.
- Images are resolved by their selected test IDs under `/data/data_fxu/GLAM/raw`. Physical storage-folder names are not annotation partitions: many IDs marked test_id in the metadata are stored under train/val directories. `image_storage_folders` controls path lookup only; only IDs selected from test annotations can enter either query generation or the gallery. No training/validation annotation partitions are evaluated, and no GLAM images are downloaded.
- Duplicate reference/target groups are removed when `remove_duplicates: true`, following the original configuration. The original modifier is preserved verbatim. Image-path checks and row counts are logged.

The metadata declares these sizes before local image-availability checks, with the default pair deduplication:

| Split | Raw CSV rows | Unique CSV pairs | Declared gallery |
|---|---:|---:|---:|
| test_ood | 88 | 81 | 22,373 |
| test_id | 2,559 | 2,002 | 2,631 |

The current local test_id data is missing 268 declared gallery images; see [data availability](reports/data_availability.md). Following the original loader, its available dataset contains 1,429 valid query pairs and all 2,363 existing gallery images. The full test_ood dataset contains 81 valid pairs and all 22,373 declared gallery images. These availability counts are reported with the metrics; no arbitrary gallery or target-count sampling is applied.

The default full runs use every valid query. `subset.query_limit` can restrict queries for a functional check; it has no effect on gallery membership or target coverage. Set `remove_duplicates: false` if repeated CSV pairs should count as separate queries.

## Installation and full evaluation

Use Python 3.10+ and a CUDA-compatible PyTorch/torchvision installation, then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
python run.py --stage both
```

On the existing server, `/home/fxu/my-env/bin/python` is the tested interpreter. All experiment settings are in `config.yaml`. By default, the two test splits run in separate processes on GPUs 2 (`test_ood`) and 1 (`test_id`); adjust `runtime.gpu_ids` for your server. The runner waits for both jobs and prints a combined Markdown table containing unperturbed, typo, informal, and paraphrase results for both splits.

Run either split independently:

```bash
python run.py --stage all --split test_ood
python run.py --stage all --split test_id
```

Full results and tables are written to `outputs/full/results.md`, `results.csv`, and `results.json`. Per-split artifacts are under `outputs/full/test_ood/` and `outputs/full/test_id/`. Published results are summarized in [reports/full_results.md](reports/full_results.md).

Paths are resolved relative to the configuration file. To repeat an experiment with different settings or source code, choose a new `paths.output_dir`; signatures prevent incompatible checkpoints from being mixed. Model weights are cached in `.model_cache/`. CUDA is required by default. If missing assets are restored later, use a new output directory; the manifest guard detects the availability change.

## How the experiment works

1. Freeze query and full available-gallery manifests from the two metadata files. Record declared/available/missing counts and verify coverage of every available positive target.
2. Describe the reference and every corresponding target image with the configured vision-language model. A per-image caption cache avoids describing the same image repeatedly. Only images explicitly present in CSV query pairs enter this generation cache.
3. Synthesize a modifier from the reference observation and all target observations from that same CSV row. The original low-quality modifier is retained for inspection and can optionally be supplied as a noisy hint. Target images are alternatives, not objects to merge into one scene. Synthesis includes shared traits and limitations for heterogeneous groups.
4. Retrieve using the reference image plus the generated unperturbed modifier. Save the clean metrics before perturbation.
5. Use the language model to generate spelling errors, informal wording, and paraphrases. Spelling edits are applied only to selected word spans and checked against edit budgets. By default, rephrasing protects complete visual phrases, preserving their content and order while allowing function-word/syntax changes. Set `perturbation.preserve_content_words: false` for broader lexical paraphrases, which need additional semantic review.
6. Retrieve again with each perturbed modifier, holding the reference image, complete gallery, ground truth, fusion, exclusions, and metrics fixed. Report perturbed-minus-unperturbed changes in percentage points.
7. Save random paired examples showing the reference, a relevant target, each modifier, and its top-5 retrieved images. Green borders identify relevant results.

Qwen inference is batched for both single-image observations and text generation. Invalid items are retried individually; successful items are checkpointed before failures are reported. Other model adapters use serial inference. If a protected-phrase rewrite repeatedly fails, the LLM selects among validated conservative surface forms using constrained token decoding. This preserves the visual content and provides a reliable inference fallback. Failed items are never silently omitted from the reported full query set.

## Metrics

Each query uses `normalize(w * normalize(reference_image) + (1-w) * normalize(modifier_text))`, with `w = retrieval.image_weight`. Gallery image embeddings are normalized; ranking uses cosine similarity and deterministic tie ordering.

- `Recall@K`: fraction of all relevant images recovered in the first K for each query, averaged across queries. This retains the original multi-positive `RetrievalRecall` definition.
- `HitRate@K`: fraction of queries with at least one relevant image in the first K.
- `mAP` and `MRR`: average precision and reciprocal rank from the complete ranking.
- `AVG`: mean of the configured Recall@K values, by default K = 1, 5, 10, 50.

All metrics are percentages. Deltas are **perturbed minus unperturbed, in percentage points**. A positive delta means improvement. Full-gallery metrics replace the earlier experimental reports that used a sampled gallery; those earlier reports are not comparable with the current protocol.

## Quality checks and interpretation

The default diagnostic audit reviews six seeded queries per split, and every perturbation type for those queries. This audit sampling affects only quality-review calls: every query is still generated, perturbed, and evaluated, and every annotated target is used for generation. `generation.review_target_images` limits images displayed in the diagnostic review, while synthesis always receives observations of all targets. Set `quality_review_samples: null` to audit every query.

A second prompt to the same model is not independent human validation. Review failures are recorded and flagged, and do not select or exclude queries from evaluation. Inspect the saved target observations, synthesis limitations, review reasons, and figures. Heterogeneous target sets and inferred materials can still produce inaccurate modifiers.

Because the modifier generator sees the annotated targets, this experiment measures robustness of target-conditioned modifier construction and retrieval. It should not be described as a target-blind query-generation benchmark.

## Modular scripts and individual stages

| File | Purpose |
|---|---|
| `loading_data.py` | Test-only CSV query pairing and complete split-gallery loading |
| `prompts.py` | Image-description, modifier, perturbation, and diagnostic-review prompts |
| `generation.py` | Qwen/LLaVA/DeepSeek adapters, batched generation, parsing and perturbation validation |
| `feature_extractor.py` | Retrieval encoders and reference/text composition |
| `metrics_computation.py` | Ranking, multi-positive metrics, and deltas |
| `run.py` | Resumable per-split stages and artifact management |
| `results.py` | Two-GPU test-split runner and combined result tables |
| `parallel_perturbation.py` | Optional query partitions for text inference across configured GPUs, validated merging before evaluation |
| `figure.py` | Standalone paired top-5 visualization from saved results |
| `utils.py` | Configuration, signatures, device selection, seeding and logging |

```bash
python run.py --stage prepare --split test_ood
python run.py --stage generate --split test_ood
python run.py --stage retrieve-clean --split test_ood
python run.py --stage perturb --split test_ood
python run.py --stage evaluate --split test_ood
python figure.py --split test_ood
python figure.py --split test_id
python results.py  # print/save the combined table from completed runs
```

For a large split, after generation and clean retrieval, `python parallel_perturbation.py --split test_id` can replace the single-process perturbation stage. It uses the unique GPUs listed in `runtime.gpu_ids`, writes independent checkpoints, validates every original query/target/clean-text identity during merging, and preserves CSV order. Each worker retains the full available gallery manifest. Then resume `python run.py --stage all --split test_id` for the global seeded quality audit, final evaluation, and figures. Do not run another perturbation writer for the same split concurrently.

Every entry point accepts `--config /path/to/config.yaml`. `prepare`, `figure.py`, and result-table collection do not require GPU inference.

The original generation families `qwen`, `llava`, and `deepseek` are configurable. Qwen2.5-VL-7B-Instruct is the default. DeepSeek needs its optional upstream package and explicit `runtime.trust_remote_code: true`. Retrieval choices include `clip`, `openclip`, `blip`, `siglip`, `siglip2`, `coca`, and `openvision`; checkpoint IDs are in YAML. The tested default is OpenCLIP ViT-B-16/OpenAI with matching QuickGELU. Other backends require their own environment validation.

## Saved artifacts and logs

Each split saves `manifest.json`, resolved configuration, environment information, per-image `observations.json`, `modifiers.json`, diagnostic `quality_report.json`, reusable image features, clean and paired metrics, per-query target ranks/top results, and `figures/query_*.png`. Full logs include LLM responses, validation retries, progress and metrics. `progress.json` supports monitoring without reading entire logs.

`outputs/full/both.log` reports both worker states; `test_id.console.log` and `test_ood.console.log` contain worker output. Important results and examples are copied into `reports/` after validation. Large model weights, feature tensors and working run directories are ignored by Git.

## Tests

```bash
python -m pip install pytest
python -m pytest -q
python -m compileall -q *.py tests
```

Twenty-six tests cover complete-gallery enforcement even with a query subset, CSV-only target conditioning, forbidden splits and gallery/target filters, missing/restored gallery images, original availability filtering, duplicate pairs, batched retry ordering, protected perturbations, and multi-positive ranking/metrics. GPU checks exercise the actual default model backends.
