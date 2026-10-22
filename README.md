# PerturbedGLAM4CIR

This inference-only project studies how textual mistakes affect composed image retrieval on GLAM. A multimodal language model turns a reference image and its annotated targets into a short, concrete search request. Retrieval is evaluated before and after seven controlled text corruptions. The code derives from [CIR4DH](https://github.com/StevenXuf/CIR4DH) under the MIT license.

## Run

Install a CUDA-compatible PyTorch/torchvision build, then:

```bash
python -m pip install -r requirements.txt
python pipeline.py --split test_ood
python pipeline.py --split test_id
# Or run both splits sequentially:
bash run_splits.sh
```

The server's project interpreter is `.venv/bin/python`. All paths, models, GPU settings, seeds, generation limits, and evaluation cutoffs are in `config.yaml`. Defaults are **Qwen3-VL-8B-Instruct**, **OpenCLIP ViT-B-16 / laion2b_s34b_b88k**, GPU 1, and seed 42. Set `generation.model` to `qwen`, `llava`, or `deepseek` for the configured 7–8B backends; retrieval alternatives are listed in YAML.

## Data and generation

- `cir_db.csv` supplies the existing reference–target triplets and original modifiers. Only `test_id` and `test_ood` have CIR annotations. `db_processed.txt` defines their galleries.
- Images are read locally from `/data/data_fxu/GLAM/raw`; nothing is downloaded. Every query uses the **complete available split gallery**, including when testing a query subset.
- Missing targets are removed individually; all remaining targets are retained for evaluation. Duplicate pairs are filtered. Current coverage is **1,429 queries / 2,363 available gallery images** for test_id and **81 / 22,373** for test_ood. Another 268 declared test_id gallery files are missing locally.
- Generation observes up to **three seeded target images** per query. Requests use plain language, a specific shared object name, and supported visible details, in **5–15 words**. The model checks each sampled target and repairs unsupported claims. After free-form repairs, the model can choose a specific category request from YAML; rejected choices are excluded and every choice requires review. Heterogeneous targets can still require a broad request. Generation is target-conditioned and model review does not guarantee semantic correctness.

The corruptions are character deletion, insertion, keyboard-neighbour substitution, adjacent character transposition, repetition, space insertion, and space deletion. Each starts from the clean request. Severity is sampled as mild/moderate/severe/extreme with probabilities 75/20/4/1%; edit ranges and protected-word rules are configurable or defined in `perturbation.py`. The MLLM selects among validated candidates at the fixed sampled budget.

## Retrieval and metrics

Image and text encoders produce normalized shared-space features. Composed queries use:

```text
normalize(w * normalize(reference) + (1-w) * normalize(text))
```

`w=0.5` by default. Candidates are ranked by cosine similarity, excluding the reference with deterministic ties. BLIP uses its contrastive retrieval projections, not its classification or language-model heads.

Baselines use the image alone, original text alone, generated text alone, and image + original text. They are compared with image + generated text and each corruption on the same queries and gallery.

Native TorchMetrics **RetrievalRecall, RetrievalMRR, and RetrievalMAP** compute macro scores at **K=1,5,10,50**. AVG is the mean across these cutoffs. Scores are percentages; deltas compare against clean generated-text composition. TorchMetrics 1.8.0 AP@K averages precision over positives retrieved within K, with no custom rescaling. Positive rank-equivalent metric inputs preserve cosine ordering for candidates with nonpositive scores; figures display actual cosine similarities.

## Outputs

Each `outputs/<split>/` contains only:

- `modifiers_qwen_seed_42.json`: clean and perturbed requests, triplets, and sampled generation targets; updated as a resumable checkpoint.
- `metrics_qwen_openclip_seed_42.json`: final scores, deltas, coverage, and experiment settings.
- `run_qwen_openclip_seed_42.log`: the latest invocation's messages and final tables, also printed on screen. Only one run log is retained per split.
- `figures/example_<csv_row>_qwen_openclip_seed_42.png`: three random triplets showing original, generated, and corrupted requests with top-5 results. Any annotated target receives a green border.

Model/seed suffixes follow YAML automatically. Clean baselines are evaluated once and reused; final metrics are saved once. Features, full rankings, image observations, and review responses are not exported. Outputs, model caches, and local checks are ignored by Git.

## Components and resuming

`pipeline.py` assembles `loading_data.py`, `generation.py` / `prompts.py`, `perturbation.py`, `feature_extractor.py`, `metrics_computation.py`, and `figure.py`. Components can run through `--stage data|generate|perturb|evaluate|visualize`; all entry points accept `--config`.

Unchanged runs resume checkpoints. Configuration, source, or image-availability changes require an empty output directory. `--reuse-clean <file>` or `--reuse-modifiers <file>` explicitly import compatible checkpoints after validation; older generation policies are rejected. `bash run_splits.sh config.yaml /path/to/previous_outputs` imports compatible modifiers for both splits. To regenerate from scratch, use an empty output directory and omit reuse options.
