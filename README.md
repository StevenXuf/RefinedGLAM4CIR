# RefinedGLAM4CIR

Inference-only experiments on refining cultural-heritage image-search modifiers and
measuring their robustness to typing errors. Derived from
[CIR4DH](https://github.com/StevenXuf/CIR4DH) under the MIT license.

## Run

Install CUDA-compatible PyTorch/torchvision, then:

```bash
python -m pip install -r requirements.txt
python pipeline.py --split test_id --seed 42 --gpu 0
python pipeline.py --split test_ood --seed 42 --gpu 1
# Or run both splits sequentially:
bash run_splits.sh
```

Paths, model versions, seeds, batches, severity distributions, and cutoffs are in
`config.yaml`; prompts are in `prompts.py`. Defaults are Qwen3-VL-8B-Instruct and
OpenCLIP ViT-B-16 pretrained on `laion2b_s34b_b88k`.

## Pipeline

1. Read existing reference–target triplets from `cir_db.csv`. Qwen jointly sees
   the reference and up to three seeded targets and writes a concrete 5–15-word
   search request. Generation uses the images directly, without intermediate
   descriptions or predefined fallback requests.
2. For each clean request, Qwen directly generates character deletion, insertion,
   keyboard-neighbour substitution, adjacent-character transposition, repetition,
   space insertion, and space deletion. Each type starts from the clean text.
3. Qwen receives both the clean and perturbed text and judges operation, actual
   change, approximate severity, and preservation of the intended search. Each
   rejection is regenerated once using the reason, then judged again. Final
   failures are retained and marked, preserving the evaluation cohort. Model
   judgments are not human verification.
4. Rank the complete available gallery using cosine similarity. Image and text
   embeddings are individually L2-normalized; composed features use
   `normalize(image_embedding + text_embedding)`.

Severity is sampled categorically: mild/moderate/severe/extreme have probabilities
75/20/4/1%. A uniform fraction within that level's configured edit range is given
to Qwen. These are sampling probabilities, not exact quotas. There are no
Python-generated corruption candidates or enforced character-edit budgets.

Baselines are image only, original modifier only, generated modifier only, and
image + original modifier. They are compared with image + generated modifier and
all seven corruptions. Native TorchMetrics `RetrievalRecall`, `RetrievalMRR`, and
`RetrievalMAP` report percentages at K=1,5,10,50 and their arithmetic average.
Multiple targets are retained; missing targets are removed individually. The
reference is excluded from retrieval, with deterministic tie handling.

Images are loaded locally from `/data/data_fxu/GLAM/raw`; nothing is downloaded.
Only `test_id` and `test_ood` have CIR triplets. `db_processed.txt` defines the
full galleries, which are never sampled. Current available coverage is
1,429 queries / 2,363 images for test_id and 81 / 22,373 for test_ood; 268 declared
test_id gallery files are missing locally.

## Components and results

`--stage data|generate|perturb|evaluate|visualize` runs a component separately.
Generation writes `modifiers_qwen_seed_42.json`; perturbation/review writes
`perturbations_qwen_seed_42.json`. Later stages read these files. The default
`all` stage regenerates the experiment from the beginning.

Each `outputs/<split>/` also contains final
`metrics_qwen_openclip_seed_42.json`, one `run_qwen_openclip_seed_42.log`, and
three example figures under `figures/`. Metrics include initial/final success and failure counts, recovery counts, and success rates;
corruption records retain failed first attempts and final judgments. Figures show
original, generated, and corrupted modifiers with top-5 results; every annotated
retrieved target is highlighted. All filenames include their models and seed.
Outputs, model caches, and local tests are ignored by Git.

For completed independent seeds, `python report_seeds.py --seeds 42 43 44` prints
per-seed tables and mean ± sample standard deviation without writing extra reports.
