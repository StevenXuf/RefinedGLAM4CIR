# RefinedGLAM4CIR

Inference-only modifier refinement and robustness experiments for cultural-heritage
image retrieval, derived from [CIR4DH](https://github.com/StevenXuf/CIR4DH) (MIT).

Install CUDA-compatible PyTorch/torchvision, then install dependencies and configure
paths, models, batch sizes, corruption settings, and cutoffs in [config.yaml](config.yaml).

```bash
pip install -r requirements.txt
python pipeline.py --split test_id --seed 42 --gpu 0
# Run both test splits sequentially:
bash run_splits.sh config.yaml --seed 42 --gpu 0
```

Use existing `test_id`/`test_ood` triplets from `cir_db.csv` and local images
(default: `/data/data_fxu/GLAM/raw`). Every run uses the complete available gallery,
even with sampled queries. Missing targets are removed individually; all available
annotated targets remain relevant.

The pipeline:

1. **Generate:** Qwen3-VL-8B-Instruct compares the reference with up to three seeded
   targets, requesting a minimal shared change in 3–15 everyday English words.
   Only screened outliers receive an additional visual review.
2. **Corrupt:** Produce one seeded variant per query for character deletion,
   insertion, keyboard-neighbour substitution, adjacent-character transposition,
   repetition, space insertion, and space deletion. Python/NLPAug applies
   75% mild, 20% moderate, 4% severe, and 1% extreme quotas, with edit fractions
   sampled uniformly within each configured severity range.
3. **Retrieve:** Rank gallery images by cosine similarity using
   `L2_normalize(unit_image + unit_text)`, excluding the reference. Baselines use
   image only, original/generated text only, and image + original text.
   TorchMetrics `RetrievalRecall`, `RetrievalMRR`, and `RetrievalMAP` report
   percentages at K=1,5,10,50 and their arithmetic average.

Select `--extractor openclip|clip|siglip2` for ViT-B/32 retrieval; checkpoints are
configured in YAML. SigLIP is also supported via `--extractor siglip`.
Use `--stage data|generate|perturb|evaluate|visualize` for individual components.
The default `--stage all` regenerates modifiers and corruptions before retrieval.

Outputs include modifiers, corruptions, metrics, one log per run, and three example
PDFs, with model/seed filenames, all under `outputs/test_id/` or `outputs/test_ood/`.
Modifiers and corruptions are shared across extractors for each seed. Outputs and
caches are ignored by Git.

Analyze completed runs:

```bash
python report_seeds.py --seeds 42 43 44
python generation.py --seeds 42 43 44
```

The first command reports per-seed metrics and means/sample SDs. The second reports
modifier lengths and vocabulary counts, and saves original/per-seed word-cloud PDFs.
Pass `--extractor` to either command when analyzing another retrieval model.
