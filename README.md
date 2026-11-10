# RefinedGLAM4CIR

Inference-only refinement and robustness experiments for cultural-heritage image
retrieval, derived from [CIR4DH](https://github.com/StevenXuf/CIR4DH) (MIT).

Install CUDA-compatible PyTorch/torchvision, then:

```bash
pip install -r requirements.txt
python pipeline.py --split test_id --seed 42 --gpu 0
python pipeline.py --split test_ood --seed 42 --gpu 1
# Or run both splits sequentially:
bash run_splits.sh
```

Set paths, model versions, batches, severity proportions, edit ranges, and metric
cutoffs in `config.yaml`. Only existing `test_id`/`test_ood` triplets from
`cir_db.csv` are used. Images are loaded from `/data/data_fxu/GLAM/raw`; nothing
is downloaded. Every run uses the complete available gallery, including when
queries are sampled. Current coverage is 1,429 queries / 2,363 gallery images for
test_id (268 declared images missing) and 81 / 22,373 for test_ood.

Qwen3-VL-8B-Instruct directly compares each reference with up to three seeded
targets. The prompt asks for the smallest clear shared difference, leaving unchanged
reference features implicit: 3–15 everyday English words, aiming for six. Language
identification (`langid`), word count, and the opening action flag outliers;
ordinary outputs need no extra Qwen call. Only outliers are
reviewed and corrected against the same images in one extra Qwen call. Corrections
must satisfy the word limit; short English phrases can confuse language detection,
so language screening is not repeated after Qwen's review.
Saved records indicate whether Qwen review was needed. Every available annotated
target remains relevant during evaluation.

Python then produces **one corruption per query and type**: character deletion,
insertion, keyboard-neighbour substitution, adjacent-character transposition,
repetition, space insertion, and space deletion. NLPAug handles deletion,
insertion, and keyboard errors; Python's standard library handles the others.
There is no model-based corruption generation, verification, retry, or fallback.

Severity labels use fixed 75% mild / 20% moderate / 4% severe / 1% extreme quotas,
rounded by largest remainder and shuffled with the experiment seed. The same
query has the same severity across error types. Within its severity's configured
range, a uniform fraction determines `ceil(fraction × eligible positions)` edits,
with at least one when possible. Swaps use disjoint adjacent pairs; spaces use
eligible boundaries. Short strings can give identical edit counts for different
severity levels. Per-query RNGs make the perturbations reproducible.
If a request has no eligible positions, it is retained with zero edits.

OpenCLIP ViT-B-16 (`laion2b_s34b_b88k`) encodes images/text. Composition is
`L2_normalize(unit_image + unit_text)`, followed by cosine ranking with the
reference excluded. Baselines include image only, original/generated text only,
and image + original text. Native TorchMetrics `RetrievalRecall`, `RetrievalMRR`,
and `RetrievalMAP` report percentages at K=1,5,10,50 and their arithmetic average.
Missing targets are removed individually; multiple available targets are retained.

Use `--stage data|generate|perturb|evaluate|visualize` for individual components.
`all` runs the complete pipeline. Outputs contain clean modifiers, their
corruptions and edit details, final metrics, one log, and three figures per split.
Filenames include the models and experiment seed, such as
`perturbations_qwen_seed_42.json` and `metrics_qwen_openclip_seed_42.json`.
Figures compare original, generated, and corrupted modifiers with top-5 results;
any annotated target is highlighted. Outputs, caches, and local tests are ignored
by Git. `report_seeds.py --seeds 42 43 44` summarizes completed independent runs.

Run `python generation.py --seeds 42 43 44` to analyze saved modifiers for both
splits. It reports word/character lengths and vocabulary counts per seed, their
means and SDs across seeds, and the combined generated vocabulary. Word clouds
are saved for the valid originals and each generated seed.
Words use whitespace splitting; characters include spaces and punctuation.
Vocabulary counts ignore case, count alphabetic words with contractions, retain
stopwords, and split hyphenated words.
Clouds use WordCloud's English stopwords. All figures are saved as PDFs with white
backgrounds, for example `wordcloud_generated_qwen_seed_42.pdf` and
`example_514_qwen_openclip_seed_42.pdf`.
