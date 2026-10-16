# Full GLAM test inference

All values are percentages. Each condition uses every available image from the corresponding test gallery, without sampling.

| Split | Queries | Gallery | Condition | Recall@1 | Recall@5 | Recall@10 | Recall@50 | AVG | mAP | MRR |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| test_ood | 81 | 22373 | unperturbed | 2.75 | 4.94 | 9.39 | 14.01 | 7.77 | 5.16 | 8.89 |
| test_ood | 81 | 22373 | typos | 2.75 | 6.18 | 9.39 | 13.81 | 8.03 | 5.16 | 8.83 |
| test_ood | 81 | 22373 | informal | 2.75 | 5.31 | 9.88 | 14.13 | 8.02 | 5.19 | 8.94 |
| test_ood | 81 | 22373 | paraphrase | 2.75 | 6.42 | 9.88 | 13.95 | 8.25 | 5.36 | 9.07 |
| test_id | 1429 | 2363 | unperturbed | 4.23 | 14.17 | 21.91 | 44.89 | 21.30 | 11.27 | 13.77 |
| test_id | 1429 | 2363 | typos | 4.38 | 14.26 | 21.37 | 44.73 | 21.19 | 11.42 | 13.90 |
| test_id | 1429 | 2363 | informal | 4.06 | 14.39 | 21.78 | 44.27 | 21.13 | 11.21 | 13.78 |
| test_id | 1429 | 2363 | paraphrase | 4.60 | 14.51 | 22.60 | 45.58 | 21.83 | 11.65 | 14.14 |

Gallery availability:

- test_ood: 22,373 available / 22,373 declared; 0 missing.
- test_id: 2,363 available / 2,631 declared; 268 missing.


Pair deduplication follows the original loader configuration: test_id has 2,559 raw CSV rows and test_ood has 88. Query counts in the table are after the configured deduplication and missing-image checks. No target-count filtering or query subsampling is used in these full runs.

Recall is macro multi-positive recall. AVG is the mean of Recall@1/5/10/50. Clean and perturbed queries use identical reference-image/text fusion, gallery, ground truth, and reference exclusions. Modifiers are target-conditioned from the CSV pairs. The LLM quality audit is diagnostic and sampled; it does not certify caption accuracy.


The test_id run follows the original loader's missing-file filtering: 726 raw query rows lack an available reference or target, leaving 1,429 valid unique query pairs. All 2,363 existing gallery images are used; 268 declared files are absent. No query-count limit, target-count filter, or gallery sampling was applied. See [availability details](data_availability.md) and [missing IDs](missing_test_id_images.json).

The default Qwen2.5-VL-7B and OpenCLIP ViT-B-16/OpenAI backends ran on RTX A6000 GPUs. All generation reference/target relationships come from the same cir_db.csv row. Every existing CSV target was described and included in synthesis. The diagnostic quality review audits six seeded queries per split; it is not human validation. Twenty-six tests, Ruff, and Python compilation passed.

Example comparisons: [test_ood](test_ood/query_398.png), [test_id](test_id/query_514.png). Each image includes unperturbed and perturbed top-5 retrieval.
