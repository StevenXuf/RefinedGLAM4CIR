# Test image availability

Test partitions are defined by the metadata, independently of physical storage-folder names. Only test_id/test_ood annotations are selected. Selected test IDs may resolve to files stored under train/val folders; unrelated training/validation IDs are never included.

| Split | Raw query rows | Distinct metadata pairs | Locally valid query pairs | Declared gallery | Every available gallery image | Missing gallery files |
|---|---:|---:|---:|---:|---:|---:|
| test_ood | 88 | 81 | 81 | 22,373 | 22,373 | 0 |
| test_id | 2,559 | 2,002 | 1,429 | 2,631 | 2,363 | 268 |

The test_id missing-ID list is [saved here](missing_test_id_images.json). The files were not found under any provided raw storage folder, including alternate extensions. No images were downloaded.

The original loader filters absent files before deduplicating pairs. For test_id, 726 raw rows have no existing reference or no existing target. After that rule and pair deduplication, 1,429 valid pairs remain. Every existing target from each retained row is used. All 2,363 available gallery images are evaluated, without any gallery sampling or target-count filter.

The initial strict availability check stopped test_id. The final configuration follows the original loader's available-image policy and evaluates its whole locally valid query/gallery sets. Results must be interpreted with this coverage limitation: they do not represent evaluation on all 2,631 declared images. Set `missing_gallery_policy: strict` to enforce complete declared coverage instead. If missing files are restored, use a new output directory to regenerate the complete dataset; stale availability snapshots are rejected.
