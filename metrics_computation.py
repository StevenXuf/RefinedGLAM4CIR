"""Multi-positive macro recall, hit rate, mAP, and MRR with reference masking.

Recall follows the original torchmetrics RetrievalRecall definition: recovered
relevant images / all relevant images, averaged over queries, reported in percent.
"""
import logging

import torch
import torch.nn.functional as F
from torchmetrics.functional.retrieval import (retrieval_average_precision,
                                              retrieval_hit_rate,
                                              retrieval_recall,
                                              retrieval_reciprocal_rank)


def rank_queries(query_features, gallery_features, gallery_ids, queries, exclude_reference=True):
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('Duplicate gallery IDs')
    if len(queries) != len(query_features) or len(gallery_ids) != len(gallery_features):
        raise ValueError('Feature/metadata length mismatch')
    scores = F.normalize(query_features.float(), dim=-1) @ F.normalize(gallery_features.float(), dim=-1).T
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite retrieval scores')
    index = {name: i for i, name in enumerate(gallery_ids)}
    for row, query in enumerate(queries):
        if exclude_reference and query['reference_id'] in query['target_ids']:
            raise ValueError('Reference image is also a relevant target')
        if exclude_reference and query['reference_id'] in index:
            scores[row, index[query['reference_id']]] = -torch.inf
    # Stable order gives deterministic handling of ties.
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    rankings = []
    for row, query in enumerate(queries):
        rankings.append([gallery_ids[i] for i in order[row].tolist()
                         if not (exclude_reference and gallery_ids[i] == query['reference_id'])])
    return rankings, scores


def compute_metrics(rankings, queries, top_ks, gallery_ids, exclude_reference=True):
    if not rankings or len(rankings) != len(queries):
        raise ValueError('Need aligned, nonempty rankings and queries')
    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError('top_ks must contain positive integers')
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('Duplicate gallery IDs')
    available = set(gallery_ids)
    per_query = []
    for ranking, query in zip(rankings, queries):
        if len(set(ranking)) != len(ranking):
            raise ValueError('Rankings must contain unique image IDs')
        eligible = available - {query['reference_id']} if exclude_reference else available
        if set(ranking) != eligible:
            raise ValueError(f"Need the complete available gallery ranking for {query['query_id']}")
        relevant = set(query['target_ids']) & eligible
        if not relevant:
            logging.getLogger(__name__).warning('Skipping %s: no available relevant targets', query['query_id'])
            continue
        target = torch.tensor([name in relevant for name in ranking], dtype=torch.bool)
        # All metrics depend on order, not score magnitudes. Positive unique rank
        # scores preserve stable cosine ordering (including ties) without TorchMetrics
        # discarding relevant candidates whose actual cosine is zero or negative.
        preds = torch.arange(len(ranking), 0, -1, dtype=torch.float32)
        row = {'query_id': query['query_id'],
               'MRR': 100 * retrieval_reciprocal_rank(preds, target).item(),
               'mAP': 100 * retrieval_average_precision(preds, target).item()}
        for k in top_ks:
            row[f'Recall@{k}'] = 100 * retrieval_recall(preds, target, top_k=k).item()
            row[f'HitRate@{k}'] = 100 * retrieval_hit_rate(preds, target, top_k=k).item()
        row['AVG'] = sum(row[f'Recall@{k}'] for k in top_ks) / len(top_ks)
        per_query.append(row)
    if not per_query:
        raise ValueError('No queries have available relevant targets')
    means = {key: sum(row[key] for row in per_query) / len(per_query)
             for key in per_query[0] if key != 'query_id'}
    return means, per_query


def metric_deltas(clean, perturbed):
    return {key: value - clean[key] for key, value in perturbed.items()}
