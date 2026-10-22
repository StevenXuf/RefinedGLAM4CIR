"""Multi-positive macro recall, hit rate, mAP, and MRR with reference masking.

Recall follows the original torchmetrics RetrievalRecall definition: recovered
relevant images / all relevant images, averaged over queries, reported in percent.
"""

import logging

import torch
import torch.nn.functional as F
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalMRR, RetrievalRecall


def rank_queries(query_features, gallery_features, gallery_ids, queries, exclude_reference=True):
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('Duplicate gallery IDs')
    if len(queries) != len(query_features) or len(gallery_ids) != len(gallery_features):
        raise ValueError('Feature/metadata length mismatch')
    scores = (
        F.normalize(query_features.float(), dim=-1)
        @ F.normalize(gallery_features.float(), dim=-1).T
    )
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
        rankings.append(
            [
                gallery_ids[i]
                for i in order[row].tolist()
                if not (exclude_reference and gallery_ids[i] == query['reference_id'])
            ]
        )
    return rankings, scores


def compute_metrics(rankings, queries, top_ks, gallery_ids, exclude_reference=True):
    """Compute unmodified TorchMetrics module outputs, grouping candidates by query."""
    if not rankings or len(rankings) != len(queries):
        raise ValueError('Need aligned, nonempty rankings and queries')
    if (
        not top_ks
        or any(type(k) is not int or k < 1 for k in top_ks)
        or len(set(top_ks)) != len(top_ks)
    ):
        raise ValueError('top_ks must contain positive integers')
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('Duplicate gallery IDs')
    available = set(gallery_ids)
    predictions, targets, query_indexes = [], [], []
    for query_index, (ranking, query) in enumerate(zip(rankings, queries)):
        if len(set(ranking)) != len(ranking):
            raise ValueError('Rankings must contain unique image IDs')
        eligible = available - {query['reference_id']} if exclude_reference else available
        if set(ranking) != eligible:
            raise ValueError(f'Need the complete available gallery ranking for {query["query_id"]}')
        relevant = set(query['target_ids']) & eligible
        if not relevant:
            logging.getLogger(__name__).warning(
                'Skipping %s: no available relevant targets', query['query_id']
            )
            continue
        # All metrics depend on order, not score magnitudes. Positive unique rank
        # scores preserve stable cosine ordering (including ties) without TorchMetrics
        # discarding relevant candidates whose actual cosine is zero or negative.
        predictions.append(torch.arange(len(ranking), 0, -1, dtype=torch.float32))
        targets.append(torch.tensor([name in relevant for name in ranking], dtype=torch.bool))
        query_indexes.append(torch.full((len(ranking),), query_index, dtype=torch.long))
    if not predictions:
        raise ValueError('No queries have available relevant targets')
    preds, target, indexes = map(torch.cat, (predictions, targets, query_indexes))
    results = {}
    for family, metric_class in (
        ('Recall', RetrievalRecall),
        ('MRR', RetrievalMRR),
        ('mAP', RetrievalMAP),
        ('HitRate', RetrievalHitRate),
    ):
        for k in top_ks:
            metric = metric_class(top_k=k, aggregation='mean', empty_target_action='error')
            results[f'{family}@{k}'] = 100 * metric(preds, target, indexes=indexes).item()
        results[f'{family}_AVG'] = sum(results[f'{family}@{k}'] for k in top_ks) / len(top_ks)
    return results


def metric_deltas(clean, perturbed):
    return {key: value - clean[key] for key, value in perturbed.items()}
