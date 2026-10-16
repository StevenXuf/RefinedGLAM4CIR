"""Multi-positive macro recall, hit rate, mAP, and MRR with reference masking.

Recall follows the original torchmetrics RetrievalRecall definition: recovered
relevant images / all relevant images, averaged over queries, reported in percent.
"""
import torch
import torch.nn.functional as F


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
        if not query['target_ids'] or set(query['target_ids']) - index.keys():
            raise ValueError(f"Missing ground truth in gallery for {query['query_id']}")
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


def compute_metrics(rankings, queries, top_ks):
    if not rankings or len(rankings) != len(queries):
        raise ValueError('Need aligned, nonempty rankings and queries')
    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError('top_ks must contain positive integers')
    per_query = []
    for ranking, query in zip(rankings, queries):
        relevant = set(query['target_ids'])
        if not relevant or relevant - set(ranking):
            raise ValueError(f"Incomplete ground truth for {query['query_id']}")
        if len(set(ranking)) != len(ranking):
            raise ValueError('Rankings must contain unique image IDs')
        hits = [name in relevant for name in ranking]
        cumulative, ap, first = 0, 0.0, None
        for rank, hit in enumerate(hits, 1):
            if hit:
                cumulative += 1
                ap += cumulative / rank
                if first is None:
                    first = rank
        row = {'query_id': query['query_id'], 'MRR': 100 / first, 'mAP': 100 * ap / len(relevant)}
        for k in top_ks:
            found = sum(hits[:k])
            row[f'Recall@{k}'] = 100 * found / len(relevant)
            row[f'HitRate@{k}'] = 100 * bool(found)
        row['AVG'] = sum(row[f'Recall@{k}'] for k in top_ks) / len(top_ks)
        per_query.append(row)
    means = {key: sum(row[key] for row in per_query) / len(per_query)
             for key in per_query[0] if key != 'query_id'}
    return means, per_query


def metric_deltas(clean, perturbed):
    return {key: value - clean[key] for key, value in perturbed.items()}
