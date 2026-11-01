"""Direct Qwen corruptions, pairwise review, and one revision after rejection."""

import random
from collections import Counter

import prompts
from utils import batches, fingerprint


def severity_profile(cfg, record, kind):
    """Draw a categorical severity and then a uniform fraction within that level."""
    rng = random.Random(
        int(fingerprint([cfg['seed'], record['query_id'], kind, record['modifier']]), 16)
    )
    settings = cfg['perturbation']
    probabilities = settings['severity_probabilities']
    severity = rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
    return dict(
        severity=severity, requested_fraction=rng.uniform(*settings['edit_fraction'][severity])
    )


def prompt_fields(cfg, record, kind):
    """Use the same sampled severity and operation for generation, review, and revision."""
    profile = record['severity_profiles'][kind]
    return dict(
        operation=cfg['perturbation']['operations'][kind],
        modifier=record['modifier'],
        severity=profile['severity'],
        percent=100 * profile['requested_fraction'],
    )


def review_pairs(model, cfg, records, kind):
    """Give Qwen both the clean and perturbed modifier and request PASS/FAIL with a reason."""
    return model.generate(
        [
            prompts.REVIEW.format(
                **prompt_fields(cfg, record, kind), perturbed=record['perturbations'][kind]
            )
            for record in records
        ]
    )


def passed(judgment):
    return judgment.upper().startswith('PASS')


def perturb_modifiers(model, cfg, records, logger):
    """Generate once, judge every pair, revise each rejection once, then judge it again."""
    records = [
        dict(record, perturbations={}, severity_profiles={}, reviews={}) for record in records
    ]
    for kind in cfg['perturbation']['operations']:
        for batch in batches(records, cfg['generation']['text_batch_size']):
            for record in batch:
                record['severity_profiles'][kind] = severity_profile(cfg, record, kind)
            outputs = model.generate(
                [
                    prompts.PERTURBATION.format(**prompt_fields(cfg, record, kind))
                    for record in batch
                ]
            )
            for record, text in zip(batch, outputs):
                record['perturbations'][kind] = text
            judgments = review_pairs(model, cfg, batch, kind)
            rejected = []
            for record, judgment in zip(batch, judgments):
                record['reviews'][kind] = dict(judgment=judgment)
                if not passed(judgment):
                    record['reviews'][kind]['first_attempt'] = dict(
                        modifier=record['perturbations'][kind], judgment=judgment
                    )
                    rejected.append(record)
            if rejected:
                revisions = model.generate(
                    [
                        prompts.PERTURBATION.format(**prompt_fields(cfg, record, kind))
                        + prompts.REPAIR.format(
                            perturbed=record['perturbations'][kind],
                            judgment=record['reviews'][kind]['judgment'],
                        )
                        for record in rejected
                    ]
                )
                for record, text in zip(rejected, revisions):
                    record['perturbations'][kind] = text
                for record, judgment in zip(rejected, review_pairs(model, cfg, rejected, kind)):
                    record['reviews'][kind]['judgment'] = judgment
        counts = Counter(record['severity_profiles'][kind]['severity'] for record in records)
        logger.info('%s severity=%s; review=%s', kind, dict(counts), review_summary(records, kind))
    return records


def review_summary(records, kind):
    """Report final judgments without removing failed pairs from evaluation."""
    regenerated = sum('first_attempt' in record['reviews'][kind] for record in records)
    successes = sum(passed(record['reviews'][kind]['judgment']) for record in records)
    return dict(
        initial_passed=len(records) - regenerated,
        initial_failed=regenerated,
        recovered=sum(
            'first_attempt' in record['reviews'][kind]
            and passed(record['reviews'][kind]['judgment'])
            for record in records
        ),
        passed=successes,
        failed=len(records) - successes,
        success_percent=100 * successes / len(records),
        regenerated=regenerated,
        unchanged=sum(record['modifier'] == record['perturbations'][kind] for record in records),
    )
