"""Current multimodal model adapters and validated batched JSON generation."""

import json
import logging
import re
from functools import partial

import torch

from utils import batches, get_device, select_option, word_count

GENERATION_POLICY = 'sampled_concrete_requests_v2'


def parse_json(text):
    """Extract a JSON object after optional model reasoning and surrounding prose."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('Expected a JSON object')
    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def validate_modifier(result, cfg):
    """Enforce request length, shared-target wording, and corruption applicability."""
    text = result.get('modifier')
    if not isinstance(text, str):
        raise ValueError('modifier must be a string')
    text = ' '.join(text.split())
    if not cfg['generation']['min_words'] <= word_count(text) <= cfg['generation']['max_words']:
        raise ValueError(
            f'Write {cfg["generation"]["min_words"]}-{cfg["generation"]["max_words"]} words; received {word_count(text)}'
        )
    if re.search(r'\b\w+_\d+\b', text):
        raise ValueError('Do not include image IDs')
    if re.search(r'\b(?:BEFORE|COMMON FEATURES|SINGLE TARGET)\b', text):
        raise ValueError('Use natural object words or it, not prompt labels such as BEFORE')
    if re.search(r'\bor\b', text, re.I):
        raise ValueError(
            'Describe features shared by every target; do not enumerate alternatives using or'
        )
    # Every configured corruption must be possible; never silently omit a condition.
    from perturbation import applicable

    if any(not applicable(text, kind) for kind in cfg['perturbation']['types']):
        raise ValueError(
            'Use a natural request containing a descriptive word of at least four letters'
        )
    return text


def validate_description(result):
    """Read one nonempty image observation from the model's JSON response."""
    text = result.get('description')
    if not isinstance(text, str) or not text.strip() or word_count(text) > 80:
        raise ValueError('description must be a nonempty sentence of at most 80 words')
    return text.strip()


def validate_shared_features(result, cfg):
    """Accept a short common-feature description without alternative categories."""
    text = result.get('shared_features')
    if (
        not isinstance(text, str)
        or not text.strip()
        or word_count(text) > cfg['generation']['shared_features_max_words']
    ):
        raise ValueError('shared_features must be a short nonempty phrase')
    if re.search(r'\bor\b', text, re.I):
        raise ValueError(
            'Keep the most specific shared object name and supported details. Drop only differing attributes; no or alternatives.'
        )
    return text.strip()


def validate_target_review(result):
    """Require an explicit per-target decision and an explanation for rejection."""
    coverage = result.get('coverage')
    if coverage not in {'Y', 'N'}:
        raise ValueError('coverage must be Y or N for the single target under review')
    if coverage == 'N' and not result.get('reason'):
        raise ValueError('Explain which desired-result requirement is unsupported')
    return result


def choose_modifier(result, options, validator, error='Choose one listed integer index'):
    """Validate a selected category or shortened request using its original validator."""
    return validator({'modifier': select_option(result, options, error)})


def target_requirements(text):
    """Remove the source of an explicit replacement from target-only auditing."""
    match = re.match(
        r'(?is)^(?:replace|change|swap|exchange|substitute|turn|transform)\s+.+?\s+(?:with|to|for|into)\s+(.+)$',
        text,
    )
    return match.group(1) if match else text


def generate_grounded_modifiers(
    generator, cfg, before, after, shared, full_counts=None, target_paths=None
):
    """Generate, review every sampled target, and broaden rejected requirements."""
    import prompts

    if (
        not before
        or len(before) != len(after)
        or len(before) != len(shared)
        or any(not targets for targets in after)
    ):
        raise ValueError(
            'Need aligned nonempty reference, target-group, and shared-feature observations'
        )
    after = [list(targets) for targets in after]
    full_counts = full_counts or [len(targets) for targets in after]
    if len(full_counts) != len(before) or any(
        count < len(targets) for count, targets in zip(full_counts, after)
    ):
        raise ValueError('Full target counts must align with and include the sampled targets')
    if target_paths is not None and (
        len(target_paths) != len(after)
        or any(len(paths) != len(targets) for paths, targets in zip(target_paths, after))
    ):
        raise ValueError('Target image paths must align with all sampled observations')
    tasks = [
        {
            'field': 'modifier',
            'system': prompts.MODIFIER,
            'prompt': f'BEFORE: {source}\nCOMMON FEATURES OF ALL {len(targets)} TARGETS: {common}'
            + '\nINDIVIDUAL TARGET IMAGES: '
            + json.dumps(targets)
            + f'\nThese observations cover {len(targets)} of {count} available targets. '
            + (
                'Keep the most specific shared object name and stable visible traits. Avoid exact counts and incidental details that may vary across unseen targets; do not discard useful shared information.'
                if count > len(targets)
                else ''
            )
            + f'\nUse {cfg["generation"]["min_words"]}-{cfg["generation"]["max_words"]} words.',
        }
        for source, targets, common, count in zip(before, after, shared, full_counts)
    ]
    for i, common in enumerate(shared):
        if common is None:
            tasks[i]['prompt'] = (
                f'REFERENCE IMAGE: {before[i]}\nINDIVIDUAL TARGET IMAGES: '
                + json.dumps(after[i])
                + f'\nWrite {cfg["generation"]["min_words"]}-{cfg["generation"]["max_words"]} words using only features shared by all these targets.'
            )
    results = [ValueError('All-target review not passed')] * len(tasks)
    pending = list(range(len(tasks)))
    feedback = [''] * len(tasks)
    rejected_categories = [set() for _ in tasks]
    category_start = cfg['generation']['target_review_attempts'] - 1
    attempts = category_start + len(cfg['generation']['fallback_modifiers'])
    modifier_validator = partial(validate_modifier, cfg=cfg)
    for attempt in range(attempts):
        if not pending:
            break
        requests = [dict(tasks[i], prompt=tasks[i]['prompt'] + feedback[i]) for i in pending]
        validators = [modifier_validator] * len(pending)
        if attempt >= category_start:
            requests, validators = [], []
            for i in pending:
                options = [
                    text
                    for text in cfg['generation']['fallback_modifiers']
                    if text not in rejected_categories[i]
                ]
                requests.append(
                    {
                        'field': 'choice',
                        'system': prompts.CATEGORY_FALLBACK,
                        'prompt': 'ALL TARGET DESCRIPTIONS: '
                        + json.dumps(after[i])
                        + '\nOptions:\n'
                        + '\n'.join(f'{j}: {value}' for j, value in enumerate(options)),
                    }
                )

                validators.append(
                    partial(choose_modifier, options=options, validator=modifier_validator)
                )
        outputs = generator.request_json_many(requests, validators)
        proposed = {i: text for i, text in zip(pending, outputs) if not isinstance(text, Exception)}
        for i, text in zip(pending, outputs):
            if isinstance(text, Exception):
                results[i] = text
                tasks[i]['system'] = prompts.BROAD_MODIFIER
                tasks[i]['prompt'] = (
                    f'REFERENCE IMAGE: {before[i]}\nINDIVIDUAL TARGET IMAGES: '
                    + json.dumps(after[i])
                )
                feedback[i] = (
                    '\nPrevious validation error: '
                    + str(text)
                    + '\nFix the format while keeping the most specific shared object name and supported details. No alternatives.'
                )
        reviews, owners = [], []
        for i, text in proposed.items():
            for j, observation in enumerate(after[i]):
                task = {
                    'field': 'coverage',
                    'system': prompts.TARGET_REVIEW,
                    'prompt': f'DESIRED RESULT REQUIREMENTS: {target_requirements(text)}\nSINGLE TARGET: {observation}',
                }
                if target_paths is not None:
                    from loading_data import load_image

                    task.update(
                        system=prompts.VISUAL_TARGET_REVIEW,
                        prompt=f'DESIRED RESULT REQUIREMENTS: {target_requirements(text)}\nJudge only the attached target image.',
                        images=[load_image(target_paths[i][j])],
                    )
                reviews.append(task)
                owners.append((i, j))

        if reviews:
            outcomes = []
            for jobs, validators in zip(
                batches(reviews, cfg['generation']['batch_size']),
                batches([validate_target_review] * len(reviews), cfg['generation']['batch_size']),
            ):
                outcomes.extend(generator.request_json_many(jobs, validators))
            if len(outcomes) != len(owners):
                raise ValueError('Every supplied target must receive an individual review')
            rejected = {}
            for (i, j), review in zip(owners, outcomes):
                if isinstance(review, Exception) or review['coverage'] != 'Y':
                    rejected.setdefault(i, []).append((f'target {j + 1}', str(review)))
                    if isinstance(review, dict) and review.get('visible_features'):
                        after[i][j] = review['visible_features']
            for i in proposed:
                if i not in rejected:
                    results[i] = proposed[i]
                else:
                    if attempt >= category_start:
                        rejected_categories[i].add(proposed[i])
                    logging.getLogger(__name__).warning(
                        'Modifier target review: candidate=%r failures=%s', proposed[i], rejected[i]
                    )
                    results[i] = ValueError(
                        f'Sampled-target compatibility review rejected {proposed[i]!r}: {rejected[i]}'
                    )
                    # Discard the possibly incorrect shared summary on retry. Direct
                    # observations let the model recover without copying its mistake.
                    tasks[i]['prompt'] = 'INDIVIDUAL TARGET IMAGES: ' + json.dumps(after[i])
                    feedback[i] = (
                        '\nFailed target reviews: the previous request does not fit every target. '
                        'Start again from ALL the corrected target descriptions. Choose their '
                        'most specific common category. Omit details that vary; do not negate them.'
                    )
                    if attempt >= 1:
                        tasks[i]['system'] = prompts.BROAD_MODIFIER
        pending = [i for i in pending if isinstance(results[i], Exception)]
    return results


def concise_candidates(text, maximum):
    """Complete reference-to-target clauses; never cut a sentence at an arbitrary word."""
    match = re.match(
        r'(?is)^(replace|change|swap|exchange|substitute|turn|transform)\s+(.+?)\s+(with|to|for|into)\s+(.+?)[.!?]?$',
        text,
    )
    if not match:
        return []
    verb, source, connector, target = match.groups()
    target = target.rstrip('.!?')
    first_comma = target.find(',')
    ends = [len(target)] + [
        m.start()
        for m in re.finditer(
            r',|\s+(?:with|wearing|standing|holding|on|in|against|mounted|featuring|serving)\s+',
            target,
            flags=re.I,
        )
        if first_comma < 0 or m.start() <= first_comma
    ]

    options = []
    for end in ends:
        phrase = target[:end].strip(' ,')
        if word_count(phrase) < 2 or logical_words(phrase) != logical_words(target):
            continue
        for before in [source, 'it']:
            candidate = f'{verb} {before} {connector} {phrase}.'
            if word_count(candidate) <= maximum:
                options.append(candidate)
    return sorted(dict.fromkeys(options), key=word_count, reverse=True)


def logical_words(text):
    """Keep negation and alternatives intact when shortening a request."""
    return re.findall(r'\b(?:or|not|no|without)\b', text.lower())


class TextGenerator:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.name = cfg['generation']['model']
        self.device = get_device(cfg)
        spec = cfg['generation']['models'][self.name]
        common = dict(
            cache_dir=cfg['paths']['model_cache'],
            revision=spec['revision'],
            local_files_only=cfg['runtime']['local_files_only'],
            trust_remote_code=spec['trust_remote_code'],
        )
        self.logger.info('Loading %s revision=%s', spec['id'], spec['revision'])
        dtype = (
            getattr(torch, cfg['runtime']['dtype']) if self.device.type == 'cuda' else torch.float32
        )
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            JanusForConditionalGeneration,
        )

        self.processor = AutoProcessor.from_pretrained(spec['id'], **common)
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        if hasattr(self.processor, 'image_processor') and self.name in {'qwen', 'llava'}:
            self.processor.image_processor.max_pixels = cfg['generation']['max_pixels']
            if hasattr(self.processor.image_processor, 'size'):
                self.processor.image_processor.size['longest_edge'] = cfg['generation'][
                    'max_pixels'
                ]
        kwargs = dict(dtype=dtype, attn_implementation='sdpa')
        if self.name == 'qwen' and self.device.type == 'cuda':
            kwargs.update(
                device_map='auto',
                max_memory={
                    i: f'{cfg["generation"]["max_memory_gib"]}GiB'
                    for i in cfg['generation']['gpu_ids']
                },
            )
        else:
            kwargs['device_map'] = str(self.device)
        cls = (
            JanusForConditionalGeneration
            if self.name == 'deepseek'
            else AutoModelForImageTextToText
        )
        self.model = cls.from_pretrained(spec['id'], **common, **kwargs).eval()
        self.device = self.model.get_input_embeddings().weight.device
        if (
            any(
                str(value) in {'cpu', 'disk'}
                for value in getattr(self.model, 'hf_device_map', {}).values()
            )
            and cfg['runtime']['require_cuda']
        ):
            raise RuntimeError('Model does not fit configured GPUs; refusing CPU/disk offload')

    @torch.inference_mode()
    def generate_many(self, tasks):
        """Run greedy model inference, splitting batches if GPU memory is insufficient."""
        if self.name != 'qwen' and len(tasks) > 1:
            return [self.generate_many([task])[0] for task in tasks]
        messages, images = [], []
        for task in tasks:
            content = [{'type': 'text', 'text': task['system']}]
            for image in task.get('images', ()):
                content.append({'type': 'image'})
                images.append(image)
            content.append({'type': 'text', 'text': task['prompt']})
            messages.append([{'role': 'user', 'content': content}])
        prefixes = [
            '{"'
            + task.get('field', 'modifier')
            + '": '
            + ('' if task.get('field') == 'choice' else '"')
            for task in tasks
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            + prefix
            for m, prefix in zip(messages, prefixes)
        ]
        inputs = self.processor(
            text=texts, images=images or None, return_tensors='pt', padding=True
        ).to(self.device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
        options = dict(max_new_tokens=self.cfg['generation']['max_new_tokens'], do_sample=False)
        if self.name == 'deepseek':
            options['generation_mode'] = 'text'
        input_length = inputs['input_ids'].shape[1]
        try:
            ids = self.model.generate(**inputs, **options)
        except torch.cuda.OutOfMemoryError:
            if len(tasks) == 1:
                raise
            del inputs
            torch.cuda.empty_cache()
            middle = len(tasks) // 2
            self.logger.warning('Halving generation batch after CUDA OOM')
            return self.generate_many(tasks[:middle]) + self.generate_many(tasks[middle:])
        outputs = self.processor.batch_decode(ids[:, input_length:], skip_special_tokens=True)
        return [prefix + text for prefix, text in zip(prefixes, outputs)]

    def request_json_many(self, tasks, validators):
        """Retry invalid JSON items while retaining successes and returning final errors."""
        results = [None] * len(tasks)
        shortening = {}
        pending = list(range(len(tasks)))
        requests = [dict(task) for task in tasks]
        for attempt in range(self.cfg['generation']['max_attempts']):
            if not pending:
                break
            raw_outputs = self.generate_many([requests[i] for i in pending])
            if len(raw_outputs) != len(pending):
                raise ValueError('Model output count differs from batch size')
            retry = []
            for i, raw in zip(pending, raw_outputs):
                parsed = None
                try:
                    parsed = parse_json(raw)
                    results[i] = validators[i](parsed)
                except (ValueError, TypeError, KeyError) as error:
                    results[i] = error
                    self.logger.warning(
                        'Retry item=%d attempt=%d reason=%s response=%r',
                        i,
                        attempt + 1,
                        error,
                        raw[:400],
                    )
                    if (
                        tasks[i].get('field') == 'modifier'
                        and isinstance(parsed, dict)
                        and isinstance(parsed.get('modifier'), str)
                        and word_count(parsed['modifier']) > self.cfg['generation']['max_words']
                    ):
                        options = []
                        for candidate in concise_candidates(
                            parsed['modifier'], self.cfg['generation']['max_words']
                        ):
                            try:
                                validators[i]({'modifier': candidate})
                                options.append(candidate)
                            except (ValueError, TypeError, KeyError):
                                continue
                        if options:
                            shortening[i] = options
                            continue
                    retry.append(i)
                    requests[i]['prompt'] = (
                        tasks[i]['prompt']
                        + '\nThe previous response failed validation. Fix: '
                        + str(error)
                    )
            pending = retry
        if shortening:
            jobs, checks, indices = [], [], list(shortening)
            for i in indices:
                options = shortening[i]
                jobs.append(
                    {
                        'field': 'choice',
                        'system': 'Choose the most natural concise image-search request that retains the main target and direction. Return only JSON with an integer choice.',
                        'prompt': 'Select a complete request, preferably retaining useful target attributes:\n'
                        + '\n'.join(f'{j}: {value}' for j, value in enumerate(options)),
                    }
                )

                checks.append(
                    partial(
                        choose_modifier,
                        options=options,
                        validator=validators[i],
                        error='Choose a listed integer index',
                    )
                )
            for i, result in zip(indices, self.request_json_many(jobs, checks)):
                results[i] = result
        return results
