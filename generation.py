"""Lazy, GPU-backed adapters for the original three vision-language models."""
import json
import logging
import re
import random
from collections import Counter

import torch

from utils import get_device
from prompts import TYPO_SYSTEM, WORDING_SYSTEM


def parse_json(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError(f'Model did not return JSON: {text[:200]}')
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def validate_modifier(result, cfg, original=None):
    text = result.get('modifier')
    if not isinstance(text, str):
        raise ValueError('Missing string modifier')
    text = ' '.join(text.split()).strip('"')
    count = len(text.split())
    if not cfg['generation']['min_words'] <= count <= cfg['generation']['max_words']:
        raise ValueError(f'Modifier length {count} outside configured word bounds')
    if original and text.casefold() == original.casefold():
        raise ValueError('Perturbation is unchanged')
    if re.search(r'\b(?:art|tool|ceramic|jewelry)_\d+\b', text):
        raise ValueError('Modifier leaks an image identifier')
    return text


def word_tokens(text):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def validate_perturbation(result, cfg, original, kind):
    text = validate_modifier(result, cfg, original)
    before, after = word_tokens(original), word_tokens(text)
    options = cfg['perturbation']
    if kind == 'typos':
        if len(before) != len(after):
            raise ValueError('Typos must preserve word count and word order')
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        if not 1 <= len(changed) <= options['max_typo_words']:
            raise ValueError(f"Change only 1-{options['max_typo_words']} words; changed {len(changed)}")
        if any(edit_distance(a, b) > options['max_edits_per_word'] for a, b in changed):
            raise ValueError('Use small keyboard edits, not word replacements')
    elif options['preserve_content_words']:
        editable = set(options['editable_words'])
        content_before = Counter(word for word in before if word not in editable)
        content_after = Counter(word for word in after if word not in editable)
        if content_before != content_after:
            missing = list((content_before - content_after).elements())
            added = list((content_after - content_before).elements())
            raise ValueError(f'Preserve all visual words exactly. Missing: {missing}; added: {added}')
    return text


def typo_plan(cfg, original, query_id):
    """Build independent word-level LLM requests and a checked reconstruction."""
    matches = list(re.finditer(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", original))
    editable = set(cfg['perturbation']['editable_words'])
    positions = [i for i, match in enumerate(matches) if match.group().lower() in editable and len(match.group()) >= 3]
    if not positions:
        positions = [i for i, match in enumerate(matches) if len(match.group()) >= 3]
    count = min(cfg['perturbation']['typo_words'], len(positions))
    if not 1 <= count <= cfg['perturbation']['max_typo_words']:
        raise ValueError('typo_words must fit the available words and max_typo_words')
    chosen = random.Random(f"{cfg['seed']}:{query_id}:typos").sample(positions, count)
    tasks, validators = [], []
    for index in chosen:
        word = matches[index].group()
        def validate(result, word=word):
            typo = result.get('typo')
            if not isinstance(typo, str) or len(word_tokens(typo)) != 1 or re.search(r'\s', typo):
                raise ValueError('Return one misspelled word in the typo field')
            distance = edit_distance(word.lower(), typo.lower())
            if not 1 <= distance <= cfg['perturbation']['max_edits_per_word']:
                raise ValueError(f'Return a small spelling error of {word}, not an unchanged/different word')
            return typo
        tasks.append({'system': TYPO_SYSTEM, 'prompt': f'Misspell ONLY this word: {word}\nReturn a small keyboard typo of that exact word, with no other words.'})
        validators.append(validate)
    def finish(typos):
        text = original
        edits = [(matches[i].start(), matches[i].end(), typo) for i, typo in zip(chosen, typos)]
        for start, end, typo in sorted(edits, reverse=True):
            text = text[:start] + typo + text[end:]
        return validate_perturbation({'modifier': text}, cfg, original, 'typos')
    return tasks, validators, finish


def generate_typos(generator, cfg, original, query_id):
    tasks, validators, finish = typo_plan(cfg, original, query_id)
    return finish([generator.request_json(**task, validator=validator,
        attempts=cfg['perturbation']['max_attempts'], decoding=cfg['perturbation']['decoding'])
        for task, validator in zip(tasks, validators)])


def word_preserving_request(cfg, original, kind, instruction):
    """Let the LLM rephrase syntax while protected visual spans remain exact."""
    editable = set(cfg['perturbation']['editable_words'])
    # Keep adjacent visual words together so adjectives cannot become separate list items.
    spans = []
    for match in re.finditer(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", original):
        if match.group().lower() in editable or match.group().lower() in {'or', 'either', 'not', 'no', 'without'}:
            continue
        if spans and original[spans[-1][1]:match.start()].isspace():
            spans[-1] = (spans[-1][0], match.end())
        else:
            spans.append((match.start(), match.end()))
    protected = {f'CONTENT{i}': original[start:end] for i, (start, end) in enumerate(spans)}
    skeleton = original
    for i in range(len(spans) - 1, -1, -1):
        start, end = spans[i]
        skeleton = skeleton[:start] + f'CONTENT{i}' + skeleton[end:]
    # Preserve complete before/after arguments, including internal conjunctions.
    # Otherwise a model may pair individual source nouns with individual target nouns.
    primary = re.match(r'(?is)^(replace|change|swap|exchange|substitute|turn|transform)\s+(.+?)\s+(with|to|for|into)\s+(.+?)([.!?]?)$', original)
    if primary:
        verb, reference, connector, target, punctuation = primary.groups()
        protected = {}
        arguments = []
        for index, phrase in enumerate([reference, target]):
            article = re.match(r'(?is)^(the|a|an)\s+(.+)$', phrase)
            prefix, content = (article[1] + ' ', article[2]) if article else ('', phrase)
            protected[f'CONTENT{index}'] = content
            arguments.append(prefix + f'CONTENT{index}')
        skeleton = f'{verb} {arguments[0]} {connector} {arguments[1]}{punctuation}'
    def validate(result):
        text = result.get('modifier')
        if not isinstance(text, str):
            raise ValueError('Missing string modifier')
        tokens = re.findall(r'CONTENT\d+', text, flags=re.I)
        if [t.upper() for t in tokens] != list(protected):
            raise ValueError(f'Expected tokens in order: {list(protected)}; received {tokens}. Keep all exactly once.')
        text = re.sub(r'CONTENT\d+', lambda m: protected[m.group().upper()], text, flags=re.I)
        return validate_perturbation({'modifier': text}, cfg, original, kind)
    task = {'system': WORDING_SYSTEM, 'prompt':
        f'Style: {instruction}\nRequest: {skeleton}\nAllowed function words: '
        + ', '.join(sorted(editable)) + '\nRequired protected tokens, in order: ' + ', '.join(protected)
        + '\nKeep literal or/either/not/no/without wherever present; they are logical constraints.'}
    return task, validate


def generate_word_preserving(generator, cfg, original, kind, instruction):
    task, validator = word_preserving_request(cfg, original, kind, instruction)
    return generator.request_json(**task, attempts=cfg['perturbation']['max_attempts'],
                                  decoding=cfg['perturbation']['decoding'], validator=validator)


def safe_rewrite_candidates(cfg, original, kind):
    """Conservative surface forms for constrained LLM selection after free-form retries."""
    candidates = []
    words = original.split(maxsplit=1)
    if len(words) == 2 and words[0].lower() in {'replace', 'change', 'swap', 'substitute', 'exchange', 'transform', 'turn', 'alter'}:
        verbs = ['Swap', 'Change', 'Replace'] if kind == 'informal' else ['Substitute', 'Exchange', 'Transform', 'Replace']
        candidates.extend(verb + ' ' + words[1] for verb in verbs)
    candidates.extend(['Please ' + original, 'Could you ' + original, original.rstrip('.!?') + ', please.'])
    # Determiner changes also fit modifiers already at the configured maximum length.
    for match in re.finditer(r'\b(?:the|a|an)\b', original, flags=re.I):
        candidates.append(original[:match.start()] + 'this' + original[match.end():])
    valid = []
    for candidate in dict.fromkeys(candidates):
        try:
            valid.append(validate_perturbation({'modifier': candidate}, cfg, original, kind))
        except ValueError:
            continue
    if not valid:
        raise ValueError('No valid constrained wording candidates for this modifier')
    return valid


def whole_sentence_request(cfg, original, kind):
    def validate(result):
        text = result.get('modifier')
        if not isinstance(text, str) or text.count('CONTENT0') != 1:
            raise ValueError('Keep CONTENT0 exactly once')
        return validate_perturbation({'modifier': text.replace('CONTENT0', original)}, cfg, original, kind)
    return {'system': WORDING_SYSTEM,
            'prompt': 'Preserve CONTENT0 as one complete request. Add a short polite or casual introduction. '
                      'Use only Please/Can/Could/you/I/want/to around CONTENT0. '
                      'Examples of valid structure: Please CONTENT0; Could you CONTENT0. '
                      'Return JSON with modifier; the result must differ from CONTENT0 alone.'}, validate


class TextGenerator:
    def __init__(self, cfg, logger=None):
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.device = get_device(cfg)
        self.name = cfg['generation']['model']
        model_id = cfg['generation']['models'][self.name]
        common = dict(cache_dir=cfg['paths']['model_cache'],
                      local_files_only=cfg['runtime']['local_files_only'])
        dtype = getattr(torch, cfg['runtime']['dtype']) if self.device.type == 'cuda' else torch.float32
        self.logger.info('Loading generation model=%s device=%s dtype=%s', model_id, self.device, dtype)
        if self.name == 'qwen':
            from transformers import Qwen2_5_VLForConditionalGeneration
            cls = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_id, **common, **cfg['generation']['vision'])
        elif self.name == 'llava':
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            cls = LlavaNextForConditionalGeneration
            self.processor = LlavaNextProcessor.from_pretrained(model_id, **common)
        elif self.name == 'deepseek':
            if not cfg['runtime']['trust_remote_code']:
                raise ValueError('DeepSeek requires runtime.trust_remote_code: true and deepseek_vl installed')
            from deepseek_vl.models import VLChatProcessor
            cls = AutoModelForCausalLM
            self.processor = VLChatProcessor.from_pretrained(model_id, **common)
            common['trust_remote_code'] = True
        else:
            raise ValueError(f'Unknown generation model: {self.name}')
        self.model = cls.from_pretrained(model_id, torch_dtype=dtype, **common).to(self.device).eval()
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'

    @torch.inference_mode()
    def generate(self, system, prompt, images=(), decoding=None, image_labels=None):
        # PIL inputs and explicit placeholders avoid networking and preserve multi-image order.
        content = []
        for index in range(len(images)):
            label = image_labels[index] if image_labels else ('BEFORE: reference image' if index == 0 else f'AFTER: relevant target {index}')
            content.extend([{'type': 'text', 'text': label}, {'type': 'image'}])
        if self.name == 'qwen':
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
        elif self.name == 'llava':
            content.append({'type': 'text', 'text': system + '\n' + prompt})
            messages = [{'role': 'user', 'content': content}]
        else:
            messages = [
                {'role': 'User', 'content': system + '\n' + '<image_placeholder>\n' * len(images) + prompt,
                 'images': []}, {'role': 'Assistant', 'content': ''}]
        decode = decoding or self.cfg['generation']['decoding']
        options = {'max_new_tokens': decode['max_new_tokens'], 'do_sample': decode['do_sample']}
        if decode['do_sample']:
            options.update({k: decode[k] for k in ['temperature', 'top_p', 'top_k']})
        if self.name == 'deepseek':
            inputs = self.processor(conversations=messages, images=list(images), force_batchify=True).to(self.device)
            embeds = self.model.prepare_inputs_embeds(**inputs)
            tokenizer = self.processor.tokenizer
            ids = self.model.language_model.generate(inputs_embeds=embeds, attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id, **options)
            return tokenizer.decode(ids[0], skip_special_tokens=True)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=list(images) or None, return_tensors='pt', padding=True).to(self.device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
        ids = self.model.generate(**inputs, **options)
        return self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def generate_many(self, tasks, decoding=None):
        if self.name != 'qwen':
            return [self.generate(**task, decoding=decoding) for task in tasks]
        messages, images = [], []
        for task in tasks:
            content = []
            task_images = task.get('images', ())
            labels = task.get('image_labels') or ['Image'] * len(task_images)
            for label, image in zip(labels, task_images):
                content.extend([{'type': 'text', 'text': label}, {'type': 'image'}])
                images.append(image)
            content.append({'type': 'text', 'text': task['prompt']})
            messages.append([{'role': 'system', 'content': task['system']}, {'role': 'user', 'content': content}])
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        inputs = self.processor(text=texts, images=images or None, return_tensors='pt', padding=True).to(self.device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
        decode = decoding or self.cfg['generation']['decoding']
        options = {'max_new_tokens': decode['max_new_tokens'], 'do_sample': decode['do_sample']}
        if decode['do_sample']:
            options.update({key: decode[key] for key in ['temperature', 'top_p', 'top_k']})
        input_length = inputs['input_ids'].shape[1]
        try:
            ids = self.model.generate(**inputs, **options)
        except torch.cuda.OutOfMemoryError:
            if len(tasks) == 1:
                raise
            del inputs
            torch.cuda.empty_cache()
            middle = len(tasks) // 2
            self.logger.warning('GPU batch too large; retrying batches of %d and %d', middle, len(tasks) - middle)
            return self.generate_many(tasks[:middle], decoding) + self.generate_many(tasks[middle:], decoding)
        return self.processor.batch_decode(ids[:, input_length:], skip_special_tokens=True)

    @torch.inference_mode()
    def choose_valid_rewrite(self, original, kind):
        """The LLM selects a valid surface rewrite with token-constrained decoding."""
        candidates = safe_rewrite_candidates(self.cfg, original, kind)
        task = [{'role': 'system', 'content': 'Choose a natural wording variant for the search request. Return JSON with modifier.'},
                {'role': 'user', 'content': f'Style: {kind}\nOriginal: {original}\nAllowed variants: ' + json.dumps(candidates)}]
        if self.name == 'deepseek':
            raise ValueError('Constrained wording selection is currently supported by Qwen/LLaVA')
        text = self.processor.apply_chat_template(task, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors='pt').to(self.device)
        tokenizer = self.processor.tokenizer
        choices = [tokenizer.encode(json.dumps({'modifier': candidate}), add_special_tokens=False) for candidate in candidates]
        length = inputs['input_ids'].shape[1]
        eos = tokenizer.eos_token_id
        def allowed_tokens(batch_id, token_ids):
            prefix = token_ids[length:].tolist()
            allowed = set()
            for choice in choices:
                if choice[:len(prefix)] == prefix:
                    allowed.add(choice[len(prefix)] if len(prefix) < len(choice) else eos)
            return sorted(allowed) or [eos]
        ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=max(map(len, choices)) + 1,
                                  prefix_allowed_tokens_fn=allowed_tokens, eos_token_id=eos)
        raw = tokenizer.decode(ids[0, length:], skip_special_tokens=True)
        result = validate_perturbation(parse_json(raw), self.cfg, original, kind)
        self.logger.info('Constrained LLM wording selection: %s', result)
        return result

    def request_json_many(self, tasks, validators, attempts=1, decoding=None):
        """Retry invalid items only; return exceptions so callers checkpoint successes."""
        results = [None] * len(tasks)
        pending = list(range(len(tasks)))
        requests = [dict(task) for task in tasks]
        for attempt in range(attempts):
            if not pending:
                break
            raw_outputs = self.generate_many([requests[i] for i in pending], decoding=decoding)
            if len(raw_outputs) != len(pending):
                raise ValueError('Model batch output count differs from request count')
            retry = []
            for index, raw in zip(pending, raw_outputs):
                self.logger.info('LLM batch item=%d attempt=%d response=%s', index, attempt + 1, raw)
                try:
                    parsed = parse_json(raw)
                    results[index] = validators[index](parsed) if validators[index] else parsed
                except (ValueError, TypeError, KeyError) as error:
                    results[index] = error
                    retry.append(index)
                    self.logger.warning('Rejected batch item=%d: %s', index, error)
                    requests[index]['prompt'] = tasks[index]['prompt'] + '\nPrevious rejected response: ' + raw + '\nFix: ' + str(error)
            pending = retry
        return results

    def request_json(self, system, prompt, images=(), attempts=1, validator=None, decoding=None, image_labels=None):
        for attempt in range(attempts):
            raw = self.generate(system, prompt, images, decoding=decoding, image_labels=image_labels)
            self.logger.info('LLM response attempt=%d: %s', attempt + 1, raw)
            try:
                result = parse_json(raw)
                if validator:
                    result = validator(result)
                return result
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.logger.warning('Rejected response: %s', error)
                if attempt + 1 == attempts:
                    raise
                prompt += '\nPrevious rejected response: ' + raw + '\nWrite a DIFFERENT valid response. Fix: ' + str(error)


def load_text_gen_model(cfg, logger):
    return TextGenerator(cfg, logger)
