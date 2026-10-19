"""Current multimodal model adapters and validated batched JSON generation."""
import json
import logging
import re

import torch

from utils import get_device


def parse_json(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('Expected a JSON object')
    result = json.loads(text[start:end+1])
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def validate_modifier(result, cfg):
    text = result.get('modifier')
    if not isinstance(text, str):
        raise ValueError('modifier must be a string')
    text = ' '.join(text.split())
    if not cfg['generation']['min_words'] <= len(text.split()) <= cfg['generation']['max_words']:
        raise ValueError(f"Write {cfg['generation']['min_words']}-{cfg['generation']['max_words']} words; received {len(text.split())}")
    if re.search(r'\b\w+_\d+\b', text):
        raise ValueError('Do not include image IDs')
    # Every configured corruption must be possible; never silently omit a condition.
    from perturbation import applicable
    if any(not applicable(text, kind) for kind in cfg['perturbation']['types']):
        raise ValueError('Use a natural request containing a descriptive word of at least four letters')
    return text


def concise_candidates(text, maximum):
    """Complete reference-to-target clauses; never cut a sentence at an arbitrary word."""
    match = re.match(r'(?is)^(replace|change|swap|exchange|substitute|turn|transform)\s+(.+?)\s+(with|to|for|into)\s+(.+?)[.!?]?$', text)
    if not match:
        return []
    verb, source, connector, target = match.groups()
    target = target.rstrip('.!?')
    first_comma = target.find(',')
    ends = [len(target)] + [m.start() for m in re.finditer(r',|\s+(?:with|wearing|standing|holding|on|in|against|mounted|featuring|serving)\s+', target, flags=re.I) if first_comma < 0 or m.start() <= first_comma]
    def logical(phrase):
        return re.findall(r'\b(?:or|not|no|without)\b', phrase.lower())
    options = []
    for end in ends:
        phrase = target[:end].strip(' ,')
        if len(phrase.split()) < 2 or logical(phrase) != logical(target):
            continue
        for before in [source, 'it']:
            candidate = f'{verb} {before} {connector} {phrase}.'
            if len(candidate.split()) <= maximum:
                options.append(candidate)
    return sorted(dict.fromkeys(options), key=lambda value: -len(value.split()))


class TextGenerator:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.name = cfg['generation']['model']
        self.device = get_device(cfg)
        spec = cfg['generation']['models'][self.name]
        common = dict(cache_dir=cfg['paths']['model_cache'], revision=spec['revision'],
                      local_files_only=cfg['runtime']['local_files_only'],
                      trust_remote_code=spec['trust_remote_code'])
        self.logger.info('Loading %s revision=%s', spec['id'], spec['revision'])
        dtype = getattr(torch, cfg['runtime']['dtype']) if self.device.type == 'cuda' else torch.float32
        from transformers import AutoModelForImageTextToText, AutoProcessor, JanusForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(spec['id'], **common)
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        if hasattr(self.processor, 'image_processor') and self.name in {'qwen', 'llava'}:
            self.processor.image_processor.max_pixels = cfg['generation']['max_pixels']
            if hasattr(self.processor.image_processor, 'size'):
                self.processor.image_processor.size['longest_edge'] = cfg['generation']['max_pixels']
        kwargs = dict(dtype=dtype, attn_implementation='sdpa')
        if self.name == 'qwen' and self.device.type == 'cuda':
            kwargs.update(device_map='auto', max_memory={i: f"{cfg['generation']['max_memory_gib']}GiB"
                          for i in cfg['generation']['gpu_ids']})
        else:
            kwargs['device_map'] = str(self.device)
        cls = JanusForConditionalGeneration if self.name == 'deepseek' else AutoModelForImageTextToText
        self.model = cls.from_pretrained(spec['id'], **common, **kwargs).eval()
        self.device = self.model.get_input_embeddings().weight.device
        if any(str(value) in {'cpu', 'disk'} for value in getattr(self.model, 'hf_device_map', {}).values()) and cfg['runtime']['require_cuda']:
            raise RuntimeError('Model does not fit configured GPUs; refusing CPU/disk offload')

    @torch.inference_mode()
    def generate_many(self, tasks):
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
        prefixes = ['{\"' + task.get('field', 'modifier') + '\": ' + ('' if task.get('field') == 'choice' else '\"') for task in tasks]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) + prefix for m, prefix in zip(messages, prefixes)]
        inputs = self.processor(text=texts, images=images or None, return_tensors='pt', padding=True).to(self.device)
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
                    self.logger.warning('Retry item=%d attempt=%d reason=%s response=%r', i, attempt+1, error, raw[:400])
                    if tasks[i].get('field') == 'modifier' and isinstance(parsed, dict) and isinstance(parsed.get('modifier'), str) and len(parsed['modifier'].split()) > self.cfg['generation']['max_words']:
                        options = []
                        for candidate in concise_candidates(parsed['modifier'], self.cfg['generation']['max_words']):
                            try:
                                validators[i]({'modifier': candidate})
                                options.append(candidate)
                            except (ValueError, TypeError, KeyError):
                                continue
                        if options:
                            shortening[i] = options
                            continue
                    retry.append(i)
                    requests[i]['prompt'] = tasks[i]['prompt'] + '\nRejected response: ' + raw + '\nFix: ' + str(error)
            pending = retry
        if shortening:
            jobs, checks, indices = [], [], list(shortening)
            for i in indices:
                options = shortening[i]
                jobs.append({'field': 'choice', 'system': 'Choose the most natural concise image-search request that retains the main target and direction. Return only JSON with an integer choice.',
                    'prompt': 'Select a complete request, preferably retaining useful target attributes:\n' + '\n'.join(f'{j}: {value}' for j, value in enumerate(options))})
                def choose(result, options=options, validator=validators[i]):
                    j = result.get('choice')
                    if type(j) is not int or not 0 <= j < len(options):
                        raise ValueError('Choose a listed integer index')
                    return validator({'modifier': options[j]})
                checks.append(choose)
            for i, result in zip(indices, self.request_json_many(jobs, checks)):
                results[i] = result
        return results
