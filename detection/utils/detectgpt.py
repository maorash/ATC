""" CODE TAKEN FROM https://github.com/mbzuai-nlp/DetectLLM/blob/main/baselines/detectGPT.py """
import re
from dataclasses import dataclass

import numpy as np
import tqdm
import torch
import transformers

pattern = re.compile(r"<extra_id_\d+>")


def tokenize_and_mask_code(text, pct, ceil_pct=False):
    lines = text.split('\n')
    mask_string = '<<<mask>>>'

    # Calculate number of lines to mask
    n_spans = pct * len(lines)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)
    if n_spans < 1:
        n_spans = 1

    n_masks = 0
    tries = 0
    while n_masks < n_spans:
        line_idx = np.random.randint(0, len(lines))

        if lines[line_idx].strip() and mask_string not in lines[line_idx]:
            lines[line_idx] = re.match(r'(\s*)', lines[line_idx]).group(0) + mask_string
            n_masks += 1
        else:
            tries += 1
            if tries >= 100:
                print("Couldn't find a non-empty line to mask after 100 tries.")
                break

    # Replace each occurrence of mask_string with <extra_id_NUM>
    num_filled = 0
    for idx, line in enumerate(lines):
        if mask_string in line:
            lines[idx] = line.replace(mask_string, f'<extra_id_{num_filled}>')
            num_filled += 1

    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"

    text = '\n'.join(lines)  # Preserve original formatting
    return text


def tokenize_and_mask(text, args, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    tries = 0
    while n_masks < n_spans:
        if len(tokens) - span_length <= 0:
            print("Text too short to mask.")
            break
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            if tries < 100 and len(''.join(tokens[start:end]).strip()) == 0:
                tries += 1
                continue
            if tries >= 100:
                print(f"Couldn't find non-empty mask span after 100 tries.")
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, model_config, args):
    n_expected = count_masks(texts)
    stop_id = model_config['mask_tokenizer'].encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = model_config['mask_tokenizer'](texts, return_tensors="pt", padding=True).to('cuda')
    outputs = model_config['mask_model'].generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=stop_id)  # outputs.shape: torch.Size([20, 57])
    return model_config['mask_tokenizer'].batch_decode(outputs, skip_special_tokens=False)


def apply_extracted_fills(masked_texts, extracted_fills, args):
    # split masked text into tokens, only splitting on spaces (not newlines)
    if args.mask_whole_lines:
        tokens = [x.split('\n') for x in masked_texts]
    else:
        tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                if args.mask_whole_lines:
                    for j in range(len(text)):
                        if f"<extra_id_{fill_idx}>" in text[j]:
                            text[j] = text[j].replace(f"<extra_id_{fill_idx}>", fills[fill_idx])
                            break
                else:
                    found_idx = text.index(f"<extra_id_{fill_idx}>")
                    text[found_idx] = fills[fill_idx]

    # join tokens back into text
    if args.mask_whole_lines:
        texts = ["\n".join(x) for x in tokens]
    else:
        texts = [" ".join(x) for x in tokens]

    return texts


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def perturb_texts_(texts, args, model_config, ceil_pct=False):
    # Must limit text length due to memory errors on very large code segments
    MAX_LEN = 3000
    texts = [x[:MAX_LEN] for x in texts]
    span_length = args.span_length
    pct = args.pct_words_masked
    if args.mask_whole_lines:
        masked_texts = [tokenize_and_mask_code(x, pct, ceil_pct=False) for x in texts]
    else:
        masked_texts = [tokenize_and_mask(x, args, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts, model_config, args)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills, args)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, args, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if
                        idx in idxs]
        raw_fills = replace_masks(masked_texts, model_config, args)
        extracted_fills = extract_fills(raw_fills)
        if attempts > 1:
            # Fill in with empty strings, when texts are too long
            num_masks_to_fill = count_masks(masked_texts)
            extracted_fills = [x + [''] * (n - len(x)) for x, n in zip(extracted_fills, num_masks_to_fill)]
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills, args)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
        if attempts > 2:
            break

    return perturbed_texts


def perturb_texts(texts, mask_tokenizer, mask_model, args, ceil_pct=False):
    model_config = {
        "mask_model": mask_model,
        "mask_tokenizer": mask_tokenizer
    }

    chunk_size = args.chunk_size
    if '11b' in args.mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], args, model_config, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


# TAKEN FROM https://github.com/mbzuai-nlp/DetectLLM/blob/main/baselines/utils/loadmodel.py

def load_mask_filling_model(mask_filling_model_name):
    print(f'Loading mask filling model {mask_filling_model_name}...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, device_map="cuda",
                                                                    torch_dtype=torch.float16)
    # mask_model.parallelize()
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512

    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)

    return mask_tokenizer, mask_model


@dataclass
class DetectGPTArgs:
    mask_filling_model_name: str = 'Salesforce/codet5-large'
    span_length: int = 2
    pct_words_masked: float = 0.3
    buffer_size: int = 1
    mask_top_p: float = 1.0
    chunk_size: int = 20
    n_perturbations: int = 100
    mask_whole_lines: bool = True
