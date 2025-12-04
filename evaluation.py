import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./gpt2-finetuned-binomial"

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()


# Latinized Epithet Constraint
LATIN_EPITHET_REGEX = re.compile(r"^[a-z]+(us|a|um|is|ensis|ii)?$")

def latin_epithet_allowed_tokens_fn(batch_id, input_ids_so_far):
    decoded = tokenizer.decode(input_ids_so_far, skip_special_tokens=True)
    if "Name:" in decoded:
        name_part = decoded.split("Name:")[-1].strip()
    else:
        name_part = ""
    words = name_part.split()
    vocab = list(range(len(tokenizer)))
    if len(words) >= 2:
        return [tokenizer.eos_token_id]
    if len(words) == 0:
        return vocab
    allowed_ids = []
    for token_id in vocab:
        candidate = tokenizer.decode([token_id]).strip().lower()
        if LATIN_EPITHET_REGEX.match(candidate):
            allowed_ids.append(token_id)
    return allowed_ids if allowed_ids else vocab  


# Test
example_prompts = [
    "Description: a large brown bear with a scar on its paw\nFamily: Ursidae\nName:",
    "Description: a tiny gray mouse living in a barn\nFamily: Muridae\nName:",
    "Description: a colorful parrot that can imitate human speech\nFamily: Psittacidae\nName:",
    "Description: a dark green frog that lives near waterfalls\nFamily: Ranidae\nName:",
    "Description: a fast-running desert fox\nFamily: Canidae\nName:",
    "Description: a golden-scaled fish often seen in garden ponds\nFamily: Cyprinidae\nName:",
    "Description: a fluffy black rabbit with long ears\nFamily: Leporidae\nName:",
    "Description: a snow owl known for silent flight\nFamily: Strigidae\nName:",
    "Description: a gentle giant elephant with long tusks\nFamily: Elephantidae\nName:",
    "Description: a red-striped tiger wandering in bamboo forests\nFamily: Felidae\nName:",
    "Description: a shy hedgehog that curls into a ball\nFamily: Erinaceidae\nName:",
    "Description: a sleek black panther that hunts at night\nFamily: Felidae\nName:",
    "Description: a curious dolphin that plays with seaweed\nFamily: Delphinidae\nName:",
    "Description: a slow-moving turtle with a patterned shell\nFamily: Testudinidae\nName:",
    "Description: a bright green lizard sunbathing on warm rocks\nFamily: Lacertidae\nName:"
]

for p in example_prompts:
    ids = tokenizer(p, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_length=ids.shape[1] + 35,
            num_beams=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=latin_epithet_allowed_tokens_fn,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    sci = text.split("Name:")[-1].strip()
    sci = " ".join(sci.split()[:2])
    print("----------------------------------------")
    print("Prompt:\n", p)
    print("Generated scientific name:\n", sci)

# import torch
# from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# import re

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_DIR = "./gpt2-finetuned-binomial"

# tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
# model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
# model.eval()

# def clean_scientific_name(name):
#     words = name.split()
#     if len(words) < 2:
#         return name
    
#     genus, epithet = words[0], words[1]
    
#     valid_suffixes = ['us', 'a', 'um', 'is', 'ensis', 'ii', 'i', 'ae', 'orum']
    
#     epithet_clean = epithet
#     for suffix in valid_suffixes:
#         if epithet.endswith(suffix):
#             epithet_clean = epithet
#             break
#         pattern = f'([a-z]*{suffix})[a-z]*{suffix}'
#         match = re.search(pattern, epithet)
#         if match:
#             epithet_clean = match.group(1)
#             break
#     if len(epithet_clean) > 12:
#         for suffix in valid_suffixes:
#             if suffix in epithet_clean:
#                 parts = epithet_clean.split(suffix)
#                 if len(parts) > 1:
#                     epithet_clean = parts[0] + suffix
#                     break
    
#     return f"{genus} {epithet_clean}"

# def smart_constraint_fn(batch_id, input_ids_so_far):
#     decoded = tokenizer.decode(input_ids_so_far, skip_special_tokens=True)
    
#     if "Name:" in decoded:
#         name_part = decoded.split("Name:")[-1].strip()
#         words = name_part.split()
        
#         if len(words) >= 2:
#             genus, epithet = words[0], words[1]
#             valid_endings = ['us', 'a', 'um', 'is', 'ensis', 'ii']
#             if any(epithet.endswith(ending) for ending in valid_endings):
#                 return [tokenizer.eos_token_id]
#             if len(epithet) > 10:
#                 return [tokenizer.eos_token_id]
    
#     return list(range(len(tokenizer)))


# example_prompts = [
#     "Description: a large brown bear with a scar on its paw\nFamily: Ursidae\nName:",
#     "Description: a tiny gray mouse living in a barn\nFamily: Muridae\nName:",
#     "Description: a colorful parrot that can imitate human speech\nFamily: Psittacidae\nName:",
#     "Description: a dark green frog that lives near waterfalls\nFamily: Ranidae\nName:",
#     "Description: a fast-running desert fox\nFamily: Canidae\nName:",
#     "Description: a golden-scaled fish often seen in garden ponds\nFamily: Cyprinidae\nName:",
#     "Description: a fluffy black rabbit with long ears\nFamily: Leporidae\nName:",
#     "Description: a snow owl known for silent flight\nFamily: Strigidae\nName:",
#     "Description: a gentle giant elephant with long tusks\nFamily: Elephantidae\nName:",
#     "Description: a red-striped tiger wandering in bamboo forests\nFamily: Felidae\nName:",
#     "Description: a shy hedgehog that curls into a ball\nFamily: Erinaceidae\nName:",
#     "Description: a sleek black panther that hunts at night\nFamily: Felidae\nName:",
#     "Description: a curious dolphin that plays with seaweed\nFamily: Delphinidae\nName:",
#     "Description: a slow-moving turtle with a patterned shell\nFamily: Testudinidae\nName:",
#     "Description: a bright green lizard sunbathing on warm rocks\nFamily: Lacertidae\nName:"
# ]


# for p in example_prompts:
#     inputs = tokenizer(p, return_tensors="pt")
#     input_ids = inputs.input_ids.to(DEVICE)
    
#     with torch.no_grad():
#         out = model.generate(
#             input_ids,
#             max_length=input_ids.shape[1] + 20,
#             num_beams=3,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             prefix_allowed_tokens_fn=smart_constraint_fn,
#         )
    
#     full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    
#     if "Name:" in full_text:
#         raw_name = full_text.split("Name:")[-1].strip().split('\n')[0]
#         sci_name = clean_scientific_name(raw_name)
#     else:
#         sci_name = "Failed"
    
#     print("----------------------------------------")
#     print("Prompt:\n", p)
#     print("Raw generated:", full_text.split("Name:")[-1].strip().split('\n')[0])
#     print("Cleaned scientific name:\n", sci_name)

