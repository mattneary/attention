from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import torch

def aggregate_attention(attn):
    '''Extract attention vector mapping onto preceding token'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        # We zero the first entry because it's what's called
        # null attention (https://aclanthology.org/W19-4808.pdf)
        vec = torch.concat((
            torch.tensor([0.]),
            attns_per_head[-1][1:],
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)

def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

def decode(tokens):
    '''Turn tokens into text with mapping index'''
    full_text = ''
    offset = 0
    token_index = [0]
    chunks = []
    for i, token in enumerate(tokens):
        text = tokenizer.decode(token)
        full_text += text
        offset += len(text)
        token_index.append(offset)
        chunks.append(text)
    print('|'.join(chunks))
    return full_text, token_index

def get_completion(prompt):
    '''Get full text, token mapping, and attention matrix for a completion'''
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        tokens,
        max_new_tokens=5,
        output_attentions=True,
        return_dict_in_generate=True
    )
    sequences = outputs.sequences
    attn_m = heterogenous_stack([
        torch.tensor([
            1 if i == j else 0
            for j, token in enumerate(tokens[0])
        ])
        for i, token in enumerate(tokens[0])
    ] + list(map(aggregate_attention, outputs.attentions)))
    decoded, token_index = decode(sequences[0])
    return decoded, token_index, attn_m

def choose_range(token_index):
    '''Choose subset range from token_index'''
    choices = set([])
    while len(list(choices)) != 2:
        choices = set(random.choices(token_index, k=2))
    return sorted(list(choices))

def text_to_token_range(token_index, rng):
    '''Convert text range into token range'''
    a, b = token_index.index(rng[0]), token_index.index(rng[1])
    xs = [
        1. if (i <= b and i >= a) else 0.
        for i, _ in enumerate(token_index)
    ][1:]
    return torch.tensor(xs) / sum(xs)

def highlight_range(text, rng):
    '''Annotate a string with a given subset range'''
    front = text[:rng[0]]
    mid = text[rng[0]:rng[1]]
    back = text[rng[1]:]
    return front + '{' + mid + '}' + back

def show_matrix(xs):
    for x in xs:
        line = ''
        for y in x:
            line += '{:.4f}\t'.format(float(y))
        print(line)

prompt = """
The quick brown
""".strip().replace('\n', ' ')

result, token_index, attn_m = get_completion(prompt)
show_matrix(attn_m)
text_range = choose_range(token_index)
# visualize the randomly chosen range of the text
print(highlight_range(result, text_range))

focus_vec = text_to_token_range(token_index, text_range)
attn_vec = torch.matmul(focus_vec, attn_m)

# output the attention vector computed from attention matrix and focus vector
print(attn_vec)
