from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    prompt_length = inputs.shape[1]
    print('input tokens: {}'.format(prompt_length))
    outputs = model.generate(inputs, max_new_tokens=10, output_attentions=True, return_dict_in_generate=True)
    return outputs

def decode_response(sequences, prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    prompt_length = inputs.shape[1]
    new_token_ids = sequences[0, prompt_length:]
    # generated text sequence
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)

def aggregate_attention(attn):
    # TODO: might want to do something multiplicative rather than just take final layer, not sure
    last_layer_attns = attn[-1].squeeze(0)
    last_layer_attns_per_head = last_layer_attns.mean(dim=0)  # (sequence_length, sequence_length)
    return last_layer_attns_per_head[-1]

prompt = """
In the beginning
""".strip()

outputs = generate_response(prompt)
sequences = outputs.sequences
print(decode_response(sequences, prompt))

attns = outputs.attentions
for attn in attns:
    print(aggregate_attention(attn))
