from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

prompt = """
Translate the following text into spanish

English: We live at the hotel.
Spanish:
""".strip()
inputs = tokenizer.encode(prompt, return_tensors="pt")
prompt_length = inputs.shape[1]
print('prompt len: {}'.format(prompt_length))

outputs = model.generate(inputs, max_new_tokens=10, output_attentions=True, return_dict_in_generate=True)
attns = outputs.attentions
sequences = outputs.sequences
new_token_ids = sequences[0, prompt_length:]
# generated text sequence
print(tokenizer.decode(new_token_ids, skip_special_tokens=True))

# the attentions at final layer for each generated token
for attn in attns:
    # TODO: might want to do something multiplicative rather than just take final layer, not sure
    last_layer_attns = attn[-1].squeeze(0)
    last_layer_attns_per_head = last_layer_attns.mean(dim=0)  # (sequence_length, sequence_length)
    print(last_layer_attns_per_head[-1])
