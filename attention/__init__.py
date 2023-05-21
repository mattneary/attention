from flask import Flask
from .attention import get_completion
import json

app = Flask(__name__)

prompt = """
The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of "one world, one dream". Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the "Journey of Harmony", lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics.

After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch trav- eled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event.

Q: What was the theme
A:
""".strip()

result, tokenized, attn_m = get_completion(prompt)
sparse = attn_m.to_sparse()

@app.route("/attention")
def attention_view():
    indices, values = sparse.indices(), sparse.values()
    return json.dumps({
        'tokens': tokenized,
        'attn_indices': indices.T.numpy().tolist(),
        'attn_values': values.numpy().tolist(),
    })
