# attention

Attention is a key lens for understanding Transformer results.
This project exposes the attention scores of an LLM run as a
matrix.

Here's an example of what the matrix output of this project will look like:

![attention matrix](matrix.png)

Why model it as a matrix? Given attention matrix `m` you can model
a range of text as focus vector `f` and then multiply
`torch.matmul(f, m)` to get the attention vector for that range.`


## How to Run

```sh
$ poetry run python attention/attention.py
```
