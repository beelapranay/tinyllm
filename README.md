# tinyllm

A miniature transformer inference engine built from scratch in Python and NumPy, to understand exactly what happens between entering a prompt and generating the next token.

No PyTorch, Hugging Face or MLX. Every component is implemented by hand with NumPy. Weights are randomly initialized for now, so output is meaningless; the goal is to get the architecture and tensor shapes right before loading pretrained weights.

## Model configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 256 (UTF-8 bytes) |
| Hidden size | 32 |
| FFN intermediate size | 128 |
| Transformer layers | 4 |
| Attention heads | 4 (2 key/value heads, GQA) |
| Head dimension | 8 |
| Sliding window | 16 |

## Architecture

```text
text → Tokenizer → Embedding → [Transformer block × 4] → RMSNorm → LM head → next token

Transformer block:
  RMSNorm → QKV projections → RoPE → causal (full / sliding window) attention
  → output projection → residual → RMSNorm → gated FFN → residual
```

## Progress

- [x] Byte-level tokenizer (`tokenizer.py`)
- [x] Embedding lookup (`embedding.py`)
- [x] Feed-forward network (`feedforward.py`); ReLU for now, SwiGLU next
- [ ] RMSNorm
- [ ] Grouped-query attention with causal and sliding window masks
- [ ] Rotary positional embeddings (RoPE)
- [ ] Transformer block
- [ ] Full model and autoregressive generation
- [ ] Loading pretrained weights
- [ ] KV cache
- [ ] 4-bit quantization

## Running

Requires Python 3 and NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy
```

Each module has a self-test you can run directly:

```bash
python tokenizer.py
python embedding.py
python feedforward.py
```
