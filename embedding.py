import numpy as np

class Embedding:
	def __init__(self, vocab_size: int, hidden_size: int):
		self.weights = np.random.randn(vocab_size, hidden_size)

	def forward(self, token_ids: list[int]):
		return self.weights[token_ids]

from tokenizer import Tokenizer

if __name__ == "__main__":
	tokenizer = Tokenizer()
	embedding = Embedding(256, 32)

	token_ids = tokenizer.encode("Hello!")
	x = embedding.forward(token_ids)

	print("Token IDs:", token_ids)
	print("Weights shape:", embedding.weights.shape)
	print("Output shape:", x.shape)

	assert x.shape == (6, 32)
	assert np.array_equal(x[2], x[3])