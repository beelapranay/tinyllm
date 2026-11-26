import numpy as np

class FeedForward:
	def __init__(self, hidden_size: int, intermediate_size: int):
		self.w_up = np.random.randn(hidden_size, intermediate_size)
		self.w_down = np.random.randn(intermediate_size, hidden_size)

	def forward(self, x: np.ndarray):
		up_projection = x @ self.w_up
		up_projection = np.maximum(0, up_projection)
		down_projection = up_projection @ self.w_down

		return down_projection

from tokenizer import Tokenizer
from embedding import Embedding

if __name__ == "__main__":
	tokenizer = Tokenizer()
	embedding = Embedding(256, 32)
	ffn = FeedForward(32, 128)

	token_ids = tokenizer.encode("Hello!")
	x = embedding.forward(token_ids)

	output = ffn.forward(x)

	print("input shape:", x.shape)
	print("output shape:", output.shape)

	assert output.shape == x.shape
	assert not np.array_equal(x, output)
	assert np.isfinite(output).all()

	print("All tests passed!")