class Tokenizer:
    def encode(self, text: str) -> list[int]:
        res = []
        encoded_text = text.encode()

        res.extend(encoded_text) #extend iterates over an object and adds its elements individually

        return res

    def decode(self, token_ids: list[int]) -> str:
        text = bytes(token_ids)
        return bytes.decode(text)


if __name__ == "__main__":
    tokenizer = Tokenizer()

    test_cases = [
        "Hello!",
        "What is your name?",
        "Hello 😀",
        "తెలుగు",
        "123 @#$%",
        "",
        "café"
    ]

    for text in test_cases:
        assert tokenizer.decode(tokenizer.encode(text)) == text
        print(f"Passed: {text!r}")