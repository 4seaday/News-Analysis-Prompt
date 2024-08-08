from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("skt/kobert-base-v1")

# The sentences to encode
sentences = [
    "나는 학교에 가서 공부를 한다",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]
# print(embeddings[0])

# 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings[0], embeddings[1])
# print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])