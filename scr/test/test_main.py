from FlagEmbedding import BGEM3FlagModel
import jieba
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device)
text = "The quick brown fox jumps over the lazy dog."
inputs = jieba.lcut(text)
print(inputs)
for word in inputs:
    embedding = model.encode(word)['dense_vecs']
    print(f"Word: {word}, Embedding: {embedding}")