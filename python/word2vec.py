import os
import io
import MeCab
from gensim.models import word2vec

tagger = MeCab.Tagger("-Owakati")

path = "./text_data/"

files = os.listdir(path)

for file_name in files:
    file_path = path + file_name
    with io.open(file_path, encoding="utf-8") as f:
        text = f.read().lower()
        sentences = [
            tagger.parse(sentence).strip().split() for sentence in text.split("。")
        ]
        print(sentences)
        model = word2vec.Word2Vec(
            sentences, vector_size=100, window=5, min_count=1, epochs=200
        )
        model_name = file_name.replace("preprocess_done_", "").replace(".txt", "")
        model.save(f"./models/{model_name}.model")
