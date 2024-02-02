import os
from gensim.models import word2vec


path = "./models/"

files = os.listdir(path)

for file_name in files:
    file_path = path + file_name
    model = word2vec.Word2Vec.load(file_path)
    print(file_path)
    try:
        # print("人生-愛")
        # for item, value in model.wv.most_similar(positive=["人生"], negative=["愛"]):
        #     print(item, value)
        # print("--------------")

        print("人生")
        for item, value in model.wv.most_similar(positive=["人生"]):
            print(item, value)
        print("--------------")

        print("愛")
        for item, value in model.wv.most_similar(positive=["愛"]):
            print(item, value)
        print("--------------")
    except:
        continue
