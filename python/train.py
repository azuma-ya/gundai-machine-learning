from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop

from tensorflow.keras.utils import get_file
import numpy as np
import random
import sys
import io
import os

path = "./text_data/preprocess_done.txt"
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
print("corpus length:", len(text))

chars = sorted(list(set(text)))
print("total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("nb sequences:", len(sentences))

print("Vectorization...")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# モデルをビルドする
print("Build model...")
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation="softmax"))

optimizer = RMSprop(lr=0.01)

model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# エポックごとに文章を自動生成させる
def on_epoch_end(epoch, _):
    print()
    print("----- Generating text after Epoch: %d" % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("----- diversity:", diversity)
        generated = ""
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# modelsというディレクトリを作成して、epochごとにその時点での重みを保存する
os.makedirs("models", exist_ok=True)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join("models", "model_{epoch:02d}.h5"),
    monitor="val_loss",
    verbose=1,
)

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# 学習途中での出力を確かめたい場合は、callbacksにprint_calbackを追加する
model.fit(x, y, batch_size=128, epochs=100, callbacks=[model_checkpoint])

# 学習が完了したら、モデルと重さを保存する
model_json_str = model.to_json()
open("complete_model_epoch_60.json", "w").write(model_json_str)
model.save_weights("complete_model_epoch_60.h5")
