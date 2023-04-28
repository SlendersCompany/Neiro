import tensorflow as tf
import numpy as np
import argparse

# Подготовка данных
text = open('poems.txt', 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char_to_idx = {char:idx for idx, char in enumerate(vocab)}
idx_to_char = np.array(vocab)

text_as_int = np.array([char_to_idx[char] for char in text])

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Создание модели
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Загрузка весов модели
model.load_weights("poetry_generator.h5")

# Генерация текста
def generate_text(model, start_string):
    num_generate = 300
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])

    return ''.join(text_generated)

# Обработка аргументов командной строки
parser = argparse.ArgumentParser(description='Генерация стихотворений и романсов')
parser.add_argument('type', choices=['стихотворение', 'романс'], help='Тип текста для генерации')
args = parser.parse_args()

if args.type == 'стихотворение':
    start_string = u"Романтика"
elif args.type == 'романс':
    start_string = u"Жили-были два кума"

print(generate_text(model, start_string))
