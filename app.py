# File: app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # THÊM MỚI
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

MAX_LEN = 2000 # Phải giống hệt lúc train

# =================================================================
# BƯỚC 1: ĐỊNH NGHĨA LẠI CÁC LỚP CUSTOM
# Copy y hệt các class này từ file train_transformer.py sang đây
# =================================================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, maxlen, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
# =================================================================

# Tải tokenizer (giữ nguyên)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# =================================================================
# BƯỚC 2: SỬA LẠI LỆNH TẢI MÔ HÌNH
# =================================================================
# Tạo một dictionary để "đăng ký" các lớp custom
custom_objects = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerEncoder": TransformerEncoder
}

# Tải mô hình và truyền vào custom_objects
model = tf.keras.models.load_model(
    "transformer_xs_model.h5",
    custom_objects=custom_objects
)
# =================================================================


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Phần logic predict giữ nguyên, không thay đổi
        history_results = request.json['history']
        
        df_hist = pd.DataFrame(history_results)
        df_hist['so_str'] = df_hist['so'].astype(str)
        input_text = ''.join(df_hist['so_str'].tolist())
        
        seq = tokenizer.texts_to_sequences([input_text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        
        predictions = model.predict(padded_seq)
        
        result = {
            'hangChucNgan': int(np.argmax(predictions[0])),
            'hangNgan': int(np.argmax(predictions[1])),
            'hangTram': int(np.argmax(predictions[2])),
            'hangChuc': int(np.argmax(predictions[3])),
            'hangDonVi': int(np.argmax(predictions[4])),
        }
        
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Chú ý: Railway sẽ dùng Gunicorn, lệnh này chỉ để chạy local
    app.run(host='0.0.0.0', port=5001)
