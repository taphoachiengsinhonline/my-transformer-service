# File: app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import pandas as pd
import os
import json

# THÊM MỚI: Thư viện Google Cloud Storage
from google.cloud import storage

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH ---
MAX_LEN = 2000 # Phải giống hệt lúc train
MODEL_LOCAL_PATH = "transformer_xs_model.h5"
TOKENIZER_LOCAL_PATH = "tokenizer.pickle"
MODEL_GCS_PATH = "models/transformer_v1" # Đường dẫn thư mục trên GCS để lưu model

# --- THIẾT LẬP KẾT NỐI GCS ---
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
GCS_CREDENTIALS_JSON = os.environ.get('GCS_CREDENTIALS')
gcs_bucket = None

if GCS_BUCKET_NAME and GCS_CREDENTIALS_JSON:
    try:
        credentials_dict = json.loads(GCS_CREDENTIALS_JSON)
        storage_client = storage.Client(credentials=storage.credentials.Credentials.from_service_account_info(credentials_dict))
        gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)
        print(f"✅ [GCS] Kết nối thành công đến bucket: {GCS_BUCKET_NAME}")
    except Exception as e:
        print(f"❌ [GCS] Lỗi kết nối GCS: {e}")
else:
    print("⚠️ [GCS] Biến môi trường GCS chưa được thiết lập. Sẽ chỉ dùng model local.")

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
def load_model_from_gcs():
    """Cố gắng tải model từ GCS."""
    if not gcs_bucket:
        return None
    try:
        print("🔍 [GCS] Đang thử tải model từ GCS...")
        # Keras cần lưu/tải từ một đường dẫn thư mục, GCS IO handler sẽ xử lý việc này
        gcs_path = f"gs://{GCS_BUCKET_NAME}/{MODEL_GCS_PATH}"
        model = tf.keras.models.load_model(
            gcs_path,
            custom_objects={
                "PositionalEmbedding": PositionalEmbedding,
                "TransformerEncoder": TransformerEncoder
            }
        )
        print("✅ [GCS] Tải model từ GCS thành công!")
        return model
    except Exception as e:
        print(f"⚠️ [GCS] Không tìm thấy model trên GCS hoặc có lỗi: {e}. Sẽ dùng model local.")
        return None

def load_model_from_local():
    """Tải model dự phòng từ file local."""
    print("💾 Đang tải model dự phòng từ file local...")
    return tf.keras.models.load_model(
        MODEL_LOCAL_PATH,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoder": TransformerEncoder
        }
    )

def save_model_to_gcs(model_to_save):
    """Lưu model lên GCS."""
    if not gcs_bucket:
        print("⚠️ [GCS] Không thể lưu model vì GCS chưa được cấu hình.")
        return False
    try:
        print("💾 [GCS] Đang lưu model mới lên GCS...")
        gcs_path = f"gs://{GCS_BUCKET_NAME}/{MODEL_GCS_PATH}"
        model_to_save.save(gcs_path)
        print("✅ [GCS] Lưu model lên GCS thành công!")
        return True
    except Exception as e:
        print(f"❌ [GCS] Lỗi khi lưu model lên GCS: {e}")
        return False

# --- KHỞI TẠO BIẾN TOÀN CỤC ---
# Tải tokenizer
with open(TOKENIZER_LOCAL_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Tải model (Ưu tiên GCS, nếu thất bại thì dùng local)
model = load_model_from_gcs()
if model is None:
    model = load_model_from_local()


# --- ĐỊNH NGHĨA CÁC API ENDPOINT ---

@app.route('/predict', methods=['POST'])
def predict():
    # Logic predict không thay đổi
    try:
        history_results = request.json['history']
        # ... (toàn bộ logic tiền xử lý và dự đoán giữ nguyên)
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

@app.route('/learn', methods=['POST'])
def learn():
    """API mới để thực hiện học thêm (fine-tuning)."""
    global model # Khai báo để có thể gán lại model mới
    try:
        # 1. Lấy dữ liệu học từ request
        training_sample = request.json['sample'] # sample = { 'input': [...], 'output': [...] }
        input_seq = training_sample['input']
        target_gdb = training_sample['output']

        # 2. Tiền xử lý dữ liệu (giống lúc train)
        input_pad = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=MAX_LEN, padding='post')
        y_split = [np.array([d]) for d in target_gdb] # Chuyển thành dạng batch size 1

        # 3. Học thêm với learning rate nhỏ
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # learning rate rất nhỏ
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(input_pad, y_split, epochs=3, verbose=0) # Chỉ học vài epoch

        print("🧠 Model đã học thêm từ dữ liệu mới.")

        # 4. Lưu lại model đã "thông minh" hơn lên GCS
        save_model_to_gcs(model)

        return jsonify({'success': True, 'message': 'Model learned and updated successfully.'})

    except Exception as e:
        return jsonify({'success': False, 'message': f"Error during learning: {e}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
