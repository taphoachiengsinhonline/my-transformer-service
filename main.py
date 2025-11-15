# File: main.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
# THÊM MỚI ĐỂ SỬA LỖI
from google.oauth2 import service_account

print("--- [INFO] Bắt đầu khởi tạo ứng dụng ---")

# --- Ứng dụng Flask ---
app = Flask(__name__)
CORS(app)

# --- Cấu hình ---
MAX_LEN = 2000
MODEL_LOCAL_PATH = "transformer_xs_model.h5"
TOKENIZER_LOCAL_PATH = "tokenizer.pickle"
MODEL_GCS_PATH = "models/transformer_v1"

# --- Thiết lập GCS ---
gcs_bucket = None
try:
    GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
    GCS_CREDENTIALS_JSON = os.environ.get('GCS_CREDENTIALS')
    if GCS_BUCKET_NAME and GCS_CREDENTIALS_JSON:
        credentials_dict = json.loads(GCS_CREDENTIALS_JSON)
        
        # SỬA LỖI KẾT NỐI GCS
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        storage_client = storage.Client(credentials=credentials)
        
        gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)
        print(f"✅ [GCS] Kết nối thành công đến bucket: {GCS_BUCKET_NAME}")
    else:
        print("⚠️ [GCS] Biến môi trường GCS chưa được thiết lập. Sẽ chỉ dùng model local.")
except Exception as e:
    print(f"❌ [GCS] Lỗi khởi tạo GCS: {e}")

# --- Các phần còn lại của file giữ nguyên không đổi ---

# --- Định nghĩa Lớp Custom ---
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

custom_objects = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerEncoder": TransformerEncoder
}

# --- Hàm Tải/Lưu Model ---
def load_model_from_gcs():
    if not gcs_bucket: return None
    try:
        print("🔍 [GCS] Đang thử tải model từ GCS...")
        gcs_path = f"gs://{GCS_BUCKET_NAME}/{MODEL_GCS_PATH}"
        loaded_model = tf.keras.models.load_model(gcs_path, custom_objects=custom_objects)
        print("✅ [GCS] Tải model từ GCS thành công!")
        return loaded_model
    except Exception as e:
        if "NotFoundError" in str(e) or "doesn't exist" in str(e):
             print(f"ℹ️ [GCS] Không tìm thấy model trên GCS. Đây có thể là lần deploy đầu tiên.")
        else:
             print(f"⚠️ [GCS] Lỗi khác khi tải model từ GCS: {e}")
        return None

def load_model_from_local():
    try:
        if not os.path.exists(MODEL_LOCAL_PATH):
            print(f"⚠️ Không tìm thấy file model local tại: {MODEL_LOCAL_PATH}")
            return None
        print("💾 Đang tải model dự phòng từ file local...")
        loaded_model = tf.keras.models.load_model(MODEL_LOCAL_PATH, custom_objects=custom_objects)
        print("✅ Tải model local thành công.")
        return loaded_model
    except Exception as e:
        print(f"❌ Không thể tải model local: {e}")
        return None

def save_model_to_gcs(model_to_save):
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

# --- Khởi tạo Biến Toàn cục ---
print("--- [INFO] Đang tải tokenizer và model ---")
model = None
tokenizer = None
try:
    with open(TOKENIZER_LOCAL_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
        print("✅ Tải tokenizer thành công.")
except Exception as e:
    print(f"CRITICAL ERROR: Không thể tải tokenizer: {e}")
model = load_model_from_gcs()
if model is None:
    model = load_model_from_local()
if model is None:
    print("CRITICAL ERROR: Không thể tải được bất kỳ model nào. API '/predict' và '/learn' sẽ không hoạt động.")
else:
    print("--- [INFO] Model đã sẵn sàng ---")

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'success': False, 'message': 'Model hoặc tokenizer chưa được tải.'}), 503
    try:
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
        return jsonify({'success': False, 'message': f"Lỗi khi dự đoán: {e}"})

@app.route('/learn', methods=['POST'])
def learn():
    global model
    if model is None:
        return jsonify({'success': False, 'message': 'Model chưa được tải.'}), 503
    try:
        training_sample = request.json['sample']
        input_seq = training_sample['input']
        target_gdb = training_sample['output']
        input_pad = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=MAX_LEN, padding='post')
        y_split = [np.array([d]) for d in ta
