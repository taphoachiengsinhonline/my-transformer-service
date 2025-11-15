# File: app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app) # Cho phép request từ domain khác

# Tải mô hình và tokenizer
model = tf.keras.models.load_model("transformer_xs_model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 2000 # Phải giống hệt lúc train

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dữ liệu gửi lên là một mảng các object kết quả của 90 ngày
        history_results = request.json['history']
        
        # Tiền xử lý dữ liệu giống hệt lúc train
        df_hist = pd.DataFrame(history_results)
        df_hist['so_str'] = df_hist['so'].astype(str)
        # Nối tất cả các số của 90 ngày thành 1 chuỗi văn bản dài
        input_text = ''.join(df_hist['so_str'].tolist())
        
        # Chuyển text thành sequence số
        seq = tokenizer.texts_to_sequences([input_text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        
        # Dự đoán
        predictions = model.predict(padded_seq)
        
        # Lấy ra con số có xác suất cao nhất cho mỗi vị trí
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
    app.run(host='0.0.0.0', port=5001)