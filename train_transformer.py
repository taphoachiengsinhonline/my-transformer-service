# File: train_transformer.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Phần 1: Các Lớp và Hàm xây dựng Mô hình Transformer ---
# (Copy y hệt code các class PositionalEmbedding, TransformerEncoder từ câu trả lời trước)
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

# --- Phần 2: Tải và Tiền xử lý dữ liệu ---
print("Bắt đầu xử lý dữ liệu...")
df = pd.read_csv('results.csv')

# Chuyển tất cả các số thành chuỗi và nối lại
df['so_str'] = df['so'].astype(str)
# Pivot table để mỗi ngày là một dòng
pivot_df = df.pivot_table(index='ngay', columns='giai', values='so_str', aggfunc='first').fillna('')
pivot_df.index = pd.to_datetime(pivot_df.index, format='%d/%m/%Y')
pivot_df = pivot_df.sort_index()

# Tạo chuỗi dữ liệu cho mô hình
def create_sequences(data, lookback_days=90):
    X, Y = [], []
    gdb_series = data.loc[data['giai'] == 'ĐB', ['date_obj', 'so_str']].set_index('date_obj')['so_str']
    
    # Tạo chuỗi văn bản lớn từ tất cả các giải
    full_text_series = data.groupby('date_obj')['so_str'].apply(lambda x: ''.join(x)).sort_index()

    for i in range(len(full_text_series) - lookback_days):
        # Input: 90 ngày dữ liệu, mỗi ngày là 1 chuỗi dài
        input_seq = full_text_series.iloc[i : i + lookback_days].str.cat(sep='')
        # Output: 5 số GDB của ngày tiếp theo
        target_gdb = str(gdb_series.iloc[i + lookback_days]).zfill(5)
        
        if len(target_gdb) == 5 and target_gdb.isdigit():
            X.append(input_seq)
            Y.append([int(d) for d in target_gdb])

    return np.array(X), np.array(Y)

X_text, y = create_sequences(df, lookback_days=90)
print(f"Đã tạo {len(X_text)} mẫu huấn luyện.")

# Tokenize dữ liệu văn bản (chuyển chữ thành số)
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
# Đảm bảo các chuỗi có độ dài bằng nhau
MAX_LEN = 2000 # Giới hạn độ dài chuỗi đầu vào, bạn có thể điều chỉnh
X_pad = keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=MAX_LEN, padding='post')

# Tách y thành 5 output riêng biệt
y_split = [y[:, i] for i in range(5)]

# --- Phần 3: Xây dựng, Compile và Huấn luyện mô hình ---
VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBED_DIM = 32
NUM_HEADS = 2
DENSE_DIM = 64

def build_model():
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int64")
    x = PositionalEmbedding(VOCAB_SIZE, EMBED_DIM, MAX_LEN)(inputs)
    x = TransformerEncoder(EMBED_DIM, DENSE_DIM, NUM_HEADS)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = []
    for i in range(5):
        output = layers.Dense(10, activation="softmax", name=f"pos_{i}")(x)
        outputs.append(output)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    # Sửa ở đây: Cung cấp một dictionary cho metrics
    metrics={
        "pos_0": "accuracy",
        "pos_1": "accuracy",
        "pos_2": "accuracy",
        "pos_3": "accuracy",
        "pos_4": "accuracy",
    }
)
model.summary()

print("\nBắt đầu huấn luyện mô hình...")
EPOCHS = 30 # Số lần lặp lại quá trình học, có thể tăng lên nếu cần
BATCH_SIZE = 16
model.fit(
    X_pad, 
    y_split, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_split=0.1 # Dùng 10% dữ liệu để kiểm tra
)

# --- Phần 4: Lưu lại mô hình ---
model.save("transformer_xs_model.h5")
# Lưu cả tokenizer để dùng cho dự đoán sau này
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Đã huấn luyện và lưu mô hình thành công vào file transformer_xs_model.h5 và tokenizer.pickle!")