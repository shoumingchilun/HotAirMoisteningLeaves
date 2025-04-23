import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# 数据集路径
DATASET_DIR = "dataset"
data_files = sorted(os.listdir(DATASET_DIR))

# 训练集和测试集划分
train_files = data_files[:15]
test_files = data_files[15:]

# 超参数
sequence_length = 150  # 输入序列长度
prediction_steps = 60  # 预测未来两分钟的时间步数（假设每2秒1个样本）
features = ['material_flow', 'entry_temp', 'entry_moisture',
            'steam_valve', 'water_addition', 'env_temp', 'env_humidity']
labels = ['out_temp', 'out_moisture']

# 初始化标准化器
scaler_features = StandardScaler()
scaler_labels = StandardScaler()


# 预加载所有训练数据用于标准化
def preload_and_fit_scalers():
    all_train_data = []
    for file in train_files:
        df = pd.read_csv(os.path.join(DATASET_DIR, file))
        all_train_data.append(df)
    full_train_df = pd.concat(all_train_data)

    # 仅使用训练数据拟合标准化器
    scaler_features.fit(full_train_df[features])
    scaler_labels.fit(full_train_df[labels])

    joblib.dump(scaler_features, "/temp/scaler_features.pkl")
    joblib.dump(scaler_labels, "/temp/scaler_labels.pkl")


preload_and_fit_scalers()


def load_and_process_file(file_path, is_train=False):
    """加载并处理单个文件"""
    df = pd.read_csv(file_path)

    # 特征标准化
    scaled_features = scaler_features.transform(df[features])
    scaled_labels = scaler_labels.transform(df[labels])

    # 构建时间序列样本
    X, y = [], []
    max_index = len(df) - sequence_length - prediction_steps + 1
    for i in range(max_index):
        X.append(scaled_features[i:i + sequence_length])
        y.append(scaled_labels[i + sequence_length: i + sequence_length + prediction_steps])

    return np.array(X), np.array(y)


# 加载训练数据
X_train, y_train = [], []
for file in train_files:
    X, y = load_and_process_file(os.path.join(DATASET_DIR, file), is_train=True)
    X_train.append(X)
    y_train.append(y)
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# 加载测试数据
X_test, y_test = [], []
for file in test_files:
    X, y = load_and_process_file(os.path.join(DATASET_DIR, file))
    X_test.append(X)
    y_test.append(y)
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")


# 改进的TCN模型
def build_tcn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # 时间卷积网络层
        layers.Conv1D(64, 3, dilation_rate=1, padding='causal', activation='relu'),
        layers.Conv1D(64, 3, dilation_rate=2, padding='causal', activation='relu'),
        layers.Conv1D(128, 3, dilation_rate=4, padding='causal', activation='relu'),
        layers.Conv1D(128, 3, dilation_rate=8, padding='causal', activation='relu'),

        # 输出层：每个时间步预测两个值
        layers.Conv1D(2, 1),  # 保持时间维度
        layers.Lambda(lambda x, steps: x[:, -steps:, :], arguments={'steps': prediction_steps})  # 取最后prediction_steps步
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model


# 构建并训练模型
model = build_tcn_model((sequence_length, len(features)))
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32)

# 模型评估与可视化
y_pred = model.predict(X_test)

# 逆标准化处理
y_pred_inv = scaler_labels.inverse_transform(
    y_pred.reshape(-1, 2)).reshape(y_pred.shape)
y_test_inv = scaler_labels.inverse_transform(
    y_test.reshape(-1, 2)).reshape(y_test.shape)

# 可视化最后10个预测序列
plt.figure(figsize=(15, 8))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(y_test_inv[-10, :, i], label='Actual', marker='o')
    plt.plot(y_pred_inv[-10, :, i], label='Predicted', marker='x')
    plt.title(f"{labels[i]} Prediction Comparison")
    plt.legend()
plt.tight_layout()
plt.show()

# 保存模型
model.save("/temp/optimized_tcn_model.h5")
print("Model saved as optimized_tcn_model.h5")