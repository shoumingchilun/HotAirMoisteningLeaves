import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# 数据集路径
DATASET_DIR = "../dataset"
data_files = sorted(os.listdir(DATASET_DIR))

# 训练集和测试集划分
train_files = data_files[:15]
test_files = data_files[15:]

# 超参数
sequence_length = 150  # 选取较长时间步长，考虑设备的延时效应


def load_and_process_file(file_path):
    """加载单个文件，并构造特征和标签"""
    df = pd.read_csv(file_path)

    # 选取特征和目标变量
    features = ['material_flow', 'entry_temp', 'entry_moisture',
                'steam_valve', 'water_addition', 'env_temp', 'env_humidity']
    labels = ['out_temp', 'out_moisture']

    # 标准化特征（仅对输入变量进行）
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # 由于设备有延迟影响，目标变量要向后偏移
    delay = sequence_length  # 目标值的延迟步数，模拟设备的滞后效应
    df_labels = df[labels].shift(-delay).dropna().values
    df_features = df[features].iloc[:-delay].values

    # 组织时间序列数据
    X, y = [], []
    for i in range(len(df_features) - sequence_length):
        X.append(df_features[i:i + sequence_length])
        y.append(df_labels[i])

    return np.array(X), np.array(y)


# 处理所有训练文件
X_train_list, y_train_list = zip(*[load_and_process_file(os.path.join(DATASET_DIR, file)) for file in train_files])
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

# 处理所有测试文件
X_test_list, y_test_list = zip(*[load_and_process_file(os.path.join(DATASET_DIR, file)) for file in test_files])
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")

# 定义 TCN 模型
model = models.Sequential([
    layers.Input(shape=(sequence_length, X_train.shape[2])),
    layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal'),
    layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal'),
    layers.Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)  # 预测出口温度和水分
])

model.compile(optimizer='adam', loss='mse')

# 训练模型
epochs = 100
batch_size = 32
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)



# 评估模型
y_pred = model.predict(X_test)

# 绘制预测与实际值对比
plt.figure(figsize=(10, 6))
plt.plot(y_test[-100:, 0], label='Actual Temp', alpha=0.7)
plt.plot(y_pred[-100:, 0], label='Predicted Temp', alpha=0.7)
plt.plot(y_test[-100:, 1], label='Actual Moisture', alpha=0.7)
plt.plot(y_pred[-100:, 1], label='Predicted Moisture', alpha=0.7)
plt.title('Comparison of Actual vs Predicted Temperature & Moisture')
plt.legend()
plt.show()

# 保存模型
model.save("hot_air_tcn_model.h5")
print("Model saved as hot_air_tcn_model.h5")
model.summary()
