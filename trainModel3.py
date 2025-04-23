import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import layers, models, callbacks, activations # 引入 callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# --- 配置参数 ---
DATASET_DIR = "dataset2"
OUTPUT_DIR = "output" # 指定输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SCALER_FEATURES_PATH = os.path.join(OUTPUT_DIR, "scaler_features.pkl")
SCALER_LABELS_PATH = os.path.join(OUTPUT_DIR, "scaler_labels.pkl")
MODEL_PATH = os.path.join(OUTPUT_DIR, "Gas_model.h5")

data_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]) # 确保只读取csv

# 训练集和测试集划分 (假设按文件名排序)
train_files = data_files[:4]
test_files = data_files[4:]

# --- 超参数 ---
sequence_length = 150  # 输入序列长度 T
prediction_steps = 1  # 预测未来 N 步 (这里是预测紧接着的下一步)
features = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
labels = ['CO', 'NOX']
feature_dim = len(features)
label_dim = len(labels)

# --- 初始化标准化器 ---
scaler_features = StandardScaler()
scaler_labels = StandardScaler()

# --- 预加载所有训练数据用于标准化 ---
def preload_and_fit_scalers():
    print("Preloading training data and fitting scalers...")
    all_train_data = []
    for file in train_files:
        file_path = os.path.join(DATASET_DIR, file)
        try:
            df = pd.read_csv(file_path)
            # 简单检查列是否存在
            if not all(f in df.columns for f in features + labels):
                print(f"Warning: Skipping file {file}. Missing required columns.")
                continue
            all_train_data.append(df[features + labels]) # 只读取需要的列
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not all_train_data:
        raise ValueError("No valid training data loaded. Check dataset path and file contents.")

    full_train_df = pd.concat(all_train_data, ignore_index=True)

    # 处理可能的NaN值 (例如，用均值填充，或先检查再决定策略)
    if full_train_df[features].isnull().values.any():
        print("Warning: NaNs detected in training features. Filling with mean.")
        full_train_df[features] = full_train_df[features].fillna(full_train_df[features].mean())
    if full_train_df[labels].isnull().values.any():
        print("Warning: NaNs detected in training labels. Filling with mean.")
        full_train_df[labels] = full_train_df[labels].fillna(full_train_df[labels].mean())

    # 仅使用训练数据拟合标准化器
    scaler_features.fit(full_train_df[features])
    scaler_labels.fit(full_train_df[labels])

    joblib.dump(scaler_features, SCALER_FEATURES_PATH)
    joblib.dump(scaler_labels, SCALER_LABELS_PATH)
    print(f"Scalers saved to {SCALER_FEATURES_PATH} and {SCALER_LABELS_PATH}")

preload_and_fit_scalers() # 执行拟合

# --- 数据加载和处理函数 ---
def load_and_process_file(file_path):
    """加载并处理单个文件，构建序列"""
    try:
        df = pd.read_csv(file_path)
        if not all(f in df.columns for f in features + labels):
            print(f"Warning: Skipping file {file_path}. Missing required columns.")
            return None, None

        # 处理可能的NaN值 (测试集用训练集的均值填充更合适，但这里简化处理)
        if df[features].isnull().values.any():
            print(f"Warning: NaNs detected in features of {file_path}. Filling with mean.")
            df[features] = df[features].fillna(scaler_features.mean_) # 使用训练集的均值
        if df[labels].isnull().values.any():
            print(f"Warning: NaNs detected in labels of {file_path}. Filling with mean.")
            df[labels] = df[labels].fillna(scaler_labels.mean_) # 使用训练集的均值

        # 特征和标签标准化
        scaled_features = scaler_features.transform(df[features])
        scaled_labels = scaler_labels.transform(df[labels])

        # 构建时间序列样本
        X, y = [], []
        # 确保索引不越界
        max_start_index = len(df) - sequence_length - prediction_steps + 1
        if max_start_index <= 0:
            print(f"Warning: File {file_path} is too short ({len(df)} rows) for sequence_length={sequence_length} and prediction_steps={prediction_steps}. Skipping.")
            return None, None

        for i in range(max_start_index):
            X.append(scaled_features[i : i + sequence_length])
            # 目标是输入序列结束后的 prediction_steps 个标签
            y.append(scaled_labels[i + sequence_length : i + sequence_length + prediction_steps])

        if not X: # 如果循环没有产生任何数据
             return None, None

        return np.array(X), np.array(y).reshape(-1, prediction_steps, label_dim) # 确保y的形状是 (样本数, 预测步数, 标签维度)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# --- 加载训练数据 ---
print("Loading training data...")
X_train_list, y_train_list = [], []
for file in train_files:
    X, y = load_and_process_file(os.path.join(DATASET_DIR, file))
    if X is not None and y is not None:
        X_train_list.append(X)
        y_train_list.append(y)

if not X_train_list:
    raise ValueError("Failed to load any valid training samples.")
X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)

# --- 加载测试数据 ---
print("Loading testing data...")
X_test_list, y_test_list = [], []
for file in test_files:
    X, y = load_and_process_file(os.path.join(DATASET_DIR, file))
    if X is not None and y is not None:
        X_test_list.append(X)
        y_test_list.append(y)

if not X_test_list:
    raise ValueError("Failed to load any valid testing samples.")
X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- 改进的TCN模型 (增加层数以扩大感受野) ---
def build_tcn_model(input_shape, output_steps, output_dim):
    """构建包含残差连接、层归一化和Dropout的TCN模型"""
    kernel_size = 3
    filters = 64  # TCN块中的滤波器数量 (可以调整)
    dropout_rate = 0.1 # Dropout比例 (可以调整)

    inputs = layers.Input(shape=input_shape)
    x = inputs
    # 增加卷积层和膨胀率，确保感受野 >= sequence_length
    dilations = [1, 2, 4, 8, 16, 32, 64] # 7 layers -> RF=255 > 150

    print("Building TCN model with Residual Blocks, LayerNorm, and Dropout...")

    for i, d in enumerate(dilations):
        # --- 开始一个 TCN 残差块 ---
        prev_x = x # 保存块的输入，用于残差连接

        # 主路径: 卷积 -> 层归一化 -> ReLU激活 -> Dropout
        # 第一层卷积
        x = layers.Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           dilation_rate=d,
                           padding='causal', # 'causal'确保卷积核在计算t时刻输出时，只能看到t及之前的输入
                           name=f'conv1d_dilation_{d}')(x)

        x = layers.LayerNormalization(name=f'layernorm_{d}')(x) # 应用层归一化

        x = activations.relu(x) # 应用ReLU激活函数

        x = layers.Dropout(dropout_rate, name=f'dropout_{d}')(x) # 应用Dropout

        # TCN块通常包含两层卷积，可以取消注释下面这块来添加第二层（可选）
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=d, padding='causal', name=f'conv1d_2_dilation_{d}')(x)
        x = layers.LayerNormalization(name=f'layernorm_2_{d}')(x)
        x = activations.relu(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_2_{d}')(x)

        # 残差路径: 检查维度是否匹配
        if prev_x.shape[-1] != filters:
            # 如果输入通道数与滤波器数不同 (通常发生在第一个块)
            # 使用1x1卷积将输入投影到正确的维度
            residual = layers.Conv1D(filters, 1, padding='same', name=f'residual_projection_{d}')(prev_x)
            print(f"  Block {i+1} (dilation {d}): Added 1x1 Conv for residual connection (Input shape: {prev_x.shape}, Filters: {filters})")
        else:
            # 如果维度相同，直接使用输入
            residual = prev_x
            print(f"  Block {i+1} (dilation {d}): Direct residual connection (Input shape: {prev_x.shape}, Filters: {filters})")

        # 添加残差连接: 将主路径的输出与 (可能被投影的) 输入相加
        x = layers.Add(name=f'residual_add_{d}')([residual, x])
        # --- 结束 TCN 残差块 ---

    # --- TCN层结束，添加最终输出层 ---

    # 使用 1x1 卷积将最终特征维度映射到所需的输出维度 (output_dim * output_steps)
    # activation='linear' 因为是回归任务，不需要压缩输出范围
    x = layers.Conv1D(output_dim * output_steps, 1, activation='linear', name='output_conv1d')(x)

    # causal padding保证了卷积层的最后一个时间步的输出
    # 依赖于输入序列的最后 sequence_length 个点（如果感受野足够大）
    # 我们取这个序列的最后一个时间步的输出来做预测
    # 然后将其塑造成 (batch, output_steps, output_dim)
    if output_steps > 1:
      # 如果需要预测多个步骤，取最后 `output_steps` 个时间点
      x = layers.Lambda(lambda lam_x: lam_x[:, -output_steps:, :], name='output_slice')(x)
    else:
      # 如果只预测一步，取最后一个时间点 (等效于 output_steps=1)
      x = layers.Lambda(lambda lam_x: lam_x[:, -1:, :], name='output_slice')(x)


    output = layers.Reshape((output_steps, output_dim), name='output_reshape')(x) # 重塑形状

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError()) # 保持编译设置
    print("Model build complete.")
    return model

# --- 构建模型 ---
model = build_tcn_model((sequence_length, feature_dim), prediction_steps, label_dim)
model.summary() # 打印模型结构

# --- 定义回调函数 ---
early_stopping = callbacks.EarlyStopping(monitor='val_loss', # 监控验证集损失
                                         patience=10,        # 10轮没改善就停止
                                         restore_best_weights=True) # 恢复最佳权重

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2, # 学习率乘以0.2
                                        patience=5, # 5轮没改善就降低
                                        min_lr=1e-6) # 最小学习率

# --- 训练模型 ---
print("Starting model training...")
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100, # 设置一个较大的数，让EarlyStopping决定何时停止
                    batch_size=64, # 可以调整 batch size
                    callbacks=[early_stopping, reduce_lr]) # 添加回调

# --- 模型评估 ---
print("Evaluating model...")
y_pred = model.predict(X_test)

# 逆标准化处理 (注意 y_pred 和 y_test 的 shape 已经是 (样本数, prediction_steps, label_dim))
# 需要先 reshape 成 2D 进行逆变换，再 reshape 回来
y_pred_inv = scaler_labels.inverse_transform(y_pred.reshape(-1, label_dim))
y_test_inv = scaler_labels.inverse_transform(y_test.reshape(-1, label_dim))

# 计算逆标准化后的指标
mse_inv = mean_squared_error(y_test_inv, y_pred_inv)
mae_inv = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"Test Set Evaluation (Inverse Transformed):")
print(f"  MSE: {mse_inv:.4f}")
print(f"  MAE: {mae_inv:.4f}")

# 如果是多步预测(prediction_steps > 1)，可以分别计算每个标签的指标
if prediction_steps == 1:
    y_pred_inv = y_pred_inv.reshape(-1, label_dim) # (样本数, label_dim)
    y_test_inv = y_test_inv.reshape(-1, label_dim) # (样本数, label_dim)
    for i, label_name in enumerate(labels):
        mse_label = mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])
        mae_label = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
        print(f"  {label_name} - MSE: {mse_label:.4f}, MAE: {mae_label:.4f}")
else:
     # 对于多步预测，可能需要更复杂的评估，比如计算每一步或平均指标
     pass # 暂不展开

# --- 可视化 ---
print("Visualizing predictions...")
# 可视化前 N 个测试样本的预测结果
N = 500 # 要绘制的样本数量
plt.figure(figsize=(15, 6 * label_dim))
for i in range(label_dim):
    plt.subplot(label_dim, 1, i + 1)
    # 因为 prediction_steps = 1, y_test_inv 和 y_pred_inv 都是 (样本数, label_dim)
    # 我们直接绘制这 N 个样本的第 i 个标签
    plt.plot(y_test_inv[:N, i], label=f'Actual {labels[i]}', marker='.', linestyle='-', alpha=0.7)
    plt.plot(y_pred_inv[:N, i], label=f'Predicted {labels[i]}', marker='x', linestyle='--', alpha=0.7)
    plt.title(f"{labels[i]} Prediction vs Actual (First {N} Test Samples)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Value (Original Scale)")
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_comparison.png")) # 保存图像
plt.show()

# 可视化训练历史
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
plt.show()


# --- 保存模型 ---
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

print("Script finished.")