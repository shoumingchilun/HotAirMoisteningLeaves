import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib  # 用于保存/加载标准化器


class TCNPredictor:
    def __init__(self, model_path, scaler_features_path='scaler_features.pkl', scaler_labels_path='scaler_labels.pkl'):
        # 加载预训练模型
        self.model = load_model(model_path, compile=False)

        # 加载标准化器
        self.scaler_features = joblib.load(scaler_features_path)
        self.scaler_labels = joblib.load(scaler_labels_path)

        # 定义模型参数
        self.sequence_length = 150  # 与训练时一致
        self.prediction_steps = 60  # 预测未来两分钟
        self.features = ['material_flow', 'entry_temp', 'entry_moisture',
                         'steam_valve', 'water_addition', 'env_temp', 'env_humidity']
        self.labels = ['out_temp', 'out_moisture']

    def _validate_input(self, input_data):
        """验证输入数据格式"""
        if len(input_data) < self.sequence_length:
            raise ValueError(f"输入数据需要至少包含{self.sequence_length}个时间步的数据")

        if not all(col in input_data.columns for col in self.features):
            missing = set(self.features) - set(input_data.columns)
            raise ValueError(f"输入数据缺少必要特征列: {missing}")

    def predict(self, input_data):
        """
        预测API主函数
        参数：
            input_data: DataFrame，包含最新N个时间步的特征数据（N >= sequence_length）
        返回：
            dict: 包含预测结果的字典，包含温度和湿度的预测序列
        """
        # 数据验证
        self._validate_input(input_data)

        # 截取最后sequence_length个时间步
        recent_data = input_data[self.features].iloc[-self.sequence_length:]

        # 标准化特征数据
        scaled_features = self.scaler_features.transform(recent_data)

        # 转换为模型输入格式 (1, sequence_length, num_features)
        model_input = scaled_features[np.newaxis, ...]

        # 执行预测
        scaled_prediction = self.model.predict(model_input)

        # 逆标准化预测结果
        prediction = self.scaler_labels.inverse_transform(
            scaled_prediction.reshape(-1, 2)).reshape(scaled_prediction.shape)

        # 构造返回结果
        time_steps = np.arange(1, self.prediction_steps + 1)
        return {
            "temperature": {f"t+{t*2}s": float(prediction[0, t - 1, 0]) for t in time_steps},
            "moisture": {f"t+{t*2}s": float(prediction[0, t - 1, 1]) for t in time_steps}
        }


# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 初始化预测器（需要提前保存标准化器）
    predictor = TCNPredictor(
        model_path="temp/optimized_tcn_model.h5",
        scaler_features_path="temp/scaler_features.pkl",
        scaler_labels_path="temp/scaler_labels.pkl"
    )

    # 模拟输入数据（应替换为实际数据）
    sample_data = pd.DataFrame(
        np.random.randn(200, 7),  # 生成200个时间步的测试数据
        columns=predictor.features
    )

    # 执行预测
    predictions = predictor.predict(sample_data)

    # 打印部分预测结果
    print("温度预测:")
    for k, v in list(predictions['temperature'].items())[:]:
        print(f"{k}: {v:.2f}°C")

    print("\n湿度预测:")
    for k, v in list(predictions['moisture'].items())[:]:
        print(f"{k}: {v:.2f}%")