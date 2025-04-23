import numpy as np
import pandas as pd
from copy import deepcopy
from tensorflow.keras.models import load_model
import joblib  # 用于加载标准化器

import numpy as np
import pandas as pd
from copy import deepcopy


class TCNPredictor:
    def __init__(self, model_path, scaler_features_path, scaler_labels_path):
        # 初始化代码与原实现保持一致
        self.model = load_model(model_path, compile=False)
        self.scaler_features = joblib.load(scaler_features_path)
        self.scaler_labels = joblib.load(scaler_labels_path)

        self.sequence_length = 150
        self.prediction_steps = 60
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

    def _prepare_input(self, input_data):
        """将140时间步输入扩展为150时间步"""
        # 取最后140个时间步
        base_data = input_data[self.features].iloc[-140:]
        # 复制最后一行10次
        last_row = base_data.iloc[[-1]]
        extended_data = pd.concat([last_row] * 10, ignore_index=True)
        return pd.concat([base_data, extended_data], ignore_index=True)

    def optimize_parameters(self,
                            input_data,
                            target_temp: float,
                            target_humidity: float,
                            steam_bounds=(0, 100),
                            water_bounds=(0, 50),
                            num_particles=20,
                            max_iter=50,
                            inertia_weight=0.5,
                            cognitive_weight=0.8,
                            social_weight=0.8):
        """
        改进版PSO优化API
        参数变化：
        - 输入数据长度改为140时间步
        - 自动扩展最后10个时间步（使用第140时间步数据）
        """
        # 输入验证
        if len(input_data) < 140:
            raise ValueError("输入数据需要至少140个时间步")
        if not all(col in input_data.columns for col in self.features):
            missing = set(self.features) - set(input_data.columns)
            raise ValueError(f"缺少特征列: {missing}")

        # 准备基础数据（140+10）
        base_data = self._prepare_input(input_data)
        bounds = np.array([steam_bounds, water_bounds])

        def fitness(particle):
            """适应度函数优化"""
            sv, wa = particle
            modified_data = base_data.copy()

            # 修改扩展部分的参数（第141-150时间步）
            modified_data.loc[140:, 'steam_valve'] = sv  # 索引140-149对应第141-150时间步
            modified_data.loc[140:, 'water_addition'] = wa

            try:
                # 使用完整150时间步数据进行预测
                predictions = self.predict(modified_data)
            except Exception as e:
                print(f"预测异常: {str(e)}")
                return np.inf

            # 计算复合误差
            temp_vals = list(predictions['temperature'].values())
            hum_vals = list(predictions['moisture'].values())
            return (np.mean(np.square(np.array(temp_vals) - target_temp)) +
                    np.mean(np.square(np.array(hum_vals) - target_humidity)))

        # PSO算法核心（保持原结构）
        particles = np.random.uniform(
            low=[bounds[0][0], bounds[1][0]],
            high=[bounds[0][1], bounds[1][1]],
            size=(num_particles, 2)
        )
        velocities = np.zeros_like(particles)

        personal_best = particles.copy()
        personal_scores = np.array([fitness(p) for p in particles])
        global_idx = np.argmin(personal_scores)
        global_best = particles[global_idx]
        global_score = personal_scores[global_idx]

        # 迭代优化
        for _ in range(max_iter):
            for i in range(num_particles):
                # 速度更新
                r1, r2 = np.random.rand(2)
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_weight * r1 * (personal_best[i] - particles[i]) +
                                 social_weight * r2 * (global_best - particles[i]))

                # 位置更新
                new_pos = np.clip(particles[i] + velocities[i], bounds[:, 0], bounds[:, 1])
                new_score = fitness(new_pos)

                # 更新记录
                if new_score < personal_scores[i]:
                    personal_best[i] = new_pos
                    personal_scores[i] = new_score
                    if new_score < global_score:
                        global_best = new_pos
                        global_score = new_score
                particles[i] = new_pos

        return {
            'steam_valve': global_best[0],
            'water_addition': global_best[1],
            'score': global_score
        }


# 使用示例
predictor = TCNPredictor("temp/optimized_tcn_model.h5", "temp/scaler_features.pkl", "temp/scaler_labels.pkl")
input_140 = pd.read_csv("temp/input_data.csv")  # 140时间步数据

result = predictor.optimize_parameters(
    input_data=input_140,
    target_temp=35.0,
    target_humidity=20.0,
    steam_bounds=(30, 70),  # 更合理的参数范围
    water_bounds=(10, 40),
    num_particles=30,
    max_iter=100
)

print(result)