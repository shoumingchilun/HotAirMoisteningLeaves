import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from tensorflow.keras.models import load_model
import joblib  # 用于加载标准化器

# 创建 FastAPI 实例
app = FastAPI(title="TCN Predictor API", version="1.0")

# 启动命令：
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

class TCNPredictor:
    def __init__(self, model_path, scaler_features_path, scaler_labels_path):
        # 加载模型和标准化器
        self.model = load_model(model_path, compile=False)
        self.scaler_features = joblib.load(scaler_features_path)
        self.scaler_labels = joblib.load(scaler_labels_path)

        # 设定超参数
        self.sequence_length = 150
        self.prediction_steps = 60
        self.features = ['material_flow', 'entry_temp', 'entry_moisture',
                         'steam_valve', 'water_addition', 'env_temp', 'env_humidity']
        self.labels = ['out_temp', 'out_moisture']

    def _validate_input(self, input_data):
        """验证输入数据格式"""
        if len(input_data) < self.sequence_length:
            raise ValueError(f"输入数据至少需要{self.sequence_length}个时间步")

        if not all(col in input_data.columns for col in self.features):
            missing = set(self.features) - set(input_data.columns)
            raise ValueError(f"输入数据缺少必要特征列: {missing}")

    def predict(self, input_data):
        """执行预测"""
        self._validate_input(input_data)

        # 取最后 sequence_length 个时间步的数据
        recent_data = input_data[self.features].iloc[-self.sequence_length:]

        # 标准化特征数据
        scaled_features = self.scaler_features.transform(recent_data)

        # 转换为模型输入格式
        model_input = scaled_features[np.newaxis, ...]

        # 预测
        scaled_prediction = self.model.predict(model_input)

        # 逆标准化
        prediction = self.scaler_labels.inverse_transform(
            scaled_prediction.reshape(-1, 2)).reshape(scaled_prediction.shape)

        # 组装返回结果
        time_steps = np.arange(1, self.prediction_steps + 1)
        return {
            "temperature": {f"t+{t*2}s": float(prediction[0, t - 1, 0]) for t in time_steps},
            "moisture": {f"t+{t*2}s": float(prediction[0, t - 1, 1]) for t in time_steps}
        }


# 加载预测器
predictor = TCNPredictor(
    model_path="optimized_tcn_model.h5",
    scaler_features_path="scaler_features.pkl",
    scaler_labels_path="scaler_labels.pkl"
)


# 定义请求体格式
class PredictionRequest(BaseModel):
    input_data: List[Dict[str, float]]  # JSON 数组，每个元素是包含 7 个特征的字典


@app.post("/predict", summary="预测温度和湿度")
async def predict_temperature_and_moisture(request: PredictionRequest):
    try:
        # 将 JSON 转换为 DataFrame
        input_df = pd.DataFrame(request.input_data)

        # 进行预测
        prediction_result = predictor.predict(input_df)

        return {"status": "success", "predictions": prediction_result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


# 启动命令：
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
