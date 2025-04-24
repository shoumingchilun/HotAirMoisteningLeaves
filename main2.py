import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
from tensorflow.keras.models import load_model
import joblib
import os
import logging
import time # To measure optimization time

# --- Configuration (Ensure these match your best model) ---
# !!! 重要: 确保这些路径指向您获得最佳结果 (MSE: 36.93) 的那次训练所保存的文件 !!!
# !!! (那次训练使用了 sequence_length=300) !!!
MODEL_PATH = "output2/Gas_model.h5"  # 假设最佳模型保存在 output2
SCALER_FEATURES_PATH = "output2/scaler_features.pkl"
SCALER_LABELS_PATH = "output2/scaler_labels.pkl"
SEQUENCE_LENGTH = 300 # 必须与训练最佳模型时使用的长度一致
PREDICTION_STEPS = 1 # 模型设计为预测下一步
FEATURES = ['circulationFanFreq', 'inletMoisture', 'tobaccoFlow', 'steamValveOpening', 'envHumidity', 'hotAirTemp', 'dryHeadWeight', 'steamPressure', 'exhaustFanFreq']
OPTIMIZABLE_FEATURES = ['hotAirTemp', 'steamPressure', 'exhaustFanFreq', 'circulationFanFreq', 'steamValveOpening'] # Features PSO will optimize
LABELS = ['actualHotAirTemp', 'outletMoistureFeedback']
FEATURE_DIM = len(FEATURES)
LABEL_DIM = len(LABELS)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="热风润叶 Predictor & Optimizer API",
    description="使用TCN模型预测并使用PSO优化热风润叶机输入参数以达到目标输出",
    version="1.1.0"
)

# --- Predictor Class (Combined Prediction and PSO) ---
class GasTurbinePredictor:
    def __init__(self, model_path, scaler_features_path, scaler_labels_path, sequence_length, features, labels, optimizable_features):
        logger.info(f"Initializing predictor...")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(scaler_features_path):
            logger.error(f"Feature scaler file not found at: {scaler_features_path}")
            raise FileNotFoundError(f"Feature scaler file not found at: {scaler_features_path}")
        if not os.path.exists(scaler_labels_path):
            logger.error(f"Label scaler file not found at: {scaler_labels_path}")
            raise FileNotFoundError(f"Label scaler file not found at: {scaler_labels_path}")

        try:
            # Load model, compile=False is usually faster and sufficient for prediction
            self.model = load_model(model_path, compile=False)
            logger.info(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}", exc_info=True)
            raise RuntimeError(f"Error loading Keras model: {e}")
        try:
            self.scaler_features = joblib.load(scaler_features_path)
            self.scaler_labels = joblib.load(scaler_labels_path)
            logger.info(f"Scalers loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading scalers: {e}", exc_info=True)
            raise RuntimeError(f"Error loading scalers: {e}")

        self.sequence_length = sequence_length
        self.features = features
        self.labels = labels
        self.optimizable_features = optimizable_features # Added for PSO
        self.feature_dim = len(features)
        self.label_dim = len(labels)
        self.optimizable_dim = len(optimizable_features) # Added for PSO
        logger.info(f"Predictor initialized with sequence_length={self.sequence_length}")

    # --- Input Validation (Generalized) ---
    def _validate_input(self, input_data: pd.DataFrame, min_length: int):
        """通用验证函数"""
        logger.debug(f"Validating input data (min_length={min_length})...")
        if len(input_data) < min_length:
            msg = f"Input data must contain at least {min_length} time steps. Got {len(input_data)}."
            logger.warning(msg)
            raise ValueError(msg)

        missing_features = set(self.features) - set(input_data.columns)
        if missing_features:
            msg = f"Input data is missing required feature columns: {missing_features}"
            logger.warning(msg)
            raise ValueError(msg)
        logger.debug("Input data validation successful.")

    # --- Prediction Method (for public /predict endpoint) ---
    def predict(self, input_data: pd.DataFrame) -> Dict[str, float]:
        """
        执行单步预测 (CO 和 NOx) - for public endpoint.
        参数:
            input_data: DataFrame，包含至少 sequence_length 个时间步的特征数据.
        返回:
            dict: 包含下一个时间步预测的 CO 和 NOx 值.
        """
        logger.info(f"Received prediction request with input data shape: {input_data.shape}")
        # 1. Validate input for standard prediction
        self._validate_input(input_data, self.sequence_length)

        # 2. Select features and recent sequence
        recent_data = input_data[self.features].iloc[-self.sequence_length:].copy()
        logger.debug(f"Selected recent data for prediction shape: {recent_data.shape}")

        # 3. Scale features
        try:
            scaled_features = self.scaler_features.transform(recent_data)
            logger.debug("Features scaled successfully for prediction.")
        except Exception as e:
            logger.error(f"Error scaling features for prediction: {e}", exc_info=True)
            raise RuntimeError(f"Error during feature scaling: {e}")

        # 4. Reshape for model input
        model_input = scaled_features[np.newaxis, ...]
        logger.debug(f"Model input shape for prediction: {model_input.shape}")

        # 5. Perform prediction
        try:
            # Use verbose=0 to avoid progress bars in logs for single predictions too
            scaled_prediction = self.model.predict(model_input, verbose=0)
            logger.debug(f"Raw model prediction (scaled) shape: {scaled_prediction.shape}")
        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            raise RuntimeError(f"Error during model prediction: {e}")

        # 6. Inverse transform labels
        try:
            prediction_inv = self.scaler_labels.inverse_transform(scaled_prediction.reshape(1, self.label_dim))
            logger.debug(f"Inverse transformed prediction shape: {prediction_inv.shape}")
        except Exception as e:
            logger.error(f"Error inverse transforming prediction: {e}", exc_info=True)
            raise RuntimeError(f"Error during inverse scaling: {e}")

        # 7. Format result
        result = {
            self.labels[0]: float(prediction_inv[0, 0]), # actualHotAirTemp
            self.labels[1]: float(prediction_inv[0, 1])  # outletMoistureFeedback
        }
        logger.info(f"Prediction successful: {result}")
        return result

    # --- Internal Prediction Helper (for PSO fitness) ---
    def _internal_predict(self, input_sequence_df: pd.DataFrame) -> Dict[str, float]:
        """
        Internal prediction using a pre-constructed sequence.
        Assumes input_sequence_df has exactly sequence_length rows and correct features.
        Used within the PSO fitness function.
        """
        # 1. Scale features
        try:
            scaled_features = self.scaler_features.transform(input_sequence_df[self.features])
        except Exception as e:
            logger.error(f"Error scaling features in _internal_predict: {e}", exc_info=True)
            raise RuntimeError(f"Error during feature scaling: {e}")

        # 2. Reshape for model input
        model_input = scaled_features[np.newaxis, ...]

        # 3. Perform prediction
        try:
            scaled_prediction = self.model.predict(model_input, verbose=0) # verbose=0 essential for PSO
        except Exception as e:
            logger.error(f"Error during internal model prediction: {e}", exc_info=True)
            raise RuntimeError(f"Error during internal model prediction: {e}")

        # 4. Inverse transform labels
        try:
            prediction_inv = self.scaler_labels.inverse_transform(scaled_prediction.reshape(1, self.label_dim))
        except Exception as e:
            logger.error(f"Error inverse transforming internal prediction: {e}", exc_info=True)
            raise RuntimeError(f"Error during internal inverse scaling: {e}")

        # 5. Format result
        result = {
            self.labels[0]: float(prediction_inv[0, 0]), # actualHotAirTemp
            self.labels[1]: float(prediction_inv[0, 1])  # outletMoistureFeedback
        }
        return result

    # --- PSO Helper: Build Input Sequence ---
    def _build_pso_input_sequence(self, base_history_df: pd.DataFrame, future_params: np.ndarray, horizon: int) -> pd.DataFrame:
        """Constructs the 300-step input for TCN during PSO fitness evaluation."""
        if len(base_history_df) != self.sequence_length - horizon:
             raise ValueError(f"Base history length ({len(base_history_df)}) is incorrect for horizon {horizon}.")

        last_actual_row = base_history_df.iloc[[-1]]
        future_steps_df = pd.concat([last_actual_row] * horizon, ignore_index=True)

        for i, feature_name in enumerate(self.optimizable_features):
            future_steps_df[feature_name] = future_params[i]

        full_sequence_df = pd.concat([base_history_df, future_steps_df], ignore_index=True)
        return full_sequence_df

    # --- PSO Implementation ---
    def run_pso(self,
                history_data: pd.DataFrame,
                wishedActualHotAirTemp: float,
                wishedOutletMoistureFeedback: float,
                horizon: int,
                bounds: np.ndarray,
                num_particles: int,
                max_iter: int,
                inertia_weight: float,
                cognitive_weight: float,
                social_weight: float,
                co_loss_weight: float = 1.0,
                nox_loss_weight: float = 1.0
                ):
        """Runs the PSO algorithm."""
        start_time = time.time()
        logger.info(f"Starting PSO optimization: wished_actualHotAirTemp={wishedActualHotAirTemp}, wished_outletMoistureFeedback={wishedOutletMoistureFeedback}, horizon={horizon}")
        # ... (rest of PSO logging) ...
        logger.info(f"PSO Params: particles={num_particles}, max_iter={max_iter}, w={inertia_weight}, c1={cognitive_weight}, c2={social_weight}")
        logger.info(f"Parameter bounds:\n{pd.DataFrame(bounds, index=self.optimizable_features, columns=['min', 'max'])}")


        # 1. Prepare base historical data
        n = self.sequence_length - horizon
        if n <= 0:
            raise ValueError(f"Horizon ({horizon}) must be less than sequence length ({self.sequence_length}).")
        # Validate the full history (need context for validation, even if only using n steps)
        self._validate_input(history_data, self.sequence_length)
        base_history = history_data.iloc[-n:].copy()
        logger.debug(f"Base history for PSO input construction shape: {base_history.shape}")

        # 2. Fitness Function
        def fitness_function(particle_params: np.ndarray) -> float:
            try:
                input_sequence = self._build_pso_input_sequence(base_history, particle_params, horizon)
                # Use the internal prediction helper
                predictions = self._internal_predict(input_sequence)
                pred_co = predictions[self.labels[0]]
                pred_nox = predictions[self.labels[1]]

                error = (co_loss_weight * np.square(pred_co - wishedActualHotAirTemp) +
                         nox_loss_weight * np.square(pred_nox - wishedOutletMoistureFeedback))
                return error if np.isfinite(error) else np.inf
            except Exception as e:
                logger.error(f"Error in fitness evaluation for particle {particle_params}: {e}", exc_info=False)
                return np.inf

        # 3. Initialize PSO
        particles_pos = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_particles, self.optimizable_dim))
        particles_vel = np.zeros_like(particles_pos)
        personal_best_pos = particles_pos.copy()
        personal_best_scores = np.full(num_particles, np.inf)
        for i in range(num_particles):
             personal_best_scores[i] = fitness_function(personal_best_pos[i])

        if not np.any(np.isfinite(personal_best_scores)):
             logger.error("PSO Initialization failed: All initial particles resulted in infinite fitness.")
             raise RuntimeError("PSO Initialization failed: Could not evaluate initial particles.")

        global_best_idx = np.nanargmin(personal_best_scores)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        logger.info(f"PSO Initial global best score: {global_best_score:.4f}")


        # 4. PSO Iteration Loop
        for iter_num in range(max_iter):
            iter_start_time = time.time()
            for i in range(num_particles):
                # Update velocity & position (same as before)
                r1, r2 = np.random.rand(2)
                particles_vel[i] = (inertia_weight * particles_vel[i] +
                                    cognitive_weight * r1 * (personal_best_pos[i] - particles_pos[i]) +
                                    social_weight * r2 * (global_best_pos - particles_pos[i]))
                particles_pos[i] = particles_pos[i] + particles_vel[i]
                particles_pos[i] = np.clip(particles_pos[i], bounds[:, 0], bounds[:, 1])

                current_score = fitness_function(particles_pos[i])

                # Update personal & global best (same as before)
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_pos[i] = particles_pos[i].copy()
                    if current_score < global_best_score:
                        global_best_score = current_score
                        global_best_pos = particles_pos[i].copy()

            logger.debug(f"PSO Iter {iter_num+1}/{max_iter} | Current Global Best Score: {global_best_score:.4f} | Time: {time.time() - iter_start_time:.2f}s")


        end_time = time.time()
        logger.info(f"PSO Finished. Final Score: {global_best_score:.4f}. Time Elapsed: {end_time - start_time:.2f}s")

        # 5. Prepare and return results
        optimized_params_dict = {name: float(val) for name, val in zip(self.optimizable_features, global_best_pos)}

        try:
            final_input_sequence = self._build_pso_input_sequence(base_history, global_best_pos, horizon)
            final_predictions = self._internal_predict(final_input_sequence)
        except Exception as e:
             logger.error(f"Error predicting with final optimized parameters: {e}", exc_info=True)
             final_predictions = {self.labels[0]: None, self.labels[1]: None} # Indicate prediction failed


        return optimized_params_dict
        #     {
        #     "optimized_parameters": optimized_params_dict,
        #     "final_predicted_co": final_predictions[self.labels[0]],
        #     "final_predicted_nox": final_predictions[self.labels[1]],
        #     "final_fitness_score": float(global_best_score),
        #     "optimization_time_seconds": round(end_time - start_time, 2)
        # }


# --- Instantiate Predictor ---
try:
    predictor = GasTurbinePredictor(
        model_path=MODEL_PATH,
        scaler_features_path=SCALER_FEATURES_PATH,
        scaler_labels_path=SCALER_LABELS_PATH,
        sequence_length=SEQUENCE_LENGTH,
        features=FEATURES,
        labels=LABELS,
        optimizable_features=OPTIMIZABLE_FEATURES # Pass optimizable features
    )
except (FileNotFoundError, RuntimeError) as e:
    logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
    predictor = None

# --- Pydantic Models ---
class InputDataPoint(BaseModel): # Keep as is
    circulationFanFreq: float = Field(..., example=15.0)
    inletMoisture: float = Field(..., example=1013.0)
    tobaccoFlow: float = Field(..., example=80.0)
    steamValveOpening: float = Field(..., example=3.5)
    envHumidity: float = Field(..., example=25.0)
    hotAirTemp: float = Field(..., example=1050.0)
    dryHeadWeight: float = Field(..., example=550.0)
    steamPressure: float = Field(..., example=110.0)
    exhaustFanFreq: float = Field(..., example=12.0)

class PredictionRequest(BaseModel): # For /predict endpoint
    input_data: List[InputDataPoint] = Field(..., description=f"包含至少 {SEQUENCE_LENGTH} 个时间步的特征数据列表")

# Models for /optimize endpoint (Keep PsoParams, ParameterBounds, OptimizationRequest as in the previous PSO version)
class PsoParams(BaseModel):
    wishedActualHotAirTemp: float = Field(..., example=2.5)
    wishedOutletMoistureFeedback: float = Field(..., example=70.0)
    numParticles: int = Field(20, gt=0, description="粒子群大小")
    maxIter: int = Field(50, gt=0, description="最大迭代次数")
    inertiaWeight: float = Field(0.5, ge=0, description="惯性权重 (w)")
    cognitiveWeight: float = Field(0.8, ge=0, description="认知权重 (c1)")
    socialWeight: float = Field(0.8, ge=0, description="社会权重 (c2)")

class ParameterBounds(BaseModel):
    hotAirTemp: Tuple[float, float] = Field((1000, 1100), description="涡轮进口温度 (min, max)")
    steamPressure: Tuple[float, float] = Field((100, 150), description="涡轮能量输出 (min, max)")
    exhaustFanFreq: Tuple[float, float] = Field((10, 15), description="压缩机出口压力 (min, max)")
    circulationFanFreq: Tuple[float, float] = Field((0, 30), description="环境温度 (min, max) - Note: Adjust if not controllable")
    steamValveOpening: Tuple[float, float] = Field((2, 5), description="空滤压差 (min, max) - Note: Adjust if not state/controllable")

class OptimizationRequest(BaseModel):
    history_data: List[InputDataPoint] = Field(..., description=f"至少包含 {SEQUENCE_LENGTH} 个时间步的历史特征数据")
    optimization_horizon: int = Field(10, gt=0, lt=SEQUENCE_LENGTH, description=f"假设优化参数保持不变的未来时间步数 H (1 <= H < {SEQUENCE_LENGTH})")
    hyper_param: PsoParams = Field(default_factory=PsoParams, description="PSO 算法超参数")
    parameter_bounds: ParameterBounds = Field(default_factory=ParameterBounds, description="待优化参数的边界")
    co_loss_weight: float = Field(1.0, ge=0, description="计算适应度时CO误差的权重")
    nox_loss_weight: float = Field(1.0, ge=0, description="计算适应度时NOx误差的权重")


# --- API Endpoints ---

# --- /predict Endpoint (Restored from simpler version) ---
@app.post("/predict",
          summary="预测下一时间步的CO和NOx排放",
          response_description="预测得到的CO和NOx值")
async def predict_emissions(request: PredictionRequest):
    """
    接收最近一段时间（至少包含 `sequence_length` 个时间步）的燃气轮机传感器数据，
    预测下一个时间步的 CO 和 NOx 排放量。

    - **input_data**: 一个包含字典的列表，每个字典代表一个时间步的数据，
      需要包含所有必需特征。列表长度必须大于等于 `sequence_length`。
    """
    if predictor is None:
        logger.error("Predictor not initialized. Cannot process /predict request.")
        raise HTTPException(status_code=503, detail="服务暂时不可用：预测器未能初始化")

    try:
        # 1. Convert request data to DataFrame
        if not request.input_data or not isinstance(request.input_data, list) or not all(isinstance(item.dict() if isinstance(item, BaseModel) else item, dict) for item in request.input_data):
             raise ValueError("Invalid input_data format for /predict. Expected a list of dictionaries.")
        # Use item.dict() if list contains Pydantic models
        input_df = pd.DataFrame([item.dict() for item in request.input_data])
        logger.info(f"/predict endpoint: DataFrame created, shape: {input_df.shape}")

        # 2. Call the public predict method
        prediction_result = predictor.predict(input_df) # Calls the method handling slicing, scaling etc.

        # 3. Return success response
        return prediction_result

    except ValueError as e:
        logger.warning(f"Bad Request (400) for /predict: {e}", exc_info=False) # Less verbose logging for common errors
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime Error (500) during /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测过程中发生内部错误: {e}")
    except Exception as e:
        logger.error(f"Unexpected Server Error (500) during /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器发生意外错误: {str(e)}")


# --- /optimize Endpoint (Kept from PSO version) ---
@app.post("/optimize",
          summary="使用PSO优化输入参数以达到目标CO/NOx排放",
          response_description="优化后的参数及预期排放")
async def optimize_parameters_endpoint(request: OptimizationRequest = Body(...)):
    """
    接收历史传感器数据和目标 CO/NOx 值，使用 PSO 算法寻找一组未来输入参数
    (`TIT`, `TEY`, `CDP`, `AT`, `AFDP`)，使得 TCN 模型预测的排放尽可能接近目标值。
    """
    if predictor is None:
        logger.error("Predictor not initialized. Cannot process /optimize request.")
        raise HTTPException(status_code=503, detail="服务暂时不可用：预测器未能初始化")

    try:
        # 1. Convert history data to DataFrame
        if not request.history_data or not isinstance(request.history_data, list) or not all(isinstance(item.dict() if isinstance(item, BaseModel) else item, dict) for item in request.history_data):
             raise ValueError("Invalid history_data format for /optimize. Expected a list of dictionaries.")
        history_df = pd.DataFrame([item.dict() for item in request.history_data])
        logger.info(f"/optimize endpoint: DataFrame created, shape: {history_df.shape}")

        # 2. Prepare bounds
        bounds_list = [
            request.parameter_bounds.hotAirTemp,
            request.parameter_bounds.steamPressure,
            request.parameter_bounds.exhaustFanFreq,
            request.parameter_bounds.circulationFanFreq,
            request.parameter_bounds.steamValveOpening
        ]
        bounds_array = np.array(bounds_list)
        if bounds_array.shape != (predictor.optimizable_dim, 2):
            raise ValueError("Mismatch between optimizable features and provided bounds dimensions.")

        # 3. Run PSO
        optimization_result = predictor.run_pso(
            history_data=history_df,
            wishedActualHotAirTemp=request.hyper_param.wishedActualHotAirTemp,
            wishedOutletMoistureFeedback=request.hyper_param.wishedOutletMoistureFeedback,
            horizon=request.optimization_horizon,
            bounds=bounds_array,
            num_particles=request.hyper_param.numParticles,
            max_iter=request.hyper_param.maxIter,
            inertia_weight=request.hyper_param.inertiaWeight,
            cognitive_weight=request.hyper_param.cognitiveWeight,
            social_weight=request.hyper_param.socialWeight,
            co_loss_weight=request.co_loss_weight,
            nox_loss_weight=request.nox_loss_weight
        )

        # 4. Return success response
        return optimization_result
        # return {"optimization_details": optimization_result}

    except ValueError as e:
        logger.warning(f"Bad Request (400) for /optimize: {e}", exc_info=False)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime Error (500) during /optimize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"优化过程中发生内部错误: {e}")
    except Exception as e:
        logger.error(f"Unexpected Server Error (500) during /optimize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器发生意外错误: {str(e)}")

# --- Startup Command ---
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload