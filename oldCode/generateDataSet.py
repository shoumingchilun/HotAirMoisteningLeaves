import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from scipy.signal import lfilter

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 采样间隔 2 秒，总时长 2 小时（7200 秒），共 3600 组数据
sampling_interval = 2  # 2 秒
num_samples = 3600  # 总共 3600 组数据

time_series = np.arange(0, num_samples * sampling_interval, sampling_interval)

# 生成相对稳定的输入变量（1、2、3、6项数据）
# 物料流量
material_flow = np.full(num_samples, 4500) + np.random.normal(0, 10, num_samples)  # 4500 kg/h 稳定
# 入口温度
entry_temp = np.full(num_samples, 30) + np.random.normal(0, 0.1, num_samples)  # 30 ℃
# 入口水分
entry_moisture = np.full(num_samples, 12) + np.random.normal(0, 0.05, num_samples)  # 12%
# 环境温度
env_temp = np.full(num_samples, 25) + np.random.normal(0, 0.2, num_samples)  # 25 ℃
# 环境湿度
env_humidity = np.full(num_samples, 60) + np.random.normal(0, 0.5, num_samples)  # 60%

# 设备可变参数（4、5项数据）
# 设备蒸汽阀门开度、设备加水量
steam_valve = np.full(num_samples, 50, dtype=np.float64)  # 初始 50%
water_addition = np.full(num_samples, 20, dtype=np.float64)  # 初始 20kg

# 模拟人工调整
for i in range(300, num_samples, 600):  # 每 20 分钟调整一次
    steam_valve[i:] += np.random.uniform(-5, 5)
    water_addition[i:] += np.random.uniform(-2, 2)

# 输出变量（7、8项数据），受 steam_valve 和 water_addition 影响，滞后 5 分钟（150 组数据）
out_temp = np.zeros(num_samples)
out_moisture = np.zeros(num_samples)

# 设定物理影响公式
for i in range(150, num_samples):  # 5 分钟（150 组数据）后才受影响
    out_temp[i] = (
            entry_temp[i] + 0.1 * steam_valve[i - 150] - 0.05 * env_temp[i] + np.random.normal(0, 0.5)
    )
    out_moisture[i] = (
            entry_moisture[i] + 0.2 * water_addition[i - 150] - 0.1 * env_humidity[i] + np.random.normal(0, 0.3)
    )

# 前 150 组数据（5 分钟）波动极大，设置为随机噪声
out_temp[:150] = np.random.uniform(20, 80, 150)
out_moisture[:150] = np.random.uniform(5, 20, 150)

# 组装数据
simulated_data = pd.DataFrame({
    'time': time_series,
    'material_flow': material_flow,
    'entry_temp': entry_temp,
    'entry_moisture': entry_moisture,
    'steam_valve': steam_valve,
    'water_addition': water_addition,
    'env_temp': env_temp,
    'env_humidity': env_humidity,
    'out_temp': out_temp,
    'out_moisture': out_moisture
})

# 保存为 CSV
# 生成带时间和 UUID 的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_uuid = uuid.uuid4().hex[:8]
filename = f"simulated_tcn_data_{timestamp}_{file_uuid}.csv"

# 保存为 CSV
simulated_data.to_csv(filename, index=False)
print(f"File saved as: {filename}")
print(simulated_data.head())
