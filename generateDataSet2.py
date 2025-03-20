import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from scipy.ndimage import uniform_filter1d

# 设置随机种子，确保结果可复现
# np.random.seed(42)

# 采样间隔 2 秒，总时长 2 小时（7200 秒），共 3600 组数据
sampling_interval = 2  # 2 秒
num_samples = 3600  # 3600 组数据
time_series = np.arange(0, num_samples * sampling_interval, sampling_interval)

# 1. 模拟物料流量，包括料头多、料尾少的现象
material_flow = np.full(num_samples, 4500) + np.random.normal(0, 10, num_samples)
# 料尾部分（最后 20 分钟）逐渐减少
material_flow[-600:] = np.linspace(4500, 4000, 600) + np.random.normal(0, 10, 600)

# 2. 入口参数：入口温度、入口水分、环境温度、环境湿度
entry_temp = np.full(num_samples, 30) + np.random.normal(0, 0.1, num_samples)
entry_moisture = np.full(num_samples, 12) + np.random.normal(0, 0.05, num_samples)

# 3. 增加多种环境变量（不同时间段不同环境）
# env_temp_variants = [20, 25, 30]  # 低温、常温、高温
# env_humidity_variants = [50, 60, 70]  # 低湿度、常湿度、高湿度
# # 每 30 分钟（900 组数据）切换一次环境
# env_temp = np.tile(np.random.choice(env_temp_variants, 4), 900)[:num_samples] + np.random.normal(0, 0.2, num_samples)
# env_humidity = np.tile(np.random.choice(env_humidity_variants, 4), 900)[:num_samples] + np.random.normal(0, 0.5, num_samples)

# 环境温度
env_temp = np.full(num_samples, 25) + np.random.normal(0, 0.2, num_samples)  # 25 ℃
# 环境湿度
env_humidity = np.full(num_samples, 60) + np.random.normal(0, 0.5, num_samples)  # 60%

# 4. 设备可变参数：蒸汽阀门开度、水添加量
steam_valve = np.full(num_samples, 50, dtype=np.float64)
water_addition = np.full(num_samples, 20, dtype=np.float64)

# 模拟人工调整，每 20 分钟调整一次
for i in range(300, num_samples, 600):
    steam_valve[i:] += np.random.uniform(-5, 5)
    water_addition[i:] += np.random.uniform(-2, 2)

# 5. 模拟出口参数：温度和湿度，受 5 分钟内所有参数影响
out_temp = np.zeros(num_samples)
out_moisture = np.zeros(num_samples)

# 5 分钟（150 组数据）的窗口计算出口参数
for i in range(150, num_samples):
    past_5min_steam = uniform_filter1d(steam_valve[i - 150:i], size=150)[-1]  # 过去 5 分钟的蒸汽开度均值
    past_5min_water = uniform_filter1d(water_addition[i - 150:i], size=150)[-1]  # 过去 5 分钟的加水量均值
    past_5min_entry_temp = uniform_filter1d(entry_temp[i - 150:i], size=150)[-1]  # 入口温度均值
    past_5min_entry_moisture = uniform_filter1d(entry_moisture[i - 150:i], size=150)[-1]  # 入口水分均值

    out_temp[i] = (
            past_5min_entry_temp + 0.1 * past_5min_steam - 0.05 * env_temp[i] + np.random.normal(0, 0.5)
    )
    out_moisture[i] = (
            past_5min_entry_moisture + 0.2 * past_5min_water - 0.1 * env_humidity[i] + np.random.normal(0, 0.3)
    )

# 6. 料头部分（前 5 分钟）出口温湿度波动较大
out_temp[:150] = np.random.uniform(20, 80, 150)
out_moisture[:150] = np.random.uniform(5, 20, 150)

# 组装数据
data = pd.DataFrame({
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

# 保存数据为 CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_uuid = uuid.uuid4().hex[:8]
filename = f"dataset/simulated_tcn_data_{timestamp}_{file_uuid}.csv"
data.to_csv(filename, index=False)

print(f"File saved as: {filename}")
print(data.head())
