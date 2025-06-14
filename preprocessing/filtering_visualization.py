import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt

plt.rcParams['font.sans-serif']=['Hiragino Sans GB'] # 修改字体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号


def quaternion_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def rotate_vectors_by_quaternions(vectors, quaternions):
    """向量化四元数旋转"""
    # 将向量转换为纯四元数 (0, vx, vy, vz)
    v_quat = np.hstack([np.zeros((len(vectors), 1)), vectors])
    
    # 计算四元数的共轭 (逆)
    q_conj = quaternions * np.array([1, -1, -1, -1])
    
    # 执行旋转: v' = q ⊗ v ⊗ q*
    temp = np.array([quaternion_multiply(q, v) for q, v in zip(quaternions, v_quat)])
    rotated = np.array([quaternion_multiply(t, qc) for t, qc in zip(temp, q_conj)])
    
    # 返回旋转后的向量部分
    return rotated[:, 1:]

def rotate_vectors_back_to_body(vectors, quaternions):
    """将世界坐标系向量旋转回传感器本体坐标系"""
    # 将向量转换为纯四元数 (0, vx, vy, vz)
    v_quat = np.hstack([np.zeros((len(vectors), 1)), vectors])
    
    # 使用四元数的共轭（逆旋转）
    q_conj = quaternions * np.array([1, -1, -1, -1])
    
    # 执行逆旋转: v' = q* ⊗ v ⊗ q
    temp = np.array([quaternion_multiply(qc, v) for qc, v in zip(q_conj, v_quat)])
    rotated = np.array([quaternion_multiply(t, q) for t, q in zip(temp, quaternions)])
    
    return rotated[:, 1:]

def apply_lowpass_filter(data, cutoff=15, fs=200, order=4):
    """应用Butterworth低通滤波器
    参数建议：
    - cutoff: 截止频率(Hz)，建议范围：
      * 滑雪运动：8-15Hz (保留快速运动特征)
      * 步行检测：2-5Hz (滤除高频噪声)
      * 姿态估计：4-10Hz (平衡噪声与动态响应)
    - fs: 采样频率(Hz)，需与实际采样率一致
      * 常见值：50/100/200Hz (查看设备规格)
    - order: 滤波器阶数，建议4-6阶
      * 阶数越高衰减越陡峭，但相位延迟越大
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False) #type: ignore
    return filtfilt(b, a, data, axis=0)

def plot_sensor_data(file_path,data,name):
    # 创建对比图表
    plt.figure(figsize=(15, 10))
    # 图表标题
    title=file_path.split("-")[1].split(".")[0]
    print(title)
    plt.title(f'{name}_{title}')
    time_data = data['时间'].str.split(' ').str[1]  # 提取"时:分:秒:毫秒"部分

    # 1. 加速度原始数据与转换后数据对比
    plt.subplot(3, 1, 1)
    # 图表标题
    plt.plot(time_data, data['加速度X(m/s²)'], label='acc 转换后X', alpha=0.7)
    plt.plot(time_data, data['加速度Y(m/s²)'], label='acc 转换后Y', alpha=0.7)
    plt.plot(time_data, data['加速度Z(m/s²)'], label='acc 转换后Z', alpha=0.7)
    plt.ylabel('加速度')
    # plt.ylim(-30, 30)  # 新增加速度Y轴范围
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=8)) # type: ignore 
    # plt.ylabel('加速度 (g)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_data, data['角速度X(°/s)'], label='gry 转换后X', alpha=0.7)
    plt.plot(time_data, data['角速度Y(°/s)'], label='gry 转换后Y', alpha=0.7)
    plt.plot(time_data, data['角速度Z(°/s)'], label='gry 转换后Z', alpha=0.7)
    # plt.ylim(-30, 30)  # 新增加速度Y轴范围
    plt.ylabel(f'角速度')
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=8)) # type: ignore 
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_data, data['角度X(°)'], label='elu 转换后X', alpha=0.7)
    plt.plot(time_data, data['角度Y(°)'], label='elu 转换后Y', alpha=0.7)
    # plt.plot(time_data, data['角度Z(°)'], label='elu 转换后Z', alpha=0.7)
    plt.plot(time_data, np.rad2deg(np.unwrap(np.deg2rad(data['角度Z(°)']))), label='elu 转换后Z')
    plt.ylabel(f'欧拉角')

    # 修改plot_sensor_data函数，在三个子图中添加：
    # 获取完整时间序列
    # full_times = data['时间'].values
    # print("length of full_times:", len(full_times))
    # print(full_times)

    # for turn in turns:
    #     # 获取转弯开始时间的时间部分
    #     full_time = full_times[turn]
    #     time_str = full_time.split(' ')[1]  #
    #     print("time_str:", time_str)
    #     # start_time = turn
    #     # # 找到对应的时间索引
    #     # start_idx = np.where(time_part == start_time)[0][0]
    #     plt.axvline(x=time_str, color='g', linestyle='--', alpha=0.7)
    
    # for turn in turns:
    #     # 获取转弯结束时间的时间部分
    #     end_time = turn[1].split(' ')[1]
    #     # 找到对应的时间索引
    #     end_idx = np.where(time_part == end_time)[0][0]
    #     plt.axvline(x=end_idx, color='r', linestyle='--', alpha=0.7)
    # plt.ylim(-30, 30)  # 新增加速度Y轴范围
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=8)) # type: ignore 
    # plt.ylabel('加速度 (g)')
    plt.legend()

    plt.tight_layout()

    plt.show()

def process_device_data(device_df):

    # 提取传感器数据和四元数
    accel_vectors = device_df[['加速度X(m/s²)', '加速度Y(m/s²)', '加速度Z(m/s²)']].values
    gyro_vectors = device_df[['角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']].values
    quaternions = device_df[['四元数0()','四元数1()','四元数2()','四元数3()']].values
    
    # 向量化旋转计算
    accel_world_vectors = rotate_vectors_by_quaternions(accel_vectors, quaternions)
    gyro_world_vectors = rotate_vectors_by_quaternions(gyro_vectors, quaternions)

    device_df['imu_加速度X(m/s²)'] = device_df['加速度X(m/s²)']
    device_df['imu_加速度Y(m/s²)'] = device_df['加速度Y(m/s²)']
    device_df['imu_加速度Z(m/s²)'] = device_df['加速度Z(m/s²)']
    device_df['imu_角速度X(°/s)'] = device_df['角速度X(°/s)']
    device_df['imu_角速度Y(°/s)'] = device_df['角速度Y(°/s)']
    device_df['imu_角速度Z(°/s)'] = device_df['角速度Z(°/s)']
    
    device_df['加速度X(m/s²)'] = accel_world_vectors[:, 0]
    device_df['加速度Y(m/s²)'] = accel_world_vectors[:, 1]
    device_df['加速度Z(m/s²)'] = accel_world_vectors[:, 2]
    device_df['角速度X(°/s)'] = gyro_world_vectors[:, 0]
    device_df['角速度Y(°/s)'] = gyro_world_vectors[:, 1]
    device_df['角速度Z(°/s)'] = gyro_world_vectors[:, 2]

    # 计算imu的欧拉角
    # 新增：将世界坐标系欧拉角转换回传感器本体坐标系
    euler_vectors = device_df[['角度X(°)', '角度Y(°)', '角度Z(°)']].values
    euler_body_vectors = rotate_vectors_back_to_body(euler_vectors, quaternions)
    
    device_df['角度X(°)'] = device_df['角度X(°)']
    device_df['角度Y(°)'] = device_df['角度Y(°)']
    device_df['角度Z(°)'] = device_df['角度Z(°)']
    device_df['imu_角度X(°)'] = euler_body_vectors[:, 0]
    device_df['imu_角度Y(°)'] = euler_body_vectors[:, 1]
    device_df['imu_角度Z(°)'] = euler_body_vectors[:, 2]


    """处理单个设备数据（新增滤波处理）"""

    # 新增滤波步骤
    acc_cols = ['加速度X(m/s²)', '加速度Y(m/s²)', '加速度Z(m/s²)']
    filtered_acc = np.zeros_like(device_df[acc_cols].to_numpy())
    for i, col in enumerate(acc_cols):
        filtered_acc[:, i] = apply_lowpass_filter(device_df[col].to_numpy())
        # filtered_acc[:, i] = device_df[col].to_numpy()
    
    # 角速度处理
    gyro_cols = ['角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']
    filtered_gyro = np.zeros_like(device_df[gyro_cols].to_numpy())
    for i, col in enumerate(gyro_cols):
        filtered_gyro[:, i] = apply_lowpass_filter(device_df[col].to_numpy())
        # filtered_gyro[:, i] = device_df[col].to_numpy()

    # 角度处理
    mag_cols = ['角度X(°)', '角度Y(°)', '角度Z(°)']
    filtered_mag = np.zeros_like(device_df[mag_cols].to_numpy())
    for i, col in enumerate(mag_cols):
        # filtered_mag[:, i] = apply_lowpass_filter(device_df[col].to_numpy(), cutoff=20)
        filtered_mag[:, i] = device_df[col].to_numpy()

    # 替换原始数据
    device_df_fil = device_df.copy()
    device_df_fil[['加速度X(m/s²)', '加速度Y(m/s²)', '加速度Z(m/s²)']] = filtered_acc
    device_df_fil[['角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']] = filtered_gyro
    device_df_fil[['角度X(°)', '角度Y(°)', '角度Z(°)']] = filtered_mag

    return device_df_fil

def main():
    # 1. 数据读取与预处理
    file_path='dataset/20250524/20250524124907-前刃推坡.txt'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    print("length of data:", len(df))
    # 新增时间格式转换和去重
    # df = df.drop_duplicates(subset=['时间', '设备名称'], keep='first')  # 按时间戳和设备去重
    # print("drop_duplicates length of data:", len(df))


    selected_columns = ['时间', '设备名称', '加速度X(g)', '加速度Y(g)', '加速度Z(g)', '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)', '四元数0()','四元数1()','四元数2()','四元数3()','角度X(°)','角度Y(°)','角度Z(°)']
    df['设备名称'] = df['设备名称'].str.split('(', n=1).str[0]
    df = df[selected_columns]

    # 转换加速度单位（g → m/s²）
    GRAVITY = 9.80665
    acc_columns = ['加速度X(g)', '加速度Y(g)', '加速度Z(g)']
    for col in acc_columns:
        new_col_name = col.replace('(g)', '(m/s²)')
        df[new_col_name] = df[col] * GRAVITY
    # 删除原始的g单位列
    df.drop(columns=acc_columns, inplace=True)

    # 调整列顺序（时间、设备名称在前，加速度XYZ在后）
    new_column_order = ['时间', '设备名称', 
                        '加速度X(m/s²)', '加速度Y(m/s²)', '加速度Z(m/s²)', 
                        '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)', 
                        '四元数0()','四元数1()','四元数2()','四元数3()',
                        '角度X(°)','角度Y(°)','角度Z(°)'
                    ]
    df = df.reindex(columns=new_column_order) # type:ignore

    # 2. 拆分设备数据
    devices = df['设备名称'].unique() 
    device_data = {name: df[df['设备名称'] == name] for name in devices}

    imu_data_left = device_data['WTGSB-A6'].copy() 
    imu_data_right = device_data['WTRIGHT'].copy()

    # 旋转矩阵、滤波处理
    imu_data_left = process_device_data(imu_data_left)
    imu_data_right = process_device_data(imu_data_right)


    plot_sensor_data(file_path,imu_data_left, 'left')
    plot_sensor_data(file_path,imu_data_right, 'right')


if __name__ == "__main__":
    main()