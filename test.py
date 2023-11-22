import pandas as pd
import torch

# 读取csv数据
df = pd.read_csv("./data/TOC_WF_LMX_4K_log.csv")
df = df.drop(columns='Depth/Thickness(m)')
df = df.drop(columns='Journal')
df = df.drop(columns='Author')
df = df.drop(columns='Well')
df = df.drop(columns='Area')
df = df.drop(columns='DOI1')
df = df.drop(columns='DOI2')
df = df.drop(columns='Unnamed: 11')
df = df.drop(columns='Unnamed: 12')

target = df['TOC(%)']
x = df.drop('TOC(%)', axis=1)

# 定义评估函数
def evaluate(x, para_matrix, well_log_matrix, ro_matrix, bias_matrix):
    baseline_trans = torch.matmul(x, well_log_matrix.T)
    para_gain = torch.matmul(x, para_matrix.T)
    para_trans = torch.matmul(baseline_trans, para_gain)
    output = torch.matmul(para_trans.T, torch.pow(10.0, ro_matrix)) + bias_matrix
    return output

# 模拟退火算法
def simulated_annealing(x, initial_para_matrix, initial_well_log_matrix, initial_ro_matrix, initial_bias_matrix, temperature, cooling_rate, num_iterations):
    current_para_matrix = initial_para_matrix.clone()
    best_para_matrix = initial_para_matrix.clone()

    current_well_log_matrix = initial_well_log_matrix.clone()
    best_well_log_matrix = initial_well_log_matrix.clone()

    current_para_matrix = initial_para_matrix.clone()
    best_para_matrix = initial_para_matrix.clone()

    current_ro_matrix = initial_ro_matrix.clone()
    best_ro_matrix = initial_ro_matrix.clone()

    current_bias_matrix = initial_bias_matrix.clone()
    best_bias_matrix = initial_bias_matrix.clone()

    current_output = evaluate(x, current_para_matrix, current_well_log_matrix, current_ro_matrix, current_bias_matrix)
    best_output = current_output.clone()

    for i in range(num_iterations):
        # 生成新的参数矩阵
        new_para_matrix = current_para_matrix + torch.randn_like(current_para_matrix)
        new_well_log_matrix = current_well_log_matrix + torch.randn_like(current_well_log_matrix)
        new_ro_matrix = current_ro_matrix + torch.randn_like(current_ro_matrix)
        new_bias_matrix = current_bias_matrix + torch.randn_like(current_bias_matrix)

        new_output = evaluate(x, new_para_matrix, new_well_log_matrix, new_ro_matrix, new_bias_matrix)

        # 计算能量差
        energy_diff = new_output - current_output

        # 判断是否接受新的参数矩阵
        if torch.sum(energy_diff) < 0 or torch.rand(1) < torch.exp(-torch.sum(energy_diff) / temperature):
            current_para_matrix = new_para_matrix
            current_output = new_output

        # 更新最优解
        if torch.sum(current_output) < torch.sum(best_output):
            best_para_matrix = current_para_matrix
            best_well_log_matrix = current_well_log_matrix
            best_ro_matrix = current_ro_matrix
            best_bias_matrix = current_bias_matrix
            best_output = current_output

        # 降低温度
        temperature *= cooling_rate

    return best_para_matrix, best_well_log_matrix, best_ro_matrix, best_bias_matrix, best_output

def calculate_statistics(matrix):
   """
   Calculate the mean, standard deviation, 2σ error, maximum, and minimum of a matrix.
   
   Args:
       matrix (torch.Tensor): A matrix of shape (num_samples, num_features).
       
   Returns:
       mean (torch.Tensor): The mean of each feature, of shape (num_features,).
       std (float): The standard deviation of each feature.
       two_sigma (float): The 2σ error of each feature.
       max_val (float): The maximum value of the matrix.
       min_val (float): The minimum value of the matrix.
   """
   num_samples, num_features = matrix.shape
   flattened_matrix = matrix.flatten()
   mean = torch.mean(flattened_matrix)
   std = torch.std(flattened_matrix).item()
   z_score = (flattened_matrix - mean) / std
   two_sigma = 2 * torch.mean(z_score).item()
   max_val = torch.max(flattened_matrix).item()
   min_val = torch.min(flattened_matrix).item()
   return std, two_sigma, max_val, min_val


# 使用
x = torch.randn(10, 10)  # 输入矩阵
initial_para_matrix = torch.randn(10, 10)  # 初始参数矩阵
initial_well_log_matrix = torch.randn(10, 10)  # well_log_matrix
initial_ro_matrix = torch.randn(10, 10)  # ro_matrix
initial_bias_matrix = torch.randn(10, 10)  # bias_matrix
temperature = 100.0  # 初始温度
cooling_rate = 0.05  # 降温速率
num_iterations = 1000000  # 迭代次数

best_para_matrix, best_well_log_matrix, best_ro_matrix, best_bias_matrix, best_output = simulated_annealing(
    x, 
    initial_para_matrix, 
    initial_well_log_matrix, 
    initial_ro_matrix, 
    initial_bias_matrix, 
    temperature, 
    cooling_rate, 
    num_iterations)

para_std, para_two_sigma, para_max, para_min = calculate_statistics(best_para_matrix)
well_log_std, well_log_two_sigma, well_log_max, well_log_min = calculate_statistics(best_well_log_matrix)
ro_std, ro_two_sigma, ro_max, ro_min = calculate_statistics(best_ro_matrix)
bias_std, bias_two_sigma, bias_max, bias_min = calculate_statistics(best_bias_matrix)

print(
   f"Para Matrix: {best_para_matrix}, "
   f"Mean: {para_std:.4f}, "
   f"Max: {para_max:.4f}, "
   f"Min: {para_min:.4f}, "
   f"Two Sigma: {para_two_sigma:.8f}"
)

print(
   f"Well Log Matrix: {best_well_log_matrix}, "
   f"Mean: {well_log_std:.4f}, "
   f"Max: {well_log_max:.4f}, "
   f"Min: {well_log_min:.4f}, "
   f"Two Sigma: {well_log_two_sigma:.8f}"
)

print(
   f"RO Matrix: {best_ro_matrix}, "
   f"Mean: {ro_std:.4f}, "
   f"Max: {ro_max:.4f}, "
   f"Min: {ro_min:.4f}, "
   f"Two Sigma: {ro_two_sigma:.8f}"
)

print(
   f"Bias Matrix: {best_bias_matrix}, "
   f"Mean: {bias_std:.4f}, "
   f"Max: {bias_max:.4f}, "
   f"Min: {bias_min:.4f}, "
   f"Two Sigma: {bias_two_sigma:.8f}"
)
