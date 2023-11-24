import path_planning as pp
import matplotlib.pyplot as plt
from pso import PSO
import cv2
import numpy as np
plt.rcParams["figure.autolayout"] = True
def select_start_and_goal(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Chọn điểm bắt đầu (click) và điểm kết thúc (click)')
    start = plt.ginput(1)  # Chọn điểm bắt đầu
    goal = plt.ginput(1)  # Chọn điểm kết thúc
    plt.close()
    # Chuyển tọa độ chuột (pixel) thành tọa độ ô grid
    start = (int(start[0][1] / grid_width), int(start[0][0] / grid_height))
    goal = (int(goal[0][1] / grid_width), int(goal[0][0] / grid_height))

    return start, goal
path = r'C:\Users\lamto\OneDrive\Desktop\school3.png'
image = cv2.imread(path)
grid_height = 1
grid_width = 1
start, goal = select_start_and_goal(image)
num_rows = image.shape[0] // grid_height
num_cols = image.shape[1] // grid_width
grid = np.zeros((num_rows, num_cols))
# Xác định ngưỡng để phân biệt giữa màu đen và màu trung bình của ô grid
threshold_value = 220  # Giá trị ngưỡng (có thể điều chỉnh)
for row in range(num_rows):
    for col in range(num_cols):
        top = row * grid_height
        bottom = (row + 1) * grid_height
        left = col * grid_width
        right = (col + 1) * grid_width
        grid_image = image[top:bottom, left:right]
        average_color = np.mean(grid_image)
        if average_color < threshold_value:
            grid[row][col] = 1
# Create environment
env_params = {
    'width': image.shape[0],
    'height': image.shape[1],
    'robot_radius': 1,
    'start': start,
    'goal': goal,
}
env = pp.Environment(**env_params)

# Obstacles
obstacles = []
# Lặp qua ma trận grid và thêm vào tập obstacles
for row in range(num_rows):
    for col in range(num_cols):
        if grid[row][col] == 1:
            # Xác định tọa độ thực của ô grid
            real_x = col * grid_width + grid_width / 2
            real_y = row * grid_height + grid_height / 2
            # Thêm vào tập obstacles
            obstacles.append({
                'center': [real_x, real_y],
                'radius': min(grid_width, grid_height) / 2  # Đường kính là tối thiểu giữa chiều rộng và chiều cao
            })
for obs in obstacles:
    env.add_obstacle(pp.Obstacle(**obs))

# Create cost function
num_control_points = 3
resolution = 50
cost_function = pp.EnvCostFunction(env, num_control_points, resolution)

# Optimization Problem
problem = {
    'num_var': 2*num_control_points,
    'var_min': 0,
    'var_max': 1,
    'cost_function': cost_function,
}

# Callback function
path_line = None
def callback(data):
    global path_line
    it = data['it']
    sol = data['gbest']['details']['sol']
    if it==1:
        fig = plt.figure(figsize=[7, 7])
        pp.plot_environment(env)
        path_line = pp.plot_path(sol, color='b')
        plt.grid(True)
        plt.show(block=False)
    else:
        pp.update_path(sol, path_line)

    length = data['gbest']['details']['length']
    plt.title(f"Iteration: {it}, Length: {length:.2f}")

# Run PSO
pso_params = {
    'max_iter': 100,
    'pop_size': 1000,
    'c1': 2,
    'c2': 1,
    'w': 0.8,
    'wdamp': 1,
    'resetting': 25,
}
bestsol, pop = PSO(problem, callback=callback, **pso_params)

plt.show()
