from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, RegularPolygon
from scipy.optimize import minimize
from matplotlib.path import Path
from numba import jit
from tqdm import tqdm
import time
import warnings
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')

@dataclass
class CarConfig:
    # 车辆物理参数配置
    dt: float = 0.1  # 仿真时间步长，单位：秒(s)
    length: float = 2.0  # 车辆长度，单位：米(m)
    width: float = 0.8  # 车辆宽度，单位：米(m)
    height: float = 0.4  # 车辆高度，单位：米(m)

@dataclass 
class MPCConfig:
    # MPC控制器参数配置
    N: int = 20  # 预测时域长度，无单位
    dt: float = 0.1  # 控制时间步长，单位：秒(s)
    v_max: float = 3.0  # 最大速度限制，单位：米/秒(m/s)
    v_min: float = -3.0  # 最小速度限制，单位：米/秒(m/s)
    delta_max: float = 0.8  # 最大转向角，单位：弧度(rad)
    delta_min: float = -0.8  # 最小转向角，单位：弧度(rad)
    max_iter: int = 100  # 最大迭代次数，无单位
    ftol: float = 1e-3  # 优化器收敛容差，无单位
    eps: float = 1e-2  # 数值计算精度，无单位

@dataclass
class ObstacleConfig:
    # 障碍物参数配置
    center: Tuple[float, float] = (0.0, 0.0)  # 障碍物中心坐标，单位：米(m)
    radius: float = 2.0  # 障碍物半径，单位：米(m)
    vertices: int = 6  # 多边形顶点数，无单位
    bbox_margin: float = 0.2  # 碰撞检测边界框边距，单位：米(m)

@dataclass
class SimulationConfig:
    # 仿真环境参数配置
    start: Tuple[float, float] = (-8.0, -8.0)  # 起点坐标，单位：米(m)
    goal: Tuple[float, float] = (8.0, 8.0)  # 终点坐标，单位：米(m)
    xlim: Tuple[float, float] = (-12.0, 12.0)  # x轴显示范围，单位：米(m)
    ylim: Tuple[float, float] = (-12.0, 12.0)  # y轴显示范围，单位：米(m)
    frames: int = 100  # 动画帧数，无单位
    interval: int = 50  # 帧间隔时间，单位：毫秒(ms)
    goal_threshold: float = 0.5  # 到达目标点判定阈值，单位：米(m)

@dataclass
class CostWeights:
    goal: float = 10.0  # 到达目标点的权重，无单位
    collision: float = 5000.0  # 避障权重，无单位
    safe_distance: float = 2.0  # 安全距离权重，无单位
    obstacle: float = 50.0  # 障碍物距离权重，无单位
    velocity: float = 0.05  # 速度平滑权重，无单位
    steering: float = 0.1  # 转向平滑权重，无单位
    heading: float = 0.3  # 航向角权重，无单位

@jit(nopython=True)
def car_model_numba(state, v, delta, dt=0.1):
    """Car kinematic model accelerated with numba"""
    x, y, theta = state
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(delta) / 2.0 * dt
    return np.array([x_new, y_new, theta_new])

class ImprovedObstacleAvoidanceMPC:
    def __init__(self, mpc_config: MPCConfig = MPCConfig(), 
                 obstacle_config: ObstacleConfig = ObstacleConfig(),
                 simulation_config: SimulationConfig = SimulationConfig(),
                 cost_weights: CostWeights = CostWeights()):
        self.config = mpc_config
        self.obstacle_config = obstacle_config
        self.sim_config = simulation_config
        self.weights = cost_weights
        
        self.start = np.array(self.sim_config.start)
        self.goal = np.array(self.sim_config.goal)
        self.last_control = None
        self.create_hexagon_obstacle()
        
    def create_hexagon_obstacle(self):
        angles = np.linspace(0, 2*np.pi, self.obstacle_config.vertices + 1)[:-1]
        self.obstacle_vertices = np.array([
            [self.obstacle_config.center[0] + self.obstacle_config.radius * np.cos(theta),
             self.obstacle_config.center[1] + self.obstacle_config.radius * np.sin(theta)]
            for theta in angles
        ])
        self.obstacle_path = Path(self.obstacle_vertices)
        self.bbox_min = np.min(self.obstacle_vertices, axis=0) - self.obstacle_config.bbox_margin
        self.bbox_max = np.max(self.obstacle_vertices, axis=0) + self.obstacle_config.bbox_margin
        
    def check_collision(self, point: np.ndarray) -> bool:
        x, y = point
        if (x < self.bbox_min[0] or x > self.bbox_max[0] or
            y < self.bbox_min[1] or y > self.bbox_max[1]):
            return False
        # Check if point is inside obstacle polygon
        return self.obstacle_path.contains_point(point)
    
    def cost_function(self, u: np.ndarray, state: np.ndarray) -> float:
        cost = 0
        current_state = np.array(state)
        
        for i in range(self.config.N):
            v = u[2*i]
            delta = u[2*i + 1]
            
            current_state = car_model_numba(current_state, v, delta, self.config.dt)
            x, y = current_state[0], current_state[1]
            
            # Goal cost
            distance_to_goal = np.hypot(x - self.goal[0], y - self.goal[1])
            cost += self.weights.goal * distance_to_goal
            
            # Collision cost
            if self.check_collision([x, y]):
                return self.weights.collision + cost
                
            # Obstacle avoidance cost
            distances = np.hypot(x - self.obstacle_vertices[:, 0],
                               y - self.obstacle_vertices[:, 1])
            distance_to_obstacle = np.min(distances)
            
            safe_distance = self.obstacle_config.radius * self.weights.safe_distance
            if distance_to_obstacle < safe_distance:
                cost += self.weights.obstacle * np.exp(-(distance_to_obstacle/safe_distance))
            
            # Control input cost
            cost += self.weights.velocity * v**2 + self.weights.steering * delta**2
            
            # Heading cost
            desired_theta = np.arctan2(self.goal[1] - y, self.goal[0] - x)
            angle_diff = np.abs(np.arctan2(np.sin(current_state[2] - desired_theta), 
                                         np.cos(current_state[2] - desired_theta)))
            cost += self.weights.heading * angle_diff
            
        return cost
    

    def solve_mpc(self, state: np.ndarray) -> np.ndarray:
        angle_to_goal = np.arctan2(self.goal[1] - state[1], 
                                  self.goal[0] - state[0])
        
        if self.last_control is not None:
            v_guess = self.last_control[0]
            delta_guess = self.last_control[1]
        else:
            v_guess = 1.5
            delta_guess = 0.2 * (angle_to_goal - state[2])
        
        u0 = np.array([v_guess, delta_guess] * self.config.N)
        bounds = [(self.config.v_min, self.config.v_max), 
                 (self.config.delta_min, self.config.delta_max)] * self.config.N
        
        result = minimize(self.cost_function, 
                        u0, 
                        args=(state,),
                        method='SLSQP',
                        bounds=bounds,
                        options={
                            'maxiter': self.config.max_iter,
                            'ftol': self.config.ftol,
                            'eps': self.config.eps
                        })
        
        self.last_control = (result.x[0], result.x[1])
        return result.x

class ImprovedAnimationManager:
    def __init__(self, car_config: CarConfig = CarConfig()):
        self.car_config = car_config
        self.mpc = ImprovedObstacleAvoidanceMPC()
        self.setup_visualization()
        self.initialize_state()
        
    def setup_visualization(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.setup_plot()
        self.progress_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=12, color='white',
                                        verticalalignment='top')
        
    def initialize_state(self):
        self.state = np.array([*self.mpc.start, np.arctan2(
            self.mpc.goal[1] - self.mpc.start[1],
            self.mpc.goal[0] - self.mpc.start[0]
        )])
        self.trajectory = [self.state[:2]]
        
    def setup_plot(self):
        self.ax.set_xlim(*self.mpc.sim_config.xlim)
        self.ax.set_ylim(*self.mpc.sim_config.ylim)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Improved MPC Obstacle Avoidance', fontsize=14, pad=20)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Plot start and goal points
        self.ax.plot(*self.mpc.start, 'go', markersize=15, label='Start')
        self.ax.plot(*self.mpc.goal, 'ro', markersize=15, label='Goal')
        
        # Create obstacle
        self.obstacle = RegularPolygon(
            self.mpc.obstacle_config.center, 
            self.mpc.obstacle_config.vertices,
            radius=self.mpc.obstacle_config.radius,
            fc='gray', 
            alpha=0.5
        )
        self.ax.add_patch(self.obstacle)
        
        # Create trajectory line and car rectangle
        self.trajectory_line, = self.ax.plot([], [], 'cyan', 
                                           label='Actual Trajectory', 
                                           linewidth=2)
        self.car = Rectangle((0, 0), 
                           self.car_config.width, 
                           self.car_config.height, 
                           angle=0, 
                           fc='red', 
                           alpha=0.7)
        self.ax.add_patch(self.car)
        self.ax.legend(fontsize=12, loc='upper right')
        
    def init(self):
        self.trajectory_line.set_data([], [])
        return self.trajectory_line, self.car, self.progress_text
        
    def update(self, frame):
        progress = (frame + 1) / self.total_frames * 100
        elapsed_time = time.time() - self.start_time
        avg_speed = (frame + 1) / elapsed_time
        remaining_frames = self.total_frames - (frame + 1)
        eta = remaining_frames / avg_speed if avg_speed > 0 else 0
        
        status_text = (f'Progress: {progress:.1f}%\n'
                      f'FPS: {avg_speed:.1f}\n'
                      f'ETA: {eta:.1f}s')
        self.progress_text.set_text(status_text)
        
        try:
            u = self.mpc.solve_mpc(self.state)
            self.state = car_model_numba(self.state, u[0], u[1])
            
            self.trajectory.append(self.state[:2])
            trajectory = np.array(self.trajectory)
            
            self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
            self.car.set_xy([self.state[0] - self.car_config.width/2, 
                           self.state[1] - self.car_config.height/2])
            self.car.angle = np.degrees(self.state[2])
            
            if np.linalg.norm(self.state[:2] - self.mpc.goal) < self.mpc.sim_config.goal_threshold:
                print("\nGoal reached successfully!")
                self.event_source.stop()
                
        except Exception as e:
            print(f"\nError in MPC update: {e}")
            self.event_source.stop()
        
        return self.trajectory_line, self.car, self.progress_text
        
    def save_animation(self, filename='improved_mpc.gif'):
        print("Starting animation generation...")
        self.total_frames = self.mpc.sim_config.frames
        self.start_time = time.time()
        
        progress_bar = tqdm(total=self.total_frames, 
                          desc="Generating animation", 
                          unit="frames", 
                          ncols=100)
        
        def update_wrapper(frame):
            result = self.update(frame)
            progress_bar.update(1)
            return result
        
        anim = FuncAnimation(self.fig, 
                           update_wrapper, 
                           init_func=self.init,
                           frames=self.total_frames, 
                           interval=self.mpc.sim_config.interval, 
                           blit=True,
                           cache_frame_data=False)
        
        self.event_source = anim.event_source
        
        writer = PillowWriter(fps=20)
        anim.save(filename, writer=writer, dpi=100)
        
        progress_bar.close()
        
        total_time = time.time() - self.start_time
        print(f"\nAnimation generation complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {self.total_frames/total_time:.2f}")
        print(f"File saved as: {filename}")
        plt.close()

if __name__ == "__main__":
    animation = ImprovedAnimationManager()
    animation.save_animation()