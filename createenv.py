import numpy as np
import math
from sac import SAC
import argparse
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
plt.ioff()
from creatgray import plotrec

# 判断是否使用 S型道路
Smake = 'False' # 是就用True 否就用False
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def calculate_distance(pos1, pos2):
        return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)


class Agent:
    def __init__(self, position, scan_radius, possible_moves):
        self.position = position
        self.scan_radius = scan_radius
        self.possible_moves = possible_moves
    # agents的移动
    # 采用apf试错机制 try所有的可能的dx以dy
    def get_possible_moves(self):
        moves = []
        for angle in np.linspace(0, 2 * np.pi, self.possible_moves):
            dx = self.scan_radius * math.cos(angle)
            dy = self.scan_radius * math.sin(angle)
            new_position = Position(self.position.x + dx, self.position.y + dy)
            moves.append(new_position)
        return moves
# class sacenv:
#     def __init__(self,):
# 目标 类
class Goal:
    def __init__(self, position, sigma):
        self.position = position
        self.sigma = sigma
    # 基于人工势场法的势场吸引力
    def get_attraction_force(self, move):
        distance = Position.calculate_distance(move, self.position)
        return -100 * math.exp(-distance / self.sigma)


class Obstacle:
    def __init__(self, position, sigma, draw_radius, shape='circle', width=0, height=0):
        self.position = position
        self.sigma = sigma
        self.draw_radius = draw_radius
        self.shape = shape
        self.width = width
        self.height = height
    # 基于人工势场法的斥力
    def get_repulsion_force(self, move):
        if self.shape == 'circle':
            distance = Position.calculate_distance(move, self.position) - self.draw_radius
        elif self.shape == 'rectangle':
            dx = np.abs(move.x - self.position.x) - self.width / 2
            dy = np.abs(move.y - self.position.y) - self.height / 2
            dx = np.maximum(dx, 0)
            dy = np.maximum(dy, 0)
            distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance < self.sigma:
            return 1000 * (1.0 / distance - 1.0 / self.sigma) ** 2
        else:
            return 0


# 检测设置的障碍是否碰撞
def is_overlapping(obstacle1, obstacle2):
    if obstacle1.shape == 'circle' and obstacle2.shape == 'circle':
        dist = Position.calculate_distance(obstacle1.position, obstacle2.position)
        return dist < (obstacle1.draw_radius + obstacle2.draw_radius)
    elif obstacle1.shape == 'rectangle' and obstacle2.shape == 'rectangle':
        return not (obstacle1.position.x + obstacle1.width / 2 < obstacle2.position.x - obstacle2.width / 2 or
                    obstacle1.position.x - obstacle1.width / 2 > obstacle2.position.x + obstacle2.width / 2 or
                    obstacle1.position.y + obstacle1.height / 2 < obstacle2.position.y - obstacle2.height / 2 or
                    obstacle1.position.y - obstacle1.height / 2 > obstacle2.position.y + obstacle2.height / 2)
    elif obstacle1.shape == 'circle' and obstacle2.shape == 'rectangle':
        circle, rect = obstacle1, obstacle2
    else:
        circle, rect = obstacle2, obstacle1

    closest_x = np.clip(circle.position.x, rect.position.x - rect.width / 2, rect.position.x + rect.width / 2)
    closest_y = np.clip(circle.position.y, rect.position.y - rect.height / 2, rect.position.y + rect.height / 2)
    distance = Position.calculate_distance(Position(closest_x, closest_y), circle.position)
    return distance < circle.draw_radius
def Sfunction(x,a,b):
    y = a*math.sin(x/30)
    return y
# 产生障碍物
def generate_random_obstacles(num_circles, num_rectangles, world_size):
    obstacles = []
    for _ in range(num_circles):
        while True:
            x = np.random.randint(50, world_size[0] - 200)
            if Smake == 'True':
                y = Sfunction(x, 20, 30)
            else:
                y = np.random.randint(30, 100)
            radius = np.random.randint(10, 20)
            sigma = np.random.randint(0, 10)
            new_obstacle = Obstacle(Position(x, y), sigma=sigma, draw_radius=radius, shape='circle')

            if all(not is_overlapping(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
                obstacles.append(new_obstacle)
                break

    for _ in range(num_rectangles):
        while True:
            x = np.random.randint(50, world_size[0] - 200)
            if Smake =='True':
                y = Sfunction(x, 20, 30)
            else:
                y = np.random.randint(30, 100)
            width = np.random.randint(20, 30)
            height = np.random.randint(10, 20)
            sigma = np.random.randint(0, 10)
            new_obstacle = Obstacle(Position(x, y), sigma=sigma, draw_radius=0, shape='rectangle', width=width,
                                    height=height)

            if all(not is_overlapping(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
                obstacles.append(new_obstacle)
                break
    # S型道路障碍物
    # S型道路
    if Smake == 'True':
        xr = np.linspace(50, 600, 5)
        x0 = np.linspace(50, 600, 660)
        y0 = [Sfunction(x1, 20, 30) + 60 for x1 in x0]
        y1 = [Sfunction(x1, 20, 30) -30 for x1 in x0]
        yr0 = [Sfunction( x1, 20, 30) +30 for x1 in xr]
        yr1 = [Sfunction( x1, 20, 30) for x1 in xr]
        plt.plot(x0, y0,'k')
        plt.plot(x0, y1, 'k')
        plotrec(xr, yr0)
        plotrec(xr, yr1)
    return obstacles

if __name__ == '__main__':
    # Define world size
    world_size = (700, 200)

    # Define agent and goal
    start_position = Position(15, 60)
    agent = Agent(start_position, scan_radius=10, possible_moves=30)
    goal = Goal(Position(600, 20), sigma=math.sqrt(world_size[0] ** 2 + world_size[1] ** 2))
    num_circles = 2
    num_rectangles = 3
    # Automatically generate obstacles along an S-shape
    obstacles = generate_random_obstacles(num_circles, num_rectangles, world_size)

    data_to_save = {
        "start_position": (start_position.x, start_position.y),
        "goal_position": (goal.position.x, goal.position.y),
        "obstacles": [{
            "position": (obs.position.x, obs.position.y),
            "shape": obs.shape,
            "radius": obs.draw_radius,
            "width": obs.width,
            "height": obs.height
        } for obs in obstacles]
    }

    np.save("path_and_obstacles_s_shape7.npy", data_to_save)



    print('start:', (start_position.x, start_position.y))
    for obstacle in obstacles:
        if obstacle.shape == 'circle':
            circle = plt.Circle((obstacle.position.x, obstacle.position.y), obstacle.draw_radius, color='red',
                                alpha=0.5)
            plt.gca().add_patch(circle)
        elif obstacle.shape == 'rectangle':
            rect = plt.Rectangle(
                (obstacle.position.x - obstacle.width / 2, obstacle.position.y - obstacle.height / 2),
                obstacle.width, obstacle.height, color='blue', alpha=0.5)
            plt.gca().add_patch(rect)

    plt.scatter(goal.position.x, goal.position.y, c='green', s=100, label='Goal')
    plt.scatter(agent.position.x, agent.position.y, c='red', s=100, label='Agent')
    plt.scatter(start_position.x, start_position.y, c='yellow', s=100, label='Start')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential Field with S-Shape Obstacles')
    plt.legend()
    plt.show()

