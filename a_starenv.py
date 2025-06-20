import numpy as np
import math
import matplotlib.pyplot as plt
from functools import lru_cache
from queue import PriorityQueue


# Define the Node class
class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __add__(self, other):
        return Node(self.x + other[0], self.y + other[1], self.cost, self)

    def __sub__(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.cost < other.cost

    def __hash__(self):
        return hash((self.x, self.y))


# Define the Position class
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def calculate_distance(pos1, pos2):
        return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)


# Define the Agent class
class Agent:
    def __init__(self, position, scan_radius, possible_moves):
        self.position = position
        self.scan_radius = scan_radius
        self.possible_moves = possible_moves

    def get_possible_moves(self):
        moves = []
        for angle in np.linspace(0, 2 * np.pi, self.possible_moves):
            dx = self.scan_radius * math.cos(angle)
            dy = self.scan_radius * math.sin(angle)
            # Add random perturbation
            random_offset_x = np.random.uniform(-1, 1) * 0.1
            random_offset_y = np.random.uniform(-1, 1) * 0.1
            new_position = Position(self.position.x + dx + random_offset_x, self.position.y + dy + random_offset_y)
            moves.append(new_position)
        return moves


# Define the Goal class
class Goal:
    def __init__(self, position, sigma):
        self.position = position
        self.sigma = sigma

    def get_attraction_force(self, move):
        distance = Position.calculate_distance(move, self.position)
        return -50 * math.exp(-distance / self.sigma)  # Reduce the strength of attraction


# Define the Obstacle class
class Obstacle:
    def __init__(self, position, sigma, draw_radius, shape='circle', width=0, height=0):
        self.position = position
        self.sigma = sigma
        self.draw_radius = draw_radius
        self.shape = shape
        self.width = width
        self.height = height

    def get_repulsion_force(self, move):
        if self.shape == 'circle':
            distance = Position.calculate_distance(move, self.position) - self.draw_radius
        elif self.shape == 'rectangle':
            dx = np.abs(move.x - self.position.x) - self.width / 2
            dy = np.abs(move.y - self.position.y) - self.height / 2
            dx = np.maximum(dx, 0)
            dy = np.maximum(dy, 0)
            distance = np.sqrt(dx ** 2 + dy ** 2)

        epsilon = 1e-6
        if distance < epsilon:
            distance = epsilon

        # Introduce critical distance
        critical_distance = 5
        if distance < critical_distance:
            return float('inf')  # Directly repel outside the obstacle

        # Dynamically adjust repulsion range (adaptive repulsion)
        if distance < self.sigma:
            scale_factor = np.tanh(5 * (self.sigma - distance))  # Dynamically adjust repulsion range
            return scale_factor * 10000 * (1.0 / distance - 1.0 / self.sigma) ** 2
        else:
            return 5000 * (1.0 / self.sigma) ** 2  # Increase the impact range and strength of repulsion


# Check if two obstacles are overlapping
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


# Generate random obstacles
def generate_random_obstacles(num_circles, num_rectangles, world_size):
    obstacles = []
    for _ in range(num_circles):
        while True:
            x = np.random.randint(50, world_size[0] - 50)
            y = np.random.randint(50, world_size[1] - 50)
            radius = np.random.randint(20, 40)
            sigma = np.random.randint(50, 100)
            new_obstacle = Obstacle(Position(x, y), sigma=sigma, draw_radius=radius, shape='circle')

            if all(not is_overlapping(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
                obstacles.append(new_obstacle)
                break

    for _ in range(num_rectangles):
        while True:
            x = np.random.randint(50, world_size[0] - 50)
            y = np.random.randint(50, world_size[1] - 50)
            width = np.random.randint(40, 80)
            height = np.random.randint(20, 60)
            sigma = np.random.randint(50, 100)
            new_obstacle = Obstacle(Position(x, y), sigma=sigma, draw_radius=0, shape='rectangle', width=width,
                                    height=height)

            if all(not is_overlapping(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
                obstacles.append(new_obstacle)
                break

    return obstacles


# Create map array with obstacles
def create_map_with_obstacles(world_size, obstacles):
    map_array = np.ones(world_size, dtype=np.uint8) * 255  # Initialize map with free space

    for obstacle in obstacles:
        if obstacle.shape == 'circle':
            for i in range(world_size[0]):
                for j in range(world_size[1]):
                    if Position.calculate_distance(Position(i, j), obstacle.position) < obstacle.draw_radius:
                        map_array[j, i] = 0  # Mark obstacle on the map
        elif obstacle.shape == 'rectangle':
            for i in range(int(obstacle.position.x - obstacle.width // 2), int(obstacle.position.x + obstacle.width // 2)):
                for j in range(int(obstacle.position.y - obstacle.height // 2), int(obstacle.position.y + obstacle.height // 2)):
                    if 0 <= i < world_size[0] and 0 <= j < world_size[1]:
                        map_array[j, i] = 0  # Mark obstacle on the map

    return map_array


# Define the AStar class
class AStar:
    def __init__(
            self,
            start_pos,
            end_pos,
            map_array,
            obstacles,
            move_step=3,
            move_direction=8,
    ):
        self.map_array = map_array
        self.width = self.map_array.shape[1]
        self.height = self.map_array.shape[0]
        self.start = Node(*start_pos)
        self.end = Node(*end_pos)
        self.obstacles = obstacles

        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"x坐标范围0~{self.width - 1}, y坐标范围0~{self.height - 1}")
        if self._is_collided(self.start):
            raise ValueError(f"起点x坐标或y坐标在障碍物上")
        if self._is_collided(self.end):
            raise ValueError(f"终点x坐标或y坐标在障碍物上")

        self.reset(move_step, move_direction)

    def reset(self, move_step=3, move_direction=8):
        self.__reset_flag = False
        self.move_step = move_step
        self.move_direction = move_direction
        self.close_set = set()
        self.open_queue = PriorityQueue()
        self.path_list = []

    def search(self):
        return self.__call__()

    def _in_map(self, node: Node):
        return (0 <= node.x < self.width) and (0 <= node.y < self.height)

    def _is_collided(self, node: Node):
        if self.map_array[node.y, node.x] == 0:
            return True
        for obstacle in self.obstacles:
            if obstacle.shape == 'circle':
                if Position.calculate_distance(Position(node.x, node.y), obstacle.position) < obstacle.draw_radius:
                    return True
            elif obstacle.shape == 'rectangle':
                if obstacle.position.x - obstacle.width / 2 <= node.x <= obstacle.position.x + obstacle.width / 2 and \
                        obstacle.position.y - obstacle.height / 2 <= node.y <= obstacle.position.y + obstacle.height / 2:
                    return True
        return False

    def _move(self):
        @lru_cache(maxsize=16)
        def _move(move_step: int, move_direction: int):
            move = [
                (0, move_step),
                (0, -move_step),
                (-move_step, 0),
                (move_step, 0),
                (move_step, move_step),
                (move_step, -move_step),
                (-move_step, move_step),
                (-move_step, -move_step),
            ]
            return move[0:move_direction]

        return _move(self.move_step, self.move_direction)

    def _update_open_list(self, curr: Node):
        for add in self._move():
            next_ = curr + add

            if not self._in_map(next_):
                continue
            if self._is_collided(next_):
                continue
            if next_ in self.close_set:
                continue

            H = next_ - self.end
            next_.cost += H

            self.open_queue.put(next_)

            if H < 20:
                self.move_step = 1

    def __call__(self):
        assert not self.__reset_flag, "call之前需要reset"
        print("搜索中\n")

        self.open_queue.put(self.start)
        while not self.open_queue.empty():
            curr = self.open_queue.get()
            curr.cost -= (curr - self.end)

            self._update_open_list(curr)
            self.close_set.add(curr)

            if curr == self.end:
                break
        else:
            print("路径未找到\n")
            return []

        print("路径搜索完成\n")

        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()

        self.__reset_flag = True

        return self.path_list


# Visualize the path
def visualize_path(map_array, path):
    plt.imshow(map_array, cmap='gray')

    x_vals = [node.x for node in path]
    y_vals = [node.y for node in path]

    plt.plot(x_vals, y_vals, color='red', marker='o')
    plt.show()


# Main execution
if __name__ == "__main__":

    world_size = (200, 700)



    start_position = Position(15, 60)
    agent = Agent(start_position, scan_radius=10, possible_moves=30)
    goal = Goal(Position(600, 20), sigma=math.sqrt(world_size[0] ** 2 + world_size[1] ** 2))
    data_loaded = np.load("path_and_obstacles_s_shape7.npy", allow_pickle=True).item()

    obstacles = []
    for obs_data in data_loaded["obstacles"]:
        position = Position(*obs_data["position"])
        sigma = obs_data.get("sigma", 0.5)
        obstacle = Obstacle(
            position=position,
            sigma=sigma,
            draw_radius=obs_data["radius"],
            shape=obs_data["shape"],
            width=obs_data["width"],
            height=obs_data["height"]
        )
        obstacles.append(obstacle)

    map_array = create_map_with_obstacles(world_size, obstacles)

    start_pos = (15, 60)
    end_pos = (600, 20)


    astar = AStar(start_pos=start_pos, end_pos=end_pos, map_array=map_array, obstacles=obstacles)
    path = astar.search()
    path_points = []
    for node in path:
        print(f"({node.x}, {node.y})")
        path_points.append((node.x, node.y))

    np.save('path_points_astar_7.npy', path_points)
    visualize_path(map_array, path)
