import heapq
import numpy as np
import math
import matplotlib.pyplot as plt

# 定义Node类
class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __add__(self, other):
        return Node(self.x + other[0], self.y + other[1], self.cost, self)

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

# 定义优先队列 (Min-Heap) 辅助类
class NodeQueue:
    def __init__(self):
        self.elements = []

    def put(self, item):
        heapq.heappush(self.elements, item)

    def get(self):
        return heapq.heappop(self.elements)

    def empty(self):
        return len(self.elements) == 0

# Dijkstra 算法类
class Dijkstra:
    def __init__(self, start_pos, end_pos, map_array, obstacles, move_step=3, move_direction=8):
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
        self.open_queue = NodeQueue()
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

        def _move(move_step: int, move_direction: int):
            move = (
                (0, move_step),  # 上
                (0, -move_step),  # 下
                (-move_step, 0),  # 左
                (move_step, 0),  # 右
                (move_step, move_step),  # 右上
                (move_step, -move_step),  # 右下
                (-move_step, move_step),  # 左上
                (-move_step, -move_step),  # 左下
            )
            return move[0:move_direction]

        return _move(self.move_step, self.move_direction)

    def _update_open_list(self, curr: Node):
        """open_list添加可行点"""
        for add in self._move():
            next_ = curr + add

            if not self._in_map(next_):
                continue
            if self._is_collided(next_):
                continue
            if next_ in self.close_set:
                continue

            # 更新路径代价
            G = curr.cost + math.sqrt((curr.x - next_.x) ** 2 + (curr.y - next_.y) ** 2)
            next_.cost = G

            self.open_queue.put(next_)

            # 当剩余距离小时, 走慢一点
            if G < 20:
                self.move_step = 1

    def __call__(self):
        assert not self.__reset_flag, "call之前需要reset"
        print("搜索中\n")

        self.start.cost = 0
        self.open_queue.put(self.start)

        while not self.open_queue.empty():
            curr = self.open_queue.get()

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

# 定义 Position 类和障碍物生成代码
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def calculate_distance(pos1, pos2):
        return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)


class Obstacle:
    def __init__(self, position, sigma, draw_radius, shape='circle', width=0, height=0):
        self.position = position
        self.sigma = sigma
        self.draw_radius = draw_radius
        self.shape = shape
        self.width = width
        self.height = height


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


def create_map_with_obstacles(world_size, obstacles):
    map_array = np.ones(world_size, dtype=np.uint8) * 255

    for obstacle in obstacles:
        if obstacle.shape == 'circle':
            for i in range(world_size[0]):
                for j in range(world_size[1]):
                    if Position.calculate_distance(Position(i, j), obstacle.position) < obstacle.draw_radius:
                        map_array[j, i] = 0
        elif obstacle.shape == 'rectangle':
            for i in range(obstacle.position.x - obstacle.width // 2, obstacle.position.x + obstacle.width // 2):
                for j in range(obstacle.position.y - obstacle.height // 2, obstacle.position.y + obstacle.height // 2):
                    if 0 <= i < world_size[0] and 0 <= j < world_size[1]:
                        map_array[j, i] = 0

    return map_array


def visualize_path(map_array, path):
    plt.imshow(np.flipud(map_array), cmap='gray')

    x_vals = [node.x for node in path]
    y_vals = [map_array.shape[0] - 1 - node.y for node in path]

    plt.plot(x_vals, y_vals, color='red', marker='o')
    plt.gca().invert_yaxis()
    plt.show()


def is_valid_position(position, map_array, obstacles):
    # 检查是否在地图边界内
    if position.x < 0 or position.x >= map_array.shape[1] or position.y < 0 or position.y >= map_array.shape[0]:
        return False

    # 检查是否碰撞障碍物
    if map_array[position.y, position.x] == 0:
        return False

    for obstacle in obstacles:
        if obstacle.shape == 'circle':
            if Position.calculate_distance(position, obstacle.position) < obstacle.draw_radius:
                return False
        elif obstacle.shape == 'rectangle':
            if obstacle.position.x - obstacle.width / 2 <= position.x <= obstacle.position.x + obstacle.width / 2 and \
                    obstacle.position.y - obstacle.height / 2 <= position.y <= obstacle.position.y + obstacle.height / 2:
                return False

    return True


def find_valid_start_end(map_array, obstacles):
    while True:
        start_pos = Position(np.random.randint(0, map_array.shape[1]), np.random.randint(0, map_array.shape[0]))
        end_pos = Position(np.random.randint(0, map_array.shape[1]), np.random.randint(0, map_array.shape[0]))

        if is_valid_position(start_pos, map_array, obstacles) and is_valid_position(end_pos, map_array, obstacles):
            return (start_pos.x, start_pos.y), (end_pos.x, end_pos.y)


if __name__ == "__main__":
    world_size = (500, 500)
    num_circles = 5
    num_rectangles = 3

    # 生成障碍物
    obstacles = generate_random_obstacles(num_circles, num_rectangles, world_size)
    map_array = create_map_with_obstacles(world_size, obstacles)

    # 自动选择有效的起点和终点
    start_pos, end_pos = find_valid_start_end(map_array, obstacles)

    # 使用 Dijkstra 进行路径规划
    dijkstra = Dijkstra(start_pos=start_pos, end_pos=end_pos, map_array=map_array, obstacles=obstacles)
    path = dijkstra.search()

    if path:
        path_points = [(node.x, node.y) for node in path]
        np.save('path_points_dijkstra.npy', path_points)
        visualize_path(map_array, path)
    else:
        print("无法找到路径，请重新尝试。")
