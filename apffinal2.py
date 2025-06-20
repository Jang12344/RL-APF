import numpy as np
import math
import matplotlib.pyplot as plt



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

    def get_possible_moves(self):
        moves = []
        for angle in np.linspace(0, 2 * np.pi, self.possible_moves):
            dx = self.scan_radius * math.cos(angle)
            dy = self.scan_radius * math.sin(angle)
            # 增加随机扰动
            random_offset_x = np.random.uniform(-1, 1) * 0.1
            random_offset_y = np.random.uniform(-1, 1) * 0.1
            new_position = Position(self.position.x + dx + random_offset_x, self.position.y + dy + random_offset_y)
            moves.append(new_position)
        return moves


class Goal:
    def __init__(self, position, sigma):
        self.position = position
        self.sigma = sigma

    def get_attraction_force(self, move):
        distance = Position.calculate_distance(move, self.position)
        return -50 * math.exp(-distance / self.sigma)  # 减小吸引力的强度


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

        # 引入临界距离
        critical_distance = 5
        if distance < critical_distance:
            return float('inf')  # 直接排斥到障碍物外部

        # 动态调整斥力范围（自适应斥力）
        if distance < self.sigma:
            scale_factor = np.tanh(5 * (self.sigma - distance))  # 动态调整斥力范围
            return scale_factor * 10000 * (1.0 / distance - 1.0 / self.sigma) ** 2
        else:
            return 5000 * (1.0 / self.sigma) ** 2  # 增大斥力的影响范围和强度


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


if __name__ == '__main__':
    world_size = (700, 200)

    # Define agent and goal
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



    path_points = []
    while Position.calculate_distance(agent.position, goal.position) > 10:
        possible_moves = agent.get_possible_moves()
        min_value = math.inf
        best_move = possible_moves[0]
        for move in possible_moves:
            move_value = goal.get_attraction_force(move)
            for obstacle in obstacles:
                move_value += obstacle.get_repulsion_force(move)

            if move_value < min_value:
                min_value = move_value
                best_move = move

        agent.position = best_move
        path_points.append((best_move.x, best_move.y))

    data_to_save = {
        "start_position": (start_position.x, start_position.y),
        "goal_position": (goal.position.x, goal.position.y),
        "path_points": path_points,
        "obstacles": [{
            "position": (obs.position.x, obs.position.y),
            "shape": obs.shape,
            "radius": obs.draw_radius,
            "width": obs.width,
            "height": obs.height
        } for obs in obstacles]
    }

    np.save('path_points_apf_7.npy', path_points)

    x = np.linspace(0, world_size[0], world_size[0])
    y = np.linspace(0, world_size[1], world_size[1])
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)


    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = Position(X[i, j], Y[i, j])
            U[i, j] = goal.get_attraction_force(pos)
            for obstacle in obstacles:
                U[i, j] += obstacle.get_repulsion_force(pos)

    U[np.isinf(U)] = np.nanmax(U[~np.isinf(U)])
    U[np.isnan(U)] = np.nanmax(U[~np.isnan(U)])

    plt.figure(figsize=(10, 8))
    levels = np.linspace(np.nanmin(U), np.nanmax(U), 150)  # 调整等高线的数量

    plt.contourf(X, Y, U, levels=levels, cmap='plasma')  # 使用不同的颜色映射
    plt.colorbar(label='Potential')
    plt.plot([p[0] for p in path_points], [p[1] for p in path_points], 'w-', label='Path')
    plt.scatter([p[0] for p in path_points], [p[1] for p in path_points], c='w', s=5)

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
    plt.title('Potential Field and Path')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot([p[0] for p in path_points], [p[1] for p in path_points], 'k-', label='Path')
    plt.scatter([p[0] for p in path_points], [p[1] for p in path_points], c='k', s=5)

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
    plt.title('Path and Obstacles')
    plt.legend()
    plt.show()

