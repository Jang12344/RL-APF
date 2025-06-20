import numpy as np
import random
import matplotlib.pyplot as plt
from apffinal2 import Position, Agent, Goal, Obstacle

# Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.3  # 增加变异率
CROSSOVER_RATE = 0.7
MAX_ATTEMPTS = 10  # 最大尝试次数


# Fitness function with obstacle avoidance
def fitness(path, goal_position, obstacles):
    total_distance = 0

    for i in range(len(path) - 1):
        total_distance += Position.calculate_distance(path[i], path[i + 1])

        # Penalize collisions
        for obstacle in obstacles:
            if obstacle.shape == 'circle':
                if Position.calculate_distance(path[i], obstacle.position) <= obstacle.draw_radius:
                    return float('inf')  # 碰撞时，返回极大的惩罚
            elif obstacle.shape == 'rectangle':
                if (obstacle.position.x - obstacle.width / 2 <= path[
                    i].x <= obstacle.position.x + obstacle.width / 2 and
                        obstacle.position.y - obstacle.height / 2 <= path[
                            i].y <= obstacle.position.y + obstacle.height / 2):
                    return float('inf')  # 碰撞时，返回极大的惩罚

    # Add distance to goal
    total_distance += Position.calculate_distance(path[-1], goal_position)

    return total_distance


# Generate initial population with diversity
def generate_population(start_position, goal_position, world_size):
    population = []
    for _ in range(POPULATION_SIZE):
        path = [start_position]
        for _ in range(random.randint(5, 20)):  # 随机点数的路径，增加多样性
            path.append(Position(random.randint(0, world_size[0]), random.randint(0, world_size[1])))
        path.append(goal_position)
        population.append(path)
    return population


# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


# Mutation with local adjustment
def mutate(path, world_size):
    if random.random() < MUTATION_RATE:
        mutation_point = random.randint(1, len(path) - 2)
        path[mutation_point] = Position(random.randint(0, world_size[0]), random.randint(0, world_size[1]))
        # 进一步随机调整，进行局部微调
        if random.random() < 0.5:
            path[mutation_point].x = max(0, min(world_size[0], path[mutation_point].x + random.randint(-10, 10)))
        if random.random() < 0.5:
            path[mutation_point].y = max(0, min(world_size[1], path[mutation_point].y + random.randint(-10, 10)))


# Genetic Algorithm with multiple attempts
def genetic_algorithm(start_position, goal_position, obstacles, world_size):
    best_path = None
    best_fitness = float('inf')

    for attempt in range(MAX_ATTEMPTS):
        population = generate_population(start_position, goal_position, world_size)
        for generation in range(GENERATIONS):
            # Calculate fitness for each path
            fitness_scores = [(fitness(path, goal_position, obstacles), path) for path in population]
            fitness_scores.sort(key=lambda x: x[0])

            # Select the best paths
            selected_population = [path for _, path in fitness_scores[:POPULATION_SIZE // 2]]

            # Create the next generation
            next_population = []
            while len(next_population) < POPULATION_SIZE:
                parent1, parent2 = random.sample(selected_population, 2)
                if random.random() < CROSSOVER_RATE:
                    child = crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])
                mutate(child, world_size)
                next_population.append(child)

            population = next_population

            # Update best path
            if fitness_scores[0][0] < best_fitness:
                best_fitness = fitness_scores[0][0]
                best_path = fitness_scores[0][1]

            # Print the best fitness in the generation
            print(f"Attempt {attempt}, Generation {generation}: Best Fitness = {fitness_scores[0][0]}")

        # Check if the path is valid (no collisions)
        if fitness_scores[0][0] < float('inf'):
            break  # 找到有效路径，退出循环

    return best_path


# Define world, start, goal, and obstacles
world_size = (700, 200)
start_position = Position(15, 60)
goal_position = Position(600, 20)
data_loaded = np.load("path_and_obstacles_s_shape7.npy", allow_pickle=True).item()

obstacles = []
for obs_data in data_loaded["obstacles"]:
    position = Position(*obs_data["position"])
    sigma = obs_data.get("sigma", 0.5)  # 默认sigma
    obstacle = Obstacle(
        position=position,
        sigma=sigma,  # Include sigma
        draw_radius=obs_data["radius"],
        shape=obs_data["shape"],
        width=obs_data["width"],
        height=obs_data["height"]
    )
    obstacles.append(obstacle)

# Run Genetic Algorithm
best_path = genetic_algorithm(start_position, goal_position, obstacles, world_size)

# Save the best path
if best_path:
    path_points = [(pos.x, pos.y) for pos in best_path]
    np.save('path_points_ga_7.npy', path_points)

    # Visualize the best path
    x_coords, y_coords = zip(*path_points)
    plt.plot(x_coords, y_coords, 'r-', label='GA Path')
    plt.scatter(x_coords, y_coords, c='r', s=5)
    plt.scatter(goal_position.x, goal_position.y, c='green', s=100, label='Goal')
    plt.scatter(start_position.x, start_position.y, c='blue', s=100, label='Start')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Genetic Algorithm Path Planning')
    plt.legend()
    plt.show()
else:
    print("未能找到有效的路径。")
