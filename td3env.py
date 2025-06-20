import numpy as np
import math
import argparse
from replay_memory import ReplayMemory
import os
import matplotlib.pyplot as plt
import TD3
import torch as th
import td3utils as utils
plt.ioff()


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

# 产生障碍物
def generate_random_obstacles(num_circles, num_rectangles, world_size):
    obstacles = []
    for _ in range(num_circles):
        while True:
            x = np.random.randint(50, world_size[0] - 150)
            y = np.random.randint(50, world_size[1] - 150)
            radius = np.random.randint(20, 40)
            sigma = np.random.randint(50, 100)
            new_obstacle = Obstacle(Position(x, y), sigma=sigma, draw_radius=radius, shape='circle')

            if all(not is_overlapping(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
                obstacles.append(new_obstacle)
                break

    for _ in range(num_rectangles):
        while True:
            x = np.random.randint(50, world_size[0] - 150)
            y = np.random.randint(50, world_size[1] - 150)
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
    # SAC parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    # Set seeds

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 2
    action_dim = 2
    max_action = float(1.0)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    # replay_Buffer for Td3
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
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


    # TD3 control by control
    # td3 param
    max_episode = 1
    # env_observation_space
    # save episode for agents
    EPISODE = []
    Reward = []
    max_steps = 100
    # update initial
    updates = 0
    STEP = []
    for i_episode in range(max_episode):
        # Record trajectory points
        path_points = []
        # initial the agent position
        start_position = Position(15, 60)
        agent = Agent(start_position, scan_radius=10, possible_moves=30)
        episode_reward = 0
        episode_steps = 0
        done = False
        Distance = []
        # time_step 实现的避障的步数
        time_step = 0
        print('start:',(start_position.x , start_position.y))
        while Position.calculate_distance(agent.position, goal.position) > 10:
            # 引入距离列表 distance
            Distance.append(Position.calculate_distance(agent.position, goal.position))
            possible_moves = agent.get_possible_moves()
            min_value = math.inf
            best_move = possible_moves[0]  # Initialize best move to the first possible move
            # Find the optimal move
            # 状态选择 与 终点的相对位置
            state = [agent.position.x -goal.position.x ,agent.position.y -goal.position.y]
            # print(state)
            print('当前agent与终点之间的距离：',Position.calculate_distance(agent.position, goal.position))
            for move in possible_moves:
                # td3 action select
                # dx dy
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                move_value = goal.get_attraction_force(move)
                for obstacle in obstacles:
                    move_value += obstacle.get_repulsion_force(move)

                if move_value < min_value:
                    min_value = move_value
                    best_move = move

            # Set the agent's next position to the best move based TD3
            agent.position.x = best_move.x - action[0]
            agent.position.y = best_move.y - action[1]

            next_state = [agent.position.x -goal.position.x ,agent.position.y -goal.position.y]
            path_points.append((best_move.x, best_move.y))

            if time_step >0 and (Distance[time_step] <Distance[time_step-1]) :
                reward = 0.5*abs(Distance[time_step] - Distance[time_step-1])
            else:
                reward = -0.3 * abs(Distance[time_step] - Distance[time_step - 1])
            episode_reward = episode_reward + reward
            # td3 replay_buffer
            done_bool = float(done)
            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)
            # count time_step
            time_step = time_step + 1
            # Update td3 policy
            if i_episode >1:
                if time_step + sum(STEP) >= args.start_timesteps :
                    policy.train(replay_buffer, args.batch_size)
        # 记录完成步数
        STEP.append(time_step)
        # 采用另一个reward
        if i_episode>0:
            if (STEP[i_episode-1] > STEP[i_episode]):
                episode_reward = episode_reward + 15
            else:
                episode_reward = episode_reward - 15
        else:
                episode_reward = episode_reward + 0
        # save reward
        Reward.append(episode_reward)
        EPISODE.append(i_episode)
        print(f'第{i_episode}回合的累积奖励为：',episode_reward)
        print(f'第{i_episode}回合的endstep数目为：', time_step)
        # Save the coordinates
    start_position = Position(15, 60)
    print('start:', (start_position.x, start_position.y))
    # data_to_save = {
    #     "start_position": (start_position.x, start_position.y),
    #     "goal_position": (goal.position.x, goal.position.y),
    #     "path_points": path_points,
    #     "obstacles": [{
    #         "position": (obs.position.x, obs.position.y),
    #         "shape": obs.shape,
    #         "radius": obs.draw_radius,
    #         "width": obs.width,
    #         "height": obs.height
    #     } for obs in obstacles]
    # }
    #
    # np.save("path_and_obstacles.npy", data_to_save)


    # Plot sac path
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

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, U, levels=50, cmap='viridis')
    plt.colorbar(label='Potential')
    plt.plot([p[0] for p in path_points], [p[1] for p in path_points], 'w-', label='Path')
    plt.scatter([p[0] for p in path_points], [p[1] for p in path_points], c='w', s=5)
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
    # start_position = Position(50, 50)
    plt.scatter(goal.position.x, goal.position.y, c='green', s=100, label='Goal')
    plt.scatter(agent.position.x, agent.position.y, c='red', s=100, label='Agent')
    plt.scatter(start_position.x, start_position.y, c='yellow', s=100, label='Start')  # Mark the start position
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential Field and Path')
    plt.legend()
    plt.show()

    # Plot path and obstacles only (no potential field)
    plt.figure(figsize=(10, 8))
    plt.plot([p[0] for p in path_points], [p[1] for p in path_points], 'k-', label='Path')
    plt.scatter([p[0] for p in path_points], [p[1] for p in path_points], c='k', s=5)
    # 载入obstaacle
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
    # plot xy graph
    plt.scatter(goal.position.x, goal.position.y, c='green', s=100, label='Goal')
    plt.scatter(agent.position.x, agent.position.y, c='red', s=100, label='Agent')
    plt.scatter(start_position.x, start_position.y, c='yellow', s=100, label='Start')  # Mark the start position
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path and Obstacles')
    plt.legend()
    plt.show()
    # save TD3 reward
    plt.figure(figsize=(10, 8))
    plt.plot(EPISODE, Reward,label='reward')
    plt.xlabel('episode')
    plt.ylabel('episode_reward')
    plt.title('episode_reward for sac')
    plt.legend()
    plt.show()
    # save Td3 轨迹点
    np.save('path_points_td3_7.npy', path_points)
    # Td3 EPISODE REWARD
    # np.save('Rewardtd3_c.npy',Reward)
    # np.save('EPISODEtd3_c.npy', EPISODE)

    # # save actor network
    # policy.save('td3model/td3_2')

