import parl
import numpy as np
from parl.utils import action_mapping
from parl.utils import ReplayMemory
from parl.utils import logger
import os
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from Model import QuadrotorModel
from Algorithm import DDPG
from Agent import QuadrotorAgent



ACTOR_LR = 0.00002   # Actor网络更新的 learning rate
CRITIC_LR = 0.0001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward
gamma = 0.2

def run_episode(env, agent, rpm):
    total_reward, steps = 0, 0
    obs = env.reset()

    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype("float32"))
        action = np.squeeze(action)

        # 增加高斯噪音，并clip
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        # 将动作映射到对应的电压区间
        action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

        # 让四个电压值靠近，这样有意引导容易收敛
        means = np.mean(action)
        action = action + gamma * (means - action)

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

            obs = next_obs
            total_reward += reward

        if done:
            break
    return total_reward, steps

# 测试五个episode，输出reward的平均值
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs)
            action = np.clip(action, -1.0, 1.0)
            action = np.squeeze(action)
            action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

            means = np.mean(action)
            action = action + gamma * (means - action)

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

env = make_env("Quadrotor", task="hovering_control")
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 创建三层模型的实例
model = QuadrotorModel(act_dim)
algorithm = DDPG(model=model,
                 gamma=GAMMA,
                 tau=TAU,
                 actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm ,obs_dim, act_dim)

# 直接使用parl内置的ReplayMemory
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

save_path = "./model.ckpt"
# 加载模型
#if os.path.exists(save_path):
#    agent.restore(save_path)

test_flag = 0
total_steps = 0
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm)
    total_steps += steps
    # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

    if total_steps // TEST_EVERY_STEPS >= test_flag:  # 每隔一定step数，评估一次模型
        while total_steps // TEST_EVERY_STEPS >= test_flag:
            test_flag += 1

        evaluate_reward = evaluate(env, agent)
        logger.info('Steps {}, Test reward: {}'.format(
            total_steps, evaluate_reward))  # 打印评估的reward

        # 每评估一次，就保存一次模型，以训练的step数命名
        ckpt = 'model_dir/steps_{}.ckpt'.format(total_steps)
        agent.save(save_path)