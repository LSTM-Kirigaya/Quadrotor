import parl
from parl import layers
import paddle.fluid as fluid
from copy import deepcopy

class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):

        """
        :param model: 整合了策略网络和Q网络的前向网络
        :param gamma: reward的衰减因子
        :param tau: 软更新参数
        :param actor_lr: 策略网络的学习率
        :param critic_lr: Q网络的学习率
        """

        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        # 使用self.model的actor model来预测动作
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        # DDPG算法更新两个网络
        actor_cost = self.actor_learn(obs)
        critic_cost = self.critic_learn(obs, action, reward, next_obs, terminal)

        return actor_cost, critic_cost

    def actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        cost = layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def critic_learn(self, obs, action, reward, next_obs, terminal):
        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)

        terminal = layers.cast(terminal, dtype="float32")
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        Q = self.model.value(obs, action)
        cost = layers.square_error_cost(Q, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor
        )















