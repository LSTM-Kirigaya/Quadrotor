import parl
from parl import layers

# 策略网络
class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act="relu")
        self.fc2 = layers.fc(size=act_dim, act="tanh")  # 激活函数设为tanh将值限定在[-1, 1]之间

    def policy(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out

# Q网络
class CriticModel(parl.Model):
    def __init__(self):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act="relu")
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 因为s和a都是参数，神经网络中对于多向量输入可以使用联级的方法输入
        # 所以我们先把它们拼起来
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q

# 我们将策略网络和Q网络的类整合到一个无人机类里面去
class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    # 获取策略网络的网络参数
    def get_actor_params(self):
        return self.actor_model.parameters()