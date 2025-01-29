from gym import spaces
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

class SimpleWave:
    def __init__(self, args):
        self.n_agents = 1
        self.share_observation_space = [spaces.Box(low=-1.0,high=1.0,shape=(1,),dtype=np.float32) for _ in range(self.n_agents)]
        self.observation_space = [spaces.Box(low=-1.0,high=1.0,shape=(1,),dtype=np.float32) for _ in range(self.n_agents)]
        self.action_space = [spaces.Box(low=0,high=2.0,shape=(1,),dtype=np.float32) for _ in range(self.n_agents)]

        self.discrete = False
        self.t=0
        self.dt = .1
        self.max_t = 10
        self.history = []

    def _fx(self, x):
        return np.sin(x)

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """

        self.t+=self.dt

        obs = [np.array([self._fx(actions[0][0]*self.t)])]

        rew = obs[0][0]

        self.history.append(actions[0][0])

        if self.t>self.max_t:
            done = True
        else:
            done = False

        info = {}

        return [obs], [obs], [[rew]], [done], [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        obs = [np.array([self._fx(0)])]

        self.t=0

        self.history = []

        return obs, obs, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def render(self):
        # plot heatmap of _fx with history
        
        t = np.arange(0,self.max_t,self.dt)
        max_a = self.action_space[0].high[0]
        a = np.arange(0,max_a,.001)

        heatmap = np.zeros((len(a),len(t)))

        for i in range(len(t)):
            for j in range(len(a)):
                heatmap[j,i] = self._fx(a[j]*t[i])


        cmap = colors.LinearSegmentedColormap.from_list(
            name="red_black_green", 
            colors=["red", "black", "green"], 
            N=256
        )

        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        max_positions = heatmap.argmax(axis=0)

        
        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.imshow(heatmap,cmap=cmap,norm=norm,extent=[0,self.max_t,0,max_a])
        plt.colorbar()
        plt.title("Simple Wave")
        plt.plot(np.arange(0,self.t,self.dt),np.array(self.history),color='blue')
        plt.show()

    def close(self):
        return None

    def seed(self, seed):
        raise NotImplementedError