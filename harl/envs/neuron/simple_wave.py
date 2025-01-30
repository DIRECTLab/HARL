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
        self.time_resolution = .1
        self.action_resolution = .001
        self.max_t = 10
        self.history = []

    def _fx(self, x,t):
        return np.sin(x*t)
        # return -(x-np.cos(t-x**2)-1)**2

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """

        self.t+=self.time_resolution

        rew = [np.array([self._fx(actions[0][0],self.t)])]

        obs = rew[0][0]

        self.history.append(actions[0][0])

        if self.t>self.max_t:
            done = True
        else:
            done = False

        info = {}

        return [obs], [obs], [[rew]], [done], [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        obs = [np.array([self._fx(0,0)])]

        self.t=0

        self.history = []

        return obs, obs, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def render(self):
        # plot heatmap of _fx with history
        
        t = np.arange(0,self.max_t,self.time_resolution)
        max_a = self.action_space[0].high[0]
        a = np.arange(0,max_a,self.action_resolution)

        heatmap = np.zeros((len(a),len(t)))

        for i in range(len(t)):
            for j in range(len(a)):
                heatmap[j,i] = self._fx(a[j],t[i])


        cmap = colors.LinearSegmentedColormap.from_list(
            name="red_black_green", 
            colors=["red", "black", "green"], 
            N=256
        )

        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        max_positions = [
            (i*self.time_resolution, j*self.action_resolution)
            for i in range(heatmap.shape[1])
            for j in np.where(heatmap[:,i].max()-heatmap[:,i] <= self.action_resolution)[0]
        ]

        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.imshow(heatmap,cmap=cmap,norm=norm,extent=[0,self.max_t,0,max_a], origin='lower')
        plt.colorbar()
        rows, cols = zip(*max_positions)
        plt.scatter(rows,cols,marker='o',color='blue',s=3,alpha=.05)
        plt.title("Simple Wave")
        plt.plot(np.arange(0,self.t,self.time_resolution),np.array(self.history),color='orange')
        plt.show()

    def close(self):
        return None

    def seed(self, seed):
        return None
    
# #test render
# env = SimpleWave({})
# env.reset()
# for i in range(100):
#     env.step(np.array([[np.cos(i/50)**2]]))
# env.render()