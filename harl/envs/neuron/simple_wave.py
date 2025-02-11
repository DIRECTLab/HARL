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
        self.max_a = self.action_space[0].high[0]
        self.history = []
        self.noise_scale = 0.1

    def _fx(self, x,t):
        return (self.rw[0]*np.sin(x) + self.rw[1]*np.cos(x)  + self.rw[2]*np.sin(t) + self.rw[3]*np.cos(t) + self.rw[4]*np.sin(x*t) + self.rw[5]*np.cos(x*t) + self.rw[6]*np.sin(t**2) + self.rw[7]*np.cos(t**2) + self.rw[8]*np.sin(x**2) + self.rw[9]*np.cos(x**2))/self.rw.sum()

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """

        self.t+=self.time_resolution

        rew = [np.array([self._fx(actions[0][0],self.t)+self.noise_scale*np.random.randn(1)])]

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

        self.rw = (2*np.random.rand(10))-1

        obs = [np.array([self._fx(0,0)])]

        obs+=self.noise_scale*np.random.randn(1)

        self.t=0

        self.history = []

        t = np.arange(0,self.max_t,self.time_resolution)
        
        a = np.arange(0,self.max_a,self.action_resolution)

        self.heatmap = np.zeros((len(a),len(t)))

        for i in range(len(t)):
            for j in range(len(a)):
                self.heatmap[j,i] = self._fx(a[j],t[i])


        self.cmap = colors.LinearSegmentedColormap.from_list(
            name="red_black_green", 
            colors=["red", "black", "green"], 
            N=256
        )

        self.norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        self.max_positions = [
            (i*self.time_resolution, j*self.action_resolution)
            for i in range(self.heatmap.shape[1])
            for j in np.where(self.heatmap[:,i].max()-self.heatmap[:,i] <= self.action_resolution)[0]
        ]

        return obs, obs, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def render(self):
        # plot heatmap of _fx with history

        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.imshow(self.heatmap,cmap=self.cmap,norm=self.norm,extent=[0,self.max_t,0,self.max_a], origin='lower')
        plt.colorbar()
        rows, cols = zip(*self.max_positions)
        plt.scatter(rows,cols,marker='o',color='blue',s=3,alpha=.05)
        plt.title("Simple Wave")
        plt.plot(np.arange(0,self.t,self.time_resolution),np.array(self.history),color='orange')
        plt.show()

    def close(self):
        return None

    def seed(self, seed):
        return None
    
#test render
env = SimpleWave({})
env.reset()
for i in range(100):
    env.step(np.array([[np.cos(i/50)**2]]))
env.render()