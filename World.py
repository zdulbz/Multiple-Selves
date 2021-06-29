import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dist

# from scipy.stats import entropy
# from scipy.spatial import distance


class World(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_actions=5,
                 size=9,
                 bounds=4,
                 modes=3,
                 field_of_view=3,
                 stat_types=[0, 0, 0, 0],
                 variance=1,
                 radius=2,
                 offset=0,
                 circle=True,
                 stationary=True,
                 ep_length=50,
                 resource_percentile=50,
                 rotation_speed=0.01,
                 reward_type='sq_dev'):

        super(World, self).__init__()
        self.size = size  # grid size
        self.stat_types = stat_types  # list of stat types, 0 = symmetric, starts at 0.5, auto-increase, 1 = capped, stochastic loss, rest-increase
        self.n_stats = len(self.stat_types)  # number of features (one for each stat)
        self.fov = field_of_view  # field of view of agent
        self.view_size = 2 * self.fov + 1
        self.bounds = bounds  # range of means for resource patches
        self.modes = modes  # number of resource patches per stat
        self.ep_length = ep_length
        self.history = []  # history for visualization
        self.set_point = 1  # goal of stats
        self.stationary = stationary
        self.border = -0.02  # colour of border and of agent on grid
        self.resource_percentile = resource_percentile  # resources affect stat above this
        self.thresholds = []  # list of thresholds for each resource
        self.multiplier = modes  # multiplies the resource map by fixed amount to keep peaks relatively constant
        self.well_being_range = 0.2
        self.name = 'ResourceWorld'
        self.reward_type = reward_type
        self.variance = variance
        self.radius = radius
        self.offset = offset
        self.circle = circle
        self.rotation_speed = rotation_speed

        self.action_space = spaces.Discrete(n_actions)  # action space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_stats * self.view_size ** 2 + self.n_stats,))

        self.action_dim = n_actions
        self.state_dim = self.n_stats * self.view_size ** 2 + self.n_stats

        self.grid = np.zeros((self.n_stats, self.size, self.size))  # empty world grid
        # self.grid = np.random.choice([0,1],(n_stats,self.size,self.size),p=[0.8,0.2]) # p gives percent of 1s

        self.p, self.q = 4, 4  # homeostatic RL exponents for reward function

        self.stat_decrease_per_step = 0.005  # stat decreases over time
        self.stat_decrease_per_injury = 0.05  # stat decrease per danger (not in current version)
        self.stat_increase_per_recover = 0.1  # stat increases if criteria (not in current version)
        self.clumsiness = 0.2
        self.dead = False  # dead if stat hits 0 (just a read out)

        self.make_stats()  # creates stat variable self.stats
        self.reset_grid()
        self.reset()  # resets world, stats, and randomizes location of agent

    ## Reset function

    def reset(self):
        if not self.stationary: self.reset_grid()  # if changing world on every episode
        self.make_stats()
        self.time_step = 0  # resets step timer
        self.dead = False
        self.done = False
        self.history = []
        self.new = []
        # self.stats = copy.deepcopy(self.initial_stats) # puts stats back to initial values
        self.loc = [np.random.randint(self.fov, self.size - self.fov),
                    np.random.randint(self.fov, self.size - self.fov)]  # initialize state randomly
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]  # gets initial agent view
        return self.get_state()

    def reset_grid(self):
        self.thresholds = []
        for stat in range(self.n_stats):
            self.grid[stat, :, :] = self.multiplier * self.get_n_patches(grid_size=self.size, bounds=self.bounds,
                                                                         modes=self.modes,
                                                                         var=self.variance)  # makes gaussian patches
            self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        if self.circle:
            self.thresholds = []
            if not self.stationary:
                self.offset += np.random.uniform(0, 2 * np.pi)  # self.rotation_speed #np.random.uniform(0,2*np.pi)
                self.radius = np.random.uniform(1, 2)
            self.grid = self.get_n_resources(grid_size=self.size, bounds=self.bounds, resources=self.n_stats,
                                             radius=self.radius, offset=self.offset, var=self.variance)
            for stat in range(self.n_stats):
                self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        self.grid[:, [0, -1], :] = self.grid[:, :, [0, -1]] = self.border  # make border

    ## Step function

    def step(self, action):
        # Execute one time step within the environment
        self.time_step += 1
        self.set_lowest_stat()

        if self.time_step == self.ep_length: self.done = True

        if action == 0 and self.loc[0] < self.size - self.fov - 1:
            self.loc[0] += 1

        elif action == 1 and self.loc[0] > self.fov + 1:
            self.loc[0] -= 1

        elif action == 2 and self.loc[1] < self.size - self.fov - 1:
            self.loc[1] += 1

        elif action == 3 and self.loc[1] > self.fov + 1:
            self.loc[1] -= 1

        reward = self.step_stats()
        module_rewards = self.separate_squared_rewards()  # self.separate_HRRL_rewards(10)
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]
        self.history.append((self.grid_with_agent(), self.view, copy.deepcopy(self.stats)))

        return self.get_state(), reward, self.done, module_rewards  # self.dead

    ##### Functions involved in making stats, stepping stats and getting HRRL Rewards

    def make_stats(self):
        self.stats = []
        for stat in self.stat_types:
            if stat == 0: self.stats.append(np.random.uniform(0.5, 0.7))
            if stat == 1: self.stats.append(np.random.uniform(0.7, 0.9))

        self.lowest_stat = np.min(self.stats)

        # self.initial_stats = copy.deepcopy(self.stats)

    def set_lowest_stat(self):
        if np.min(self.stats) < self.lowest_stat:
            self.lowest_stat = np.min(self.stats)

    def step_stats(self):
        self.old_stats = copy.deepcopy(self.stats)

        for i in range(self.n_stats):
            self.stats[i] -= self.stat_decrease_per_step

            if self.stat_types[i] == 0:  # for food/water stats
                # if  self.grid[i,self.loc[0],self.loc[1]] == 1: self.stats[i] += self.stat_increase_per_recover
                if self.grid[i, self.loc[0], self.loc[1]] > self.thresholds[i]:
                    self.stats[i] += self.grid[i, self.loc[0], self.loc[1]]

            if self.stat_types[i] == 1:  # damage stats
                if self.grid[i, self.loc[0], self.loc[1]] == 1 and np.random.uniform() < self.clumsiness: self.stats[
                    i] -= self.stat_decrease_per_injury  # stochastic damage
                if self.action == 4: self.stats[i] += self.stat_increase_per_recover  # for rest action
                if self.stats[i] > 1: self.stats[i] = 1

            if self.stats[i] < 0:
                # self.stats[i] = 0
                self.dead = True

        if self.reward_type == 'HRRL': return self.HRRL_reward(100)
        if self.reward_type == 'sq_dev': return self.sq_dev_reward()
        if self.reward_type == 'well_being': return self.well_being_reward()
        if self.reward_type == 'lin_dev': return self.lin_dev_reward()
        if self.reward_type == 'min_sq': return self.min_sq()

    def HRRL_reward(self, scaling):
        return scaling * (self.get_cost_surface(self.old_stats) - self.get_cost_surface(
            self.stats))  # - 100*any([x<0.05 for x in self.stats])

    def sq_dev_reward(self):
        return 0.4 + sum([-np.abs(self.set_point - stat) ** 2 for stat in self.stats])

    def lin_dev_reward(self):
        return sum([-np.abs(self.set_point - stat) for stat in self.stats])

    def separate_linear_rewards(self):
        return [0.1 - -np.abs(self.set_point - stat) for stat in self.stats]

    def separate_squared_rewards(self):
        return [0.1 - np.abs(self.set_point - stat) ** 2 for stat in self.stats]

    def separate_HRRL_rewards(self, scaling):

        return [scaling * (((self.set_point - old_stat) ** self.p) ** 1 / self.q - (
                    (self.set_point - new_stat) ** self.p) ** 1 / self.q) for old_stat, new_stat in
                zip(self.old_stats, self.stats)]

    def well_being_reward(self):
        return 1 if all(
            [stat > self.set_point - self.well_being_range and stat < self.set_point + self.well_being_range for stat in
             self.stats]) else -1

    def min_sq(self):
        return min([-np.abs(self.set_point - stat) ** 2 for stat in self.stats])

    def get_cost_surface(self, stats):
        return sum([(self.set_point - stat) ** self.p for stat in stats]) ** 1 / self.q

    def get_state(self):
        return torch.cat((torch.tensor(self.stats), torch.tensor(self.view.flatten()))).float()

    def grid_with_agent(self):
        temp = copy.copy(self.grid)
        temp[:, self.loc[0], self.loc[1]] = self.border  # self.grid[:,self.state[0],self.state[1]]
        # temp[:2][temp[:2]>self.resource_threshold] = 2
        return temp

    def render(self):
        tits = [f'stat {i + 1}' for i in range(self.n_stats)]
        for i in range(self.n_stats):
            plt.subplot(100 + 10 * self.n_stats + 1 + i)
            plt.title(tits[i])
            plt.imshow(self.grid_with_agent()[i])
            # plt.show()

    ###### methods for distributing resources #####



    def multivariate_gaussian(self,grid_size=20, bounds=4, height=1, m1=None, m2=None, Sigma=None):
        """Return the multivariate Gaussian distribution on array pos."""

        # Our 2-dimensional distribution will be over variables X and Y
        X = np.linspace(-bounds, bounds, grid_size)
        Y = np.linspace(-bounds, bounds, grid_size)
        X, Y = np.meshgrid(X, Y)

        # Mean vector and covariance matrix
        if m1 is None:  # if not specifying means
            m1 = np.random.uniform(-bounds, bounds)
            m2 = np.random.uniform(-bounds, bounds)
        mu = np.array([m1, m2])

        if Sigma is None: Sigma = dist.make_spd_matrix(2)  # if not specifying covariance matrix, pick randomly

        height = np.random.uniform(1, height)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return height * self.norm(np.exp(-fac / 2) / N)

    def norm(self,z):
        return z / z.sum()

    def get_n_patches(self,grid_size=20, bounds=4, modes=3, radius=None, var=2):
        # The distribution on the variables X, Y packed into pos.
        z = np.zeros((grid_size, grid_size))
        sigma = [[var, 0], [0, var]]

        if radius is None:  # randomly disperse
            for i in range(0, modes):
                z += self.multivariate_gaussian(grid_size=grid_size, bounds=bounds, Sigma=sigma)
            return z

        points = self.get_spaced_means(modes, radius)
        for i in range(0, modes):
            z += self.multivariate_gaussian(grid_size=grid_size, bounds=bounds, m1=points[i][0], m2=points[i][1])
        return z

    def get_n_resources(self,grid_size=20, bounds=4, resources=3, radius=None, offset=0, var=1):
        # The distribution on the variables X, Y packed into pos.
        z = np.zeros((resources, grid_size, grid_size))

        sigma = [[var, 0], [0, var]]

        if radius is None:  # randomly disperse
            for i in range(0, resources):
                z[i, :, :] = self.multivariate_gaussian(grid_size=grid_size, bounds=bounds)
            return z

        points = self.get_spaced_means(resources, radius, offset)
        for i in range(resources):
            z[i, :, :] = self.multivariate_gaussian(grid_size=grid_size, bounds=bounds, m1=points[i][0], m2=points[i][1],
                                                    Sigma=sigma)
        return z

    def get_spaced_means(self,modes, radius, offset):
        radians_between_each_point = 2 * np.pi / modes
        list_of_points = []
        for p in range(0, modes):
            node = p + offset
            list_of_points.append((radius * np.cos(node * radians_between_each_point),
                                   radius * np.sin(node * radians_between_each_point)))
        return list_of_points
