import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.distributions import Categorical
# import sklearn.datasets as dist

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, view, hidden_dim, blind=False):
        """Initialization."""
        super(Network, self).__init__()
        self.h = hidden_dim
        self.blind = blind
        if self.blind: self.mask = nn.Parameter(0.1 + torch.zeros(1,in_dim)) # This is the mask (it is just an array initialized with all 0.1s)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if self.blind: x = x * self.mask
        return self.layers(x)


class DQN_Agent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
            lr: float = 0.001,
            hidden_size=512,
            switch_episode=10000,
            blind = False
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        # obs_dim = env.observation_space.shape[0]

        self.env = env
        self.n_statedims = self.env.n_statedims
        self.action_dim = self.env.n_actions
        self.memory = ReplayBuffer(self.n_statedims, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.channels = self.n_statedims  # self.env.n_stats # obs_dim
        self.lr = lr
        self.switch_episode = switch_episode
        self.plotting = True
        self.value_map_history = []
        self.q_map_history = []
        self.plotting_interval = 100
        self.is_modular = False
        self.store_value_maps = False
        self.hidden_size = hidden_size
        self.attention_agent = False
        self.clamp_to = 0
        self.updates_per_step = 1
        self.blind = blind
        self.lambd = 0.0001

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(self.channels, self.action_dim, self.env.view_size, self.hidden_size,blind=self.blind).to(self.device)
        self.dqn_target = Network(self.channels, self.action_dim, self.env.view_size, self.hidden_size,blind=self.blind).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            # selected_action = self.env.action_space.sample()
            selected_action = np.random.randint(self.action_dim)
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done, _

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        for i in range(self.updates_per_step):
          samples = self.memory.sample_batch()

          loss = self._compute_dqn_loss(samples)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        self.cumulative_deviation = 0
        self.maximum_deviation = 0

        state = self.env.reset().unsqueeze(0)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        self.rewards = []
        self.stats = []
        score = 0
        self.env.clamp = 0

        for frame_idx in range(1, num_frames + 1):

            if frame_idx >= self.switch_episode:
                # self.env.change_first_resource_map() # introduces a change in the environment
                # self.epsilon = 1
                self.env.stats[-1] = self.clamp_to
                self.env.clamp = 1

            if self.attention_agent:
                self.current_task = np.argmax(np.abs([s - 5 for s in self.env.stats]))
                # torch.cat((state,torch.tensor(self.current_task).float().unsqueeze(0).unsqueeze(0)),1)

            action = self.select_action(state)
            next_state, reward, done, _ = self.step(action)

            state = next_state.unsqueeze(0)
            score += reward

            if self.attention_agent:
                self.rewards.append(_[self.current_task])  # attention agent
            else:
                self.rewards.append(reward)  # regular agent

            self.stats.append(copy.deepcopy(self.env.stats))

            if self.store_value_maps: self.get_v_map(display=False)

            self.cumulative_deviation += sum([abs(stat - 1) for stat in self.env.stats])
            for stat in self.env.stats:
                if abs(stat) > self.maximum_deviation: self.maximum_deviation = abs(stat)

            # if episode ends
            if done:
                state = self.env.reset().unsqueeze(0)
                scores.append(score)
                score = 0

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            epsilons.append(self.epsilon)

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % self.plotting_interval == 0 and self.plotting:
                self._plot(frame_idx, scores, losses, epsilons)
                self.env.render()
                plt.show()


    def test(self, timesteps=50, display=False) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        actions = ['down', 'up', 'right', 'left', 'eat', 'drink', 'rest']
        state = self.env.reset().unsqueeze(0)
        done = False
        score = 0
        t = 0
        cumulative_deviation = 0
        maximum_deviation = 0

        frames = []
        while t < timesteps:
            t += 1
            # frames.append(self.env.render())
            action = self.select_action(state)
            next_state, reward, done, info = self.step(action)
            if display:
                print('Timestep:', t,
                      ', reward:', np.round(reward, 3),
                      ', modular reward:', np.round(np.array(info), 3),
                      ', Action: ', actions[action],
                      ', stats: ', np.round(self.env.stats, 2),
                      ', Dead?: ', self.env.dead,
                      ', Lowest: ', np.round(self.env.lowest_stat, 2))
                self.env.render()
                plt.show()

            state = next_state.unsqueeze(0)
            score += reward

            cumulative_deviation += sum([abs(stat - 1) for stat in env.stats])
            for stat in env.stats:
                if abs(stat) > maximum_deviation: maximum_deviation = abs(stat)

        print("score: ", score)

        return score, cumulative_deviation, maximum_deviation

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)

        next_q_value = self.dqn_target(next_state).gather(  # Double DQN
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()

        # next_q_value = self.dqn_target(
        #     next_state
        # ).max(dim=1, keepdim=True)[0].detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        if self.blind:
            L1_loss = self.lambd*torch.sum(torch.abs(self.dqn.mask)) # this is the 'penalty' for having numbers other than 0
            loss = F.smooth_l1_loss(curr_q_value, target) + L1_loss
        else:
          loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def get_q_map(self, stats_force=None, display=True):
        q_all = np.zeros((self.action_dim, self.env.size, self.env.size))

        move_back_loc = copy.copy(self.env.loc)
        actions = ['down', 'up', 'right', 'left']
        ## will take environment and agent as parameters
        for i in range(1, self.env.size - 1):
            for j in range(1, self.env.size - 1):

                self.env.move_location([i, j])
                state = self.env.get_state()
                if stats_force is not None: state[:self.env.n_stats] = stats_force
                with torch.no_grad():
                    q_vals = self.dqn(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
                # q_vals = (q_vals - np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) # normalizing step
                q_vals = (q_vals == np.max(q_vals)).astype(int)  # sets top q-value to 1
                q_all[:, i, j] = q_vals

        self.env.move_location(move_back_loc)

        if display:
            fig, axes = plt.subplots(1, self.action_dim, figsize=(8, 2))
            for i, ax in enumerate(axes):
                ax.imshow(q_all[i])
                ax.set_title(f'Q-{actions[i]}')
            plt.show()

        if not display:
            self.q_map_history.append(q_all)

    def get_v_map(self, stats_force=None, display=True):
        v_map = np.zeros((self.env.size, self.env.size))

        move_back_loc = copy.copy(self.env.loc)
        ## will take environment and agent as parameters
        for i in range(1, self.env.size - 1):
            for j in range(1, self.env.size - 1):

                self.env.move_location([i, j])
                state = self.env.get_state()
                if stats_force is not None: state[:self.env.n_stats] = stats_force
                with torch.no_grad():
                    q_vals = self.dqn(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
                # q_vals = (q_vals - np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) # normalizing step
                v_map[i, j] = np.max(q_vals)

        self.env.move_location(move_back_loc)

        if display:
            self.display_grid_with_text(v_map[1:-1, 1:-1])
            # fig, ax = plt.subplots(figsize=(5,5))
            # ax.imshow(v_map[1:-1,1:-1])
            # plt.show()

        if not display:
            self.value_map_history.append(v_map[1:-1, 1:-1])

    def display_grid_with_text(self, grid):
        grid = np.round(grid, 2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid)
        for (j, i), label in np.ndenumerate(grid):
            ax.text(i, j, label, ha='center', va='center', fontsize=12, fontweight='bold', color='r')
            ax.set_yticks([])
            ax.set_xticks([])
        plt.show()

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(self.rewards)
        # plt.ylim(-5,10)
        # plt.subplot(142)
        # plt.title(f'loss, stats: {np.round(self.env.stats,2)}')
        # plt.plot(losses)
        plt.subplot(132)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.subplot(133)
        plt.title('stats')
        plt.plot(self.stats)
        plt.ylim([-10, 10])
        plt.legend([f'stat {i + 1}: {np.round(self.env.stats[i], 2)}' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Stat levels')
        plt.show()
        self.get_v_map()


#### random agent #####


# def test_random_agent(env, timesteps = 50):
#         state = env.reset().unsqueeze(0)
#         score = 0
#         t = 0

#         while t < timesteps:
#             action = env.action_space.sample()
#             next_state, reward, done, info = self.step(action)
#             state = next_state.unsqueeze(0)
#             score += reward
#             t+=1

#         return score

#### random agent #####


# def test_random_agent(env, timesteps = 50):
#         state = env.reset().unsqueeze(0)
#         score = 0
#         t = 0

#         while t < timesteps:
#             action = env.action_space.sample()
#             next_state, reward, done, info = self.step(action)
#             state = next_state.unsqueeze(0)
#             score += reward
#             t+=1

#         return score

class Modular_DQN_Agent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
            lr: float = 0.001,
            explore=1000,
            arbitrator_lr=1e-3,
            hidden_size=512,
            decision_process='gmQ',
            blind=False,
            switch_episode=10000
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        # obs_dim = env.observation_space.shape[0]

        self.env = env
        self.n_statedims = self.env.n_statedims
        self.action_dim = self.env.n_actions
        self.numModules = env.n_stats
        self.memories = [ReplayBuffer(self.n_statedims, memory_size, batch_size) for i in range(self.numModules)]
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.channels = self.n_statedims  # self.env.n_stats # obs_dim
        self.lr = lr
        self.arbitrator_lr = arbitrator_lr
        self.current_module = 0
        self.explore_until = explore
        self.switch_episode = switch_episode
        self.plotting = True
        self.plotting_interval = 100
        self.value_map_history = []
        self.is_modular = True
        self.store_value_maps = False
        self.hidden_size = hidden_size
        self.decision_process = decision_process
        self.blind = blind
        self.lambd = 0.01
        self.clamp_to = 0
        self.updates_per_step = 10

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.modules = [
            Network(self.channels, self.action_dim, self.env.view_size, self.hidden_size, blind=self.blind).to(
                self.device) for i in range(self.numModules)]
        self.targets = [
            Network(self.channels, self.action_dim, self.env.view_size, self.hidden_size, blind=self.blind).to(
                self.device) for i in range(self.numModules)]
        # self.arbitrator = arbitrator_ceo(obs_dim, self.numModules, self.arbitrator_lr).to(self.device)
        self.arbitrator = arbitrator_ceo(self.numModules, self.action_dim, self.arbitrator_lr).to(self.device)

        for i in range(self.numModules):
            self.targets[i].load_state_dict(self.modules[i].state_dict())
            self.targets[i].eval()

            # optimizer
        self.optimizers = [optim.Adam(modules.parameters(), lr=self.lr) for modules in self.modules]

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))


    def select_action(self, state: np.ndarray, frame) -> np.ndarray:
        """Select an action from the input state."""
        # modular policy
        # if frame % 7 == 0: self.current_module +=1
        # if self.current_module > self.numModules - 1: self.current_module = 0
        m = nn.Softmax(dim=0)

        prepared_state = torch.FloatTensor(state).to(self.device)

        if self.epsilon > np.random.random() and not self.is_test:
            # selected_action = self.env.action_space.sample()
            selected_action = np.random.randint(self.action_dim)
        else:
            if self.decision_process == 'allQ':
                ModuleQValues = np.array(
                    [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)])
                ind = np.unravel_index(np.argmax(ModuleQValues, axis=None), ModuleQValues.shape)
                selected_action = ind[-1]
            # meanQs = np.sum(ModuleQValues,axis=1)
            # unhappy_module = np.argmin(meanQs)
            # finalQValues = ModuleQValues[unhappy_module] #unhappy takes control
            if self.decision_process == 'gmQ':
                finalQValues = sum(self.modules[i](prepared_state) for i in range(self.numModules))  ## This is GmQ
                # finalQValues = sum(self.modules[i](prepared_state)/torch.max(self.modules[i](prepared_state)) for i in range(self.numModules)) # centered GmQ
                # finalQValues = sum(m(self.modules[i](prepared_state)) for i in range(self.numModules)) ## pre-softmax to avoid domination
                selected_action = finalQValues.argmax().detach().cpu().numpy()

            if self.decision_process == 'variance':
                ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
                variances = [np.var(qvals) for qvals in ModuleQValues]
                ind = np.argmax(variances)
                selected_action = ModuleQValues[ind].argmax()

            if self.decision_process == 'voting':
                action_slots = np.zeros(4)
                ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
                ModuleActions = [np.argmax(qvals) for qvals in ModuleQValues]
                variances = [np.var(qvals) for qvals in ModuleQValues]
                for i in range(self.numModules):
                    action_slots[ModuleActions[i]] += variances[i]
                ind = np.argmax(action_slots)
                selected_action = ModuleQValues[ind].argmax()

            if self.decision_process == 'max_abs':
                ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
                max_abs = [np.sum(np.abs(qvals)) for qvals in ModuleQValues]
                ind = np.argmax(max_abs)
                selected_action = ModuleQValues[ind].argmax()

            if self.decision_process == 'CEO':
                probs = self.arbitrator(prepared_state[:, :self.numModules])  # this is arbiQ
                m = Categorical(probs)
                arb_action = m.sample()
                action = torch.argmax(probs)
                self.arbitrator.saved_log_probs.append(m.log_prob(arb_action))
                finalQValues = self.modules[arb_action.item()](prepared_state)
                selected_action = finalQValues.argmax().detach().cpu().numpy()

            if self.decision_process == 'stat_weighted':
                weights = [abs(self.env.set_point - stat) for stat in self.env.stats]
                weights = self.softmax(weights)
                finalQValues = sum(weights[i]*self.modules[i](prepared_state) for i in range(self.numModules)) ## This is GmQ
                selected_action = finalQValues.argmax().detach().cpu().numpy()


        # if frame < self.explore_until:
        #   selected_action = np.random.randint(0, self.action_dim)

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, sep_rewards = self.env.step(action)

        if not self.is_test:
            for i in range(self.numModules):
                transition = self.transition + [sep_rewards[i], next_state, done]
                self.memories[i].store(*transition)

        return next_state, reward, done, sep_rewards

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        for j in range(self.updates_per_step):

            losses = self._compute_dqn_loss()
            for i in range(self.numModules):
                self.optimizers[i].zero_grad()
                losses[i].backward()
                self.optimizers[i].step()

        return losses

    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        self.cumulative_deviation = 0
        self.maximum_deviation = 0

        state = self.env.reset().unsqueeze(0)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        self.rewards = []
        self.mod_rewards = []
        self.stats = []
        score = 0
        self.env.clamp = 0

        for frame_idx in range(1, num_frames + 1):

            if frame_idx > self.switch_episode:
                # self.env.change_first_resource_map() # introduces a change in the environment
                # self.epsilon = 1
                self.env.stats[-1] = self.clamp_to
                self.env.clamp = 1

            action = self.select_action(state, frame_idx)
            next_state, reward, done, mod_rewards = self.step(action)
            if self.decision_process == 'CEO': self.arbitrator.rewards.append(reward)

            state = next_state.unsqueeze(0)
            score += reward
            self.rewards.append(reward)
            self.mod_rewards.append(mod_rewards)
            self.stats.append(copy.deepcopy(self.env.stats))

            if self.store_value_maps: self.get_v_map(
                display=False)  # and 9950 < frame_idx < 10200: self.get_v_map(display=False)

            self.cumulative_deviation += sum([abs(stat - 1) for stat in self.env.stats])
            for stat in self.env.stats:
                if abs(stat) > self.maximum_deviation: self.maximum_deviation = abs(stat)

            # if episode ends
            if done:
                state = self.env.reset().unsqueeze(0)
                scores.append(score)
                score = 0
                if self.decision_process == 'CEO': self.arbitrator.update(self.device)

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            epsilons.append(self.epsilon)

            # if training is ready
            if len(self.memories[0]) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % self.plotting_interval == 0 and self.plotting:
                self._plot(frame_idx, scores, losses, epsilons)
                self.env.render()
                plt.show()


    def test(self, timesteps=50, display=False) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        actions = ['down', 'up', 'right', 'left', 'eat', 'drink', 'rest']
        state = self.env.reset().unsqueeze(0)
        done = False
        score = 0
        t = 0
        cumulative_deviation = 0
        maximum_deviation = 0

        frames = []
        while t < timesteps:
            t += 1
            # frames.append(self.env.render())
            action = self.select_action(state, t)
            next_state, reward, done, info = self.step(action)

            if display:
                print('Timestep:', t,
                      ', reward:', np.round(reward, 3),
                      ', modular reward:', np.round(np.array(info), 3),
                      ', Action: ', actions[action],
                      ', stats: ', np.round(self.env.stats, 2),
                      ', Dead?: ', self.env.dead,
                      ', Lowest: ', np.round(self.env.lowest_stat, 2))
                self.env.render()
                plt.show()

            state = next_state.unsqueeze(0)
            score += reward

            cumulative_deviation += sum([abs(stat - 1) for stat in env.stats])
            for stat in env.stats:
                if abs(stat) > maximum_deviation: maximum_deviation = abs(stat)

        print("score: ", score)

        return score, cumulative_deviation, maximum_deviation

    def _compute_dqn_loss(self) -> torch.Tensor:
        """Return dqn loss."""
        losses = []
        for i in range(self.numModules):
            samples = self.memories[i].sample_batch()
            device = self.device  # for shortening the following lines
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
            reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
            done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

            # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            curr_q_value = self.modules[i](state).gather(1, action)
            next_q_value = self.targets[i](next_state).gather(  # Double DQN
                1, self.modules[i](next_state).argmax(dim=1, keepdim=True)
            ).detach()
            mask = 1 - done
            target = (reward + self.gamma * next_q_value * mask).to(self.device)

            # calculate dqn loss

            # calculate loss and update weights
            if self.blind:
                L1_loss = self.lambd * torch.sum(torch.abs(self.modules[i].mask))  # this is the 'penalty' for having numbers other than 0
                loss = F.smooth_l1_loss(curr_q_value, target) + L1_loss
                losses.append(loss)

            else:
                loss = F.smooth_l1_loss(curr_q_value, target)
                losses.append(loss)

        return losses

    def _target_hard_update(self):
        """Hard update: target <- local."""
        for i in range(self.numModules):
            self.targets[i].load_state_dict(self.modules[i].state_dict())

    def display_q_map(self, stats_force=None):
        q_all = np.zeros((self.numModules, self.action_dim, self.env.size, self.env.size))
        move_back_loc = copy.copy(self.env.loc)
        actions = ['down', 'up', 'right', 'left']
        ## will take environment and agent as parameters
        for module in range(self.numModules):
            for i in range(1, self.env.size - 1):
                for j in range(1, self.env.size - 1):

                    self.env.move_location([i, j])
                    state = self.env.get_state()
                    if stats_force is not None: state[:self.env.n_stats] = stats_force
                    with torch.no_grad():
                        q_vals = self.modules[module](torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
                    # q_vals = (q_vals - np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) #normalizing step
                    q_vals = (q_vals == np.max(q_vals)).astype(int)  # sets top q-value to 1
                    q_all[module, :, i, j] = q_vals

            fig, axes = plt.subplots(1, self.action_dim, figsize=(8, 2))
            for i, ax in enumerate(axes):
                ax.imshow(q_all[module, i, :, :])
                ax.set_title(f'Q-{actions[i]}')
            plt.show()

        self.env.move_location(move_back_loc)

    def get_v_map(self, stats_force=None, display=True):
        v_map = np.empty((self.numModules, self.env.size, self.env.size))
        move_back_loc = copy.copy(self.env.loc)
        ## will take environment and agent as parameters
        for module in range(self.numModules):
            for i in range(1, self.env.size - 1):
                for j in range(1, self.env.size - 1):

                    self.env.move_location([i, j])
                    state = self.env.get_state()
                    if stats_force is not None: state[:self.env.n_stats] = stats_force
                    with torch.no_grad():
                        q_vals = self.modules[module](torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
                    v_map[module, i, j] = np.max(q_vals)

        if display:
            fig, axes = plt.subplots(1, self.numModules, figsize=(20, 20))
            if self.numModules > 1:
                for i, ax in enumerate(axes):
                    self.display_grid_with_text(v_map[i, 1:-1, 1:-1], ax)
                    ax.set_title(f'Module {i + 1}')
                    # ax.imshow(v_map[i,1:-1,1:-1])
            else:
                self.display_grid_with_text(v_map[0, 1:-1, 1:-1], axes)
            plt.show()

        if not display:
            self.value_map_history.append(v_map[:, 1:-1, 1:-1])

        self.env.move_location(move_back_loc)

    def display_grid_with_text(self, grid, ax):
        grid = np.round(grid, 2)
        # fig,ax = plt.subplots(figsize=(8,8))
        ax.imshow(grid)
        for (j, i), label in np.ndenumerate(grid):
            ax.text(i, j, label, ha='center', va='center', fontsize=10, fontweight='bold', color='r')
            ax.set_yticks([])
            ax.set_xticks([])

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        # plt.plot(scores)
        # plt.plot(self.rewards)
        plt.plot(self.mod_rewards)
        # plt.ylim(-5,10)
        plt.subplot(122)
        plt.title('stats')
        plt.plot(self.stats)
        plt.ylim([-10, 10])
        plt.legend([f'stat {i + 1}: {np.round(self.env.stats[i], 2)}' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Stat levels')
        self.get_v_map()
        plt.show()
        # plt.subplot(132)
        # plt.title('loss')
        # plt.plot(losses[0])


class arbitrator_ceo(nn.Module):  # convolutional encoder (takes in an image and outputs a single logit)
    def __init__(self, input_dim, action_dim, lr):
        super(arbitrator_ceo, self).__init__()
        self.hidden_dim = 512
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        self.relu = nn.ReLU()
        self.lr = lr
        self.eps = 10e-20

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(list(self.parameters()), lr=lr)

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_scores = self.fc3(x)
        return F.softmax(action_scores, dim=1)

    def update(self, device):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        BUG = len(returns)
        returns = torch.tensor(returns).to(device)
        if BUG > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]


class World():
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_actions=5,
                 size=9,
                 bounds=4,
                 modes=3,
                 field_of_view=3,
                 stat_types=[0, 0],
                 variance=1,
                 radius=2,
                 offset=0,
                 circle=True,
                 stationary=True,
                 ep_length=50,
                 resource_percentile=50,
                 rotation_speed=0.01,
                 reward_type='sq_dev',
                 module_reward_type='sq_dev',
                 stat_decrease_per_step=0.005,
                 stat_increase_multiplier=1,
                 initial_stats=[0.5, 0.9],
                 set_point=1,
                 pq=[2, 2],
                 reward_clip=None,
                 reward_scaling=1,
                 mod_reward_scaling=1,
                 mod_reward_bias=1):

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
        self.set_point = set_point  # goal of stats
        self.stationary = stationary
        self.border = -0.02  # colour of border and of agent on grid
        self.resource_percentile = resource_percentile  # resources affect stat above this
        self.thresholds = []  # list of thresholds for each resource
        self.multiplier = modes  # multiplies the resource map by fixed amount to keep peaks relatively constant
        self.well_being_range = 0.2
        self.name = 'ResourceWorld'
        self.reward_type = reward_type
        self.module_reward_type = module_reward_type
        self.variance = variance
        self.radius = radius
        self.offset = offset
        self.circle = circle
        self.rotation_speed = rotation_speed
        self.initial_stats = initial_stats
        self.stat_increase_multiplier = stat_increase_multiplier
        self.heat_map = np.zeros((self.size, self.size))
        self.location_history = []
        self.reward_clip = reward_clip
        self.reward_scaling = reward_scaling
        self.mod_reward_scaling = mod_reward_scaling
        self.mod_reward_bias = mod_reward_bias
        self.clamp = 0
        self.squeeze_rewards = True
        self.poisson_shuffle_mean = 0.02

        # self.action_space = spaces.Discrete(n_actions) # action space
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_stats*self.view_size**2+self.n_stats,))

        self.n_actions = n_actions
        self.n_statedims = self.n_stats * self.view_size ** 2 + self.n_stats

        self.action_dim = n_actions
        self.state_dim = self.n_stats * self.view_size ** 2 + self.n_stats

        self.grid = np.zeros((self.n_stats, self.size, self.size))  # empty world grid
        self.loc = [2, 2]
        # self.grid = np.random.choice([0,1],(n_stats,self.size,self.size),p=[0.8,0.2]) # p gives percent of 1s

        self.p, self.q = pq[0], pq[1]  # homeostatic RL exponents for reward function

        self.stat_decrease_per_step = stat_decrease_per_step  # stat decreases over time
        self.stat_decrease_per_injury = 0.05  # stat decrease per danger (not in current version)
        self.stat_increase_per_recover = 0.1  # stat increases if criteria (not in current version)
        self.clumsiness = 0.2
        self.dead = False  # dead if stat hits 0 (just a read out)

        self.make_stats()  # creates stat variable self.stats
        self.reset_grid()
        self.reset()  # resets world, stats, and randomizes location of agent
        self.original_grid = copy.copy(self.grid)

    ## Reset function

    def reset(self):
        # if not self.stationary: self.reset_grid()  # if changing world on every episode
        # self.make_stats()
        self.time_step = 0  # resets step timer
        self.dead = False
        self.done = False
        self.new = []
        # self.stats = copy.deepcopy(self.initial_stats) # puts stats back to initial values
        # self.loc = [np.random.randint(self.fov,self.size-self.fov),np.random.randint(self.fov,self.size-self.fov)] # initialize state randomly
        # self.loc = [2,2]
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]  # gets initial agent view
        return self.get_state()

    def poisson_shuffle(self):
        for stat in range(self.n_stats):
        if np.random.poisson(self.poisson_shuffle_mean) == 0: continue
        self.grid[stat,:,:] = self.multiplier*get_n_patches(grid_size = self.size, bounds = self.bounds, modes = self.modes, var = self.variance) # makes gaussian patches
        self.thresholds[stat] = np.percentile(self.grid[stat,:,:],self.resource_percentile)

    def reset_grid(self):
        self.thresholds = []
        for stat in range(self.n_stats):
            self.grid[stat, :, :] = self.multiplier * get_n_patches(grid_size=self.size, bounds=self.bounds,
                                                                    modes=self.modes,
                                                                    var=self.variance)  # makes gaussian patches
            self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        if self.circle:
            self.thresholds = []
            if not self.stationary:
                self.offset += np.random.uniform(0, 2 * np.pi)  # self.rotation_speed #np.random.uniform(0,2*np.pi)
                self.radius = np.random.uniform(1, 3)
            self.grid = get_n_resources(grid_size=self.size, bounds=self.bounds, resources=self.n_stats,
                                        radius=self.radius, offset=self.offset, var=self.variance)
            for stat in range(self.n_stats):
                self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        self.grid[:, [0, -1], :] = self.grid[:, :, [0, -1]] = self.border  # make border

    def change_first_resource_map(self):
        self.grid[0] = \
        get_n_resources(grid_size=self.size, bounds=self.bounds, resources=self.n_stats, radius=0, offset=2,
                        var=self.variance)[0]
        self.thresholds[0] = np.percentile(self.grid[0], self.resource_percentile)
        self.grid[:, [0, -1], :] = self.grid[:, :, [0, -1]] = self.border  # make border

    def revert_to_original_grid(self):
        self.grid = self.original_grid

    def reset_for_new_agent(self):
        self.make_stats()
        self.loc = [5, 5]
        self.time_step = 0  # resets step timer
        self.dead = False
        self.done = False
        self.history = []
        self.location_history = []
        self.heat_map = np.zeros((self.size, self.size))
        self.revert_to_original_grid()
        self.reset_grid()

    def move_location(self, loc):
        self.loc = loc
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]

    ## Step function

    def step(self, action):
        # Execute one time step within the environment
        self.time_step += 1
        self.set_lowest_stat()

        if self.time_step == self.ep_length: self.done = True

        if action == 0 and self.loc[0] < self.size - self.fov - 1:
            self.loc[0] += 1

        elif action == 1 and self.loc[0] > self.fov:
            self.loc[0] -= 1

        elif action == 2 and self.loc[1] < self.size - self.fov - 1:
            self.loc[1] += 1

        elif action == 3 and self.loc[1] > self.fov:
            self.loc[1] -= 1

        reward = self.step_stats()
        if self.reward_clip is not None: reward = np.clip(reward, -self.reward_clip,
                                                          self.reward_clip)  # reward clipping

        if self.module_reward_type == 'HRRL': module_rewards = self.separate_HRRL_rewards()
        if self.module_reward_type == 'sq_dev': module_rewards = self.separate_squared_rewards()
        if self.module_reward_type == 'lin_dev': module_rewards = self.separate_linear_rewards()

        # self.separate_squared_rewards() #self.separate_HRRL_rewards(100)
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]
        self.history.append((self.grid_with_agent(), self.view, copy.deepcopy(self.stats)))
        self.location_history.append(copy.copy(self.loc))
        self.heat_map[self.loc[0], self.loc[1]] += 1

        if self.squeeze_rewards:
            reward = np.tanh(reward)
            module_rewards = np.tanh(module_rewards)

        if not self.stationary: self.poisson_shuffle() # shuffles resource maps at poisson rate


        return self.get_state(), reward, self.done, module_rewards  # self.dead

    ##### Functions involved in making stats, stepping stats and getting HRRL Rewards

    def make_stats(self):
        self.stats = []
        for stat in self.stat_types:
            if stat == 0:
                self.stats.append(np.random.uniform(self.initial_stats[0], self.initial_stats[1]))
            else:
                self.stats.append(np.random.uniform(0.7, 0.9))

        self.lowest_stat = np.min(self.stats)

        # self.initial_stats = copy.deepcopy(self.stats)

    def set_lowest_stat(self):
        if np.min(self.stats) < self.lowest_stat:
            self.lowest_stat = np.min(self.stats)

    def step_stats(self):
        self.old_stats = copy.deepcopy(self.stats)

        for i in range(self.n_stats - self.clamp):

            if self.stat_types[i] == 0:  # for food/water stats
                # if  self.grid[i,self.loc[0],self.loc[1]] == 1: self.stats[i] += self.stat_increase_per_recover
                self.stats[i] -= self.stat_decrease_per_step[i]
                if self.grid[i, self.loc[0], self.loc[1]] > self.thresholds[i]:
                    self.stats[i] += self.stat_increase_multiplier * self.grid[i, self.loc[0], self.loc[1]]

            if self.stat_types[i] == 1:  # damage stats
                # if self.grid[i,self.loc[0],self.loc[1]] == 1 and np.random.uniform() < self.clumsiness: self.stats[i] -= self.stat_decrease_per_injury # stochastic damage
                # if self.action == 4: self.stats[i] += self.stat_increase_per_recover # for rest action
                self.stats[i] += self.stat_decrease_per_step[i]
                self.stats[i] -= self.stat_increase_multiplier * self.grid[i, self.loc[0], self.loc[1]]
                if self.stats[i] > 1: self.stats[i] = 1

            if self.stat_types[i] == 2:
                self.stats[i] = np.random.normal(self.set_point, 1)

            if self.stats[i] < 0:
                # self.stats[i] = 0
                self.dead = True

        if self.reward_type == 'HRRL': return self.HRRL_reward()
        if self.reward_type == 'sq_dev': return self.sq_dev_reward()
        if self.reward_type == 'well_being': return self.well_being_reward()
        if self.reward_type == 'lin_dev': return self.lin_dev_reward()
        if self.reward_type == 'min_sq': return self.min_sq()

    def HRRL_reward(self):
        return self.reward_scaling * (self.get_cost_surface(self.old_stats) - self.get_cost_surface(
            self.stats))  # - 100*any([x<0.05 for x in self.stats])

    def sq_dev_reward(self):
        return 0.2 - sum([(self.set_point - stat) ** 2 for stat in self.stats])

    def lin_dev_reward(self):
        return sum([0.2 - np.abs(self.set_point - stat) for stat in self.stats])

    def separate_linear_rewards(self):
        return [0.2 - 4 * np.abs(self.set_point - stat) for stat in self.stats]

    def separate_squared_rewards(self):
        return [0.1 - np.abs(self.set_point - stat) ** 2 for stat in self.stats]

    def separate_HRRL_rewards(self):

        return [self.mod_reward_scaling * ((np.abs(self.set_point - old_stat) ** self.p) ** (1 / self.q) - (
                    np.abs(self.set_point - new_stat) ** self.p) ** (1 / self.q)) - self.mod_reward_bias for
                old_stat, new_stat in zip(self.old_stats, self.stats)]

    def well_being_reward(self):
        return 1 if all(
            [stat > self.set_point - self.well_being_range and stat < self.set_point + self.well_being_range for stat in
             self.stats]) else -1

    def min_sq(self):
        return min([-np.abs(self.set_point - stat) ** 2 for stat in self.stats])

    def get_cost_surface(self, stats):
        return sum([np.abs(self.set_point - stat) ** self.p for stat in stats]) ** (1 / self.q)

    def get_state(self):
        return torch.cat((torch.tensor(self.stats).float(), torch.tensor(self.view.flatten()).float())).float()

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
            plt.xticks([])
            plt.yticks([])
        plt.figure()
        # plt.imshow(self.heat_map)
        # plt.show()

def multivariate_gaussian(grid_size = 20, bounds = 4, height = 1, m1 = None, m2 = None, Sigma = None):
    """Return the multivariate Gaussian distribution on array pos."""

    # Our 2-dimensional distribution will be over variables X and Y
    X = np.linspace(-bounds, bounds, grid_size)
    Y = np.linspace(-bounds, bounds, grid_size)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    if m1 is None: # if not specifying means
      m1 = np.random.uniform(-bounds,bounds)
      m2 = np.random.uniform(-bounds,bounds)
    mu = np.array([m1, m2])

    # if Sigma is None: Sigma = dist.make_spd_matrix(2) # if not specifying covariance matrix, pick randomly

    height = np.random.uniform(1,height)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return height*norm(np.exp(-fac / 2) / N)

def norm(z):
  return z/z.sum()

def get_n_patches(grid_size = 20, bounds = 4, modes = 3, radius = None, var = 2):
# The distribution on the variables X, Y packed into pos.
  z = np.zeros((grid_size, grid_size))
  sigma = [[var,0],[0,var]]

  if radius is None: #randomly disperse
    for i in range(0,modes):
      z += multivariate_gaussian(grid_size = grid_size, bounds = bounds, Sigma = sigma)
    return z

  points = get_spaced_means(modes, radius)
  for i in range(0,modes):
    z += multivariate_gaussian(grid_size = grid_size, bounds = bounds, m1 = points[i][0], m2 = points[i][1])
  return z

def get_n_resources(grid_size = 20, bounds = 4, resources = 3, radius = None, offset = 0, var = 1):
# The distribution on the variables X, Y packed into pos.
  z = np.zeros((resources, grid_size, grid_size))

  sigma = [[var,0],[0,var]]

  if radius is None: #randomly disperse
    for i in range(0,resources):
      z[i,:,:] = multivariate_gaussian(grid_size = grid_size, bounds = bounds)
    return z

  points = get_spaced_means(resources, radius, offset)
  for i in range(resources):
    z[i,:,:] = multivariate_gaussian(grid_size = grid_size, bounds = bounds, m1 = points[i][0], m2 = points[i][1], Sigma = sigma)
  return z

# def KL_div(y):
#   return np.abs(0.05-distance.jensenshannon(patches1.flatten(),y.flatten()))

def get_spaced_means(modes, radius, offset):
    radians_between_each_point = 2*np.pi/modes
    list_of_points = []
    for p in range(0, modes):
        node = p + offset
        list_of_points.append( (radius*np.cos(node*radians_between_each_point),radius*np.sin(node*radians_between_each_point)) )
    return list_of_points


# patches1 = get_n_patches(grid_size = 20, bounds = 4, modes = 3, radius = None)
# patches2 = get_n_patches(grid_size = 20, bounds = 4, modes = 3, radius = None)

res = 4
z = get_n_resources(grid_size = 10, bounds = 3, resources = res, radius = 2, offset = 0, var = 2)

for i in range(res):
  plt.subplot(100 + 10*res + i + 1)
  plt.imshow(z[i])

# plt.subplots()
# plt.imshow(np.sum(z,axis=0))

if __name__ == "__main__":

        # Get the slurm array task id to figure out which job to run
    try:
        array_id = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        array_id = 0

    # variables
    learning_rates = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    set = int(np.floor(array_id/10))
    label = '/nonstationary_blind_singleupdate_poisson'# f'/learning_rate_{learning_rates[set]}_'  # 'jumpstat' #grid,  'setpoint' bigreg
    gamma = 0.5
    exp = [1/1, 1/1000, 1/5000, 1/10000]
    nons = [4,5,6,7,8]
    min_epsilon = 0.01

# setpoints = [1,2,3,4,5,6,7,8,9]

    path = os.getcwd()


    # parameters
    num_frames = 30000
    memory_size = 100000
    batch_size = 512
    target_update = 200
    lr = 1e-3
    explore = 0
    switch_episode = 1500000  # for reversal learning
    arbitrator_lr = 1e-3
    mod_hidden_size = 512
    updates = 1

    count = 0

    # for setpoint in setpoints:
    for non in nons:
        for epsilon_decay in exp:
            print(i, gamma, epsilon_decay)

            env = World(n_actions=4,
                        size=10,
                        bounds=3,
                        modes=1,
                        field_of_view=1,
                        stat_types=[0 for x in range(non)],
                        variance=1,
                        radius=3.5,
                        offset=0.5,  # 1.7
                        circle=False,
                        stationary=False,
                        rotation_speed=np.pi,
                        ep_length=50,
                        resource_percentile=95,
                        reward_type='HRRL',  # HRRL, sq_dev, lin_dev, well_being, min_sq
                        module_reward_type='HRRL',  # HRRL, sq_dev, lin_dev
                        stat_decrease_per_step=[0.01 for x in range(non)],
                        stat_increase_multiplier=1,
                        initial_stats=[0.5, 0.5],
                        set_point=5,
                        pq=[4, 2],
                        reward_clip=500,
                        reward_scaling=1,
                        mod_reward_scaling=1,
                        mod_reward_bias=0)

            # # gmq
            agent_mod = Modular_DQN_Agent(env,
                                          memory_size,
                                          batch_size,
                                          target_update,
                                          epsilon_decay,
                                          lr=lr,
                                          gamma=gamma,
                                          min_epsilon=min_epsilon,
                                          explore=explore,
                                          arbitrator_lr=arbitrator_lr,
                                          hidden_size=mod_hidden_size,
                                          decision_process='gmQ',  # gmQ, allQ, CEO, variance
                                          switch_episode=switch_episode,
                                          blind=True)

            specifier = f'{label}_non_{non}_exp_{epsilon_decay}_modular_test_{array_id}.npy'
            #specifier = f'{label}_gmq_gamma_{gamma}_exp_{epsilon_decay}_test_{i}'
            env.reset_for_new_agent()
            agent_mod.plotting = False
            agent_mod.store_value_maps = False
            agent_mod.lambd = 0.0001
            agent_mod.updates_per_step = updates
            agent_mod.train(num_frames)
            np.save(path + specifier, agent_mod.stats)
            # plt.figure()
            # plt.plot(agent_mod.stats)
            # plt.title(f'Mod gamma {gamma} eps {epsilon_decay}')
            # plt.show()

            # varq

            # agent_mod = Modular_DQN_Agent(env,
            #                         memory_size,
            #                         batch_size,
            #                         target_update,
            #                         epsilon_decay,
            #                         lr = lr,
            #                         gamma = gamma,
            #                         min_epsilon = min_epsilon,
            #                         explore = explore,
            #                         arbitrator_lr = arbitrator_lr,
            #                         hidden_size = mod_hidden_size,
            #                         decision_process = 'variance', # gmQ, allQ, CEO, variance
            #                         switch_episode = switch_episode)

            # specifier = f'{label}_varq_gamma_{gamma}_exp_{epsilon_decay}_test_{i}'
            # env.reset_for_new_agent()
            # agent_mod.plotting = False
            # agent_mod.store_value_maps = False
            # agent_mod.train(num_frames)
            # np.save(parent_dir+specifier,agent_mod.stats)
            # # plt.figure()
            # # plt.plot(agent_mod.stats)
            # # plt.title(f'Mod gamma {gamma} eps {epsilon_decay}')
            # # plt.show()
            # dqn

            agent_mon = DQN_Agent(env,
                                  memory_size,
                                  batch_size,
                                  target_update,
                                  epsilon_decay,
                                  lr=lr,
                                  gamma=gamma,
                                  min_epsilon=min_epsilon,
                                  hidden_size=1024,
                                  switch_episode=switch_episode,
                                  blind = True)

            specifier = f'{label}_non_{non}_exp_{epsilon_decay}_monolithic_test_{array_id}.npy'
            #specifier = f'{label}_gamma_{gamma}_exp_{epsilon_decay}_test_{i}'
            env.reset_for_new_agent()
            agent_mon.lambd = 0.0001
            agent_mon.plotting = False
            agent_mon.store_value_maps = False
            agent_mon.attention_agent = False
            agent_mon.updates_per_step = updates
            agent_mon.train(num_frames)
            np.save(path + specifier, agent_mon.stats)
            # plt.figure()
            # plt.plot(agent_mon.stats)
            # plt.title(f'Mon gamma {gamma} eps {epsilon_decay}')
            # plt.show()

# #dqn_binary

#       agent_mon = DQN_Agent(env,
#                             memory_size,
#                             batch_size,
#                             target_update,
#                             epsilon_decay,
#                             lr = lr,
#                             gamma = gamma,
#                             min_epsilon = min_epsilon,
#                             hidden_size = 1024,
#                             switch_episode = switch_episode)

#       specifier = f'{label}_gamma_{gamma}_exp_{epsilon_decay}_monolithicbinary_test_{i}.npy'
#       # specifier = f'{label}_gamma_{gamma}_exp_{epsilon_decay}_test_{i}'
#       env.reset_for_new_agent()
#       agent_mon.plotting = False
#       agent_mon.store_value_maps = False
#       agent_mon.attention_agent = True
#       agent_mon.train(num_frames)
#       np.save(parent_dir+specifier,agent_mon.stats)
#       # plt.figure()
#       # plt.plot(agent_mon.stats)
#       # plt.title(f'Mon gamma {gamma} eps {epsilon_decay}')
#       # plt.show()
