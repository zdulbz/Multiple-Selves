
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
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        lr: float = 0.001,
        hidden_size = 512,
        blind = False,
        switch_episode = 10000,
        norm_rewards = False):
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
        self.channels = self.n_statedims #self.env.n_stats # obs_dim
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
        self.lambd = 0.001
        self.epsilon_reset = self.min_epsilon
        self.nonstationarity = None
        self.q_vals = []
        self.norm_rewards = norm_rewards

        
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
        qs = self.dqn(torch.FloatTensor(state).to(self.device))
        self.q_vals.append(qs[0].detach().cpu().numpy())

        if self.epsilon > np.random.random() and not self.is_test:
            # selected_action = self.env.action_space.sample()
            selected_action = np.random.randint(self.action_dim)
        else:
            selected_action = qs.argmax().detach().cpu().numpy()
            # selected_action = selected_action.detach().cpu().numpy()
        
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
              if self.nonstationarity == 'jump':
                self.env.stats[-1] = self.clamp_to
                self.env.clamp = 1
                self.epsilon = self.epsilon_reset

              if self.nonstationarity == 'location':
                self.env.change_first_resource_map()
                self.epsilon = self.epsilon_reset

              if self.nonstationarity == 'blank':
                self.env.zero_last_map()
                self.epsilon = self.epsilon_reset


            if self.attention_agent:
              self.current_task = np.argmax(np.abs([s-5 for s in self.env.stats]))
              # torch.cat((state,torch.tensor(self.current_task).float().unsqueeze(0).unsqueeze(0)),1)

            action = self.select_action(state)
            next_state, reward, done, _ = self.step(action)

            state = next_state.unsqueeze(0)
            score += reward

            if self.attention_agent: self.rewards.append(_[self.current_task]) #attention agent
            else: self.rewards.append(reward) #regular agent

            self.stats.append(copy.deepcopy(self.env.stats))


            if self.store_value_maps: self.get_v_map(display=False)

            self.cumulative_deviation += sum([abs(stat - 1) for stat in self.env.stats])
            for stat in self.env.stats:
              if abs(stat) > self.maximum_deviation: self.maximum_deviation = abs(stat)

            # if episode ends
            # if done:
            #     state = self.env.reset().unsqueeze(0)
            #     scores.append(score)
            #     score = 0

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
                
        self.env.close()
                
    def test(self, timesteps = 50, display = False) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        actions = ['down','up','right','left','eat','drink','rest']
        state = self.env.reset().unsqueeze(0)
        done = False
        score = 0
        t = 0
        cumulative_deviation = 0
        maximum_deviation = 0
        
        frames = []
        while t < timesteps:
            t+=1
            # frames.append(self.env.render())
            action = self.select_action(state)
            next_state, reward, done, info = self.step(action)
            if display:
              print('Timestep:', t, 
                    ', reward:', np.round(reward,3), 
                    ', modular reward:', np.round(np.array(info),3), 
                    ', Action: ',actions[action],
                    ', stats: ', np.round(self.env.stats,2), 
                    ', Dead?: ', self.env.dead,
                    ', Lowest: ', np.round(self.env.lowest_stat,2))
              self.env.render()
              plt.show()



            state = next_state.unsqueeze(0)
            score += reward

            cumulative_deviation += sum([abs(stat - 1) for stat in env.stats])
            for stat in env.stats:
              if abs(stat) > maximum_deviation: maximum_deviation = abs(stat)
        
        print("score: ", score)
        self.env.close()
        
        return score, cumulative_deviation, maximum_deviation

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        if self.norm_rewards:
          reward = (reward - reward.mean()) / (reward.std()+ 1e-10)
        
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

    def get_q_map(self, stats_force = None, display=True):
      q_all = np.zeros((self.action_dim,self.env.size,self.env.size))

      move_back_loc = copy.copy(self.env.loc)
      actions = ['down','up','right','left']
      ## will take environment and agent as parameters
      for i in range(1,self.env.size-1):
        for j in range(1,self.env.size-1):

          self.env.move_location([i,j])
          state = self.env.get_state()
          if stats_force is not None: state[:self.env.n_stats] = stats_force
          with torch.no_grad():
            q_vals = self.dqn(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
          # q_vals = (q_vals - np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) # normalizing step
          q_vals= (q_vals==np.max(q_vals)).astype(int) # sets top q-value to 1
          q_all[:,i,j] = q_vals

      self.env.move_location(move_back_loc)

      if display:
        fig, axes = plt.subplots(1,self.action_dim,figsize=(8,2))
        for i, ax in enumerate(axes):
          ax.imshow(q_all[i])
          ax.set_title(f'Q-{actions[i]}')
        plt.show()

      if not display:
        self.q_map_history.append(q_all)

    def get_v_map(self, stats_force = None,display = True):
      v_map = np.zeros((self.env.size,self.env.size))

      move_back_loc = copy.copy(self.env.loc)
      ## will take environment and agent as parameters
      for i in range(1,self.env.size-1):
        for j in range(1,self.env.size-1):

          self.env.move_location([i,j])
          state = self.env.get_state()
          if stats_force is not None: state[:self.env.n_stats] = stats_force
          with torch.no_grad():
            q_vals = self.dqn(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
          # q_vals = (q_vals - np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) # normalizing step
          v_map[i,j] = np.max(q_vals)

      self.env.move_location(move_back_loc)

      if display:
        self.display_grid_with_text(v_map[1:-1,1:-1])
        # fig, ax = plt.subplots(figsize=(5,5))
        # ax.imshow(v_map[1:-1,1:-1])
        # plt.show()

      if not display:
        self.value_map_history.append(v_map[1:-1,1:-1])

    def display_grid_with_text(self,grid):
      grid = np.round(grid,2)
      fig,ax = plt.subplots(figsize=(8,8))
      ax.imshow(grid)
      for (j,i),label in np.ndenumerate(grid):
          ax.text(i,j,label,ha='center',va='center',fontsize=12,fontweight='bold',color='r')
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
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(self.rewards)
        # plt.ylim(-5,10)
        # plt.subplot(142)
        # plt.title(f'loss, stats: {np.round(self.env.stats,2)}')
        # plt.plot(losses)

        plt.subplot(142)
        plt.title('epsilons')
        plt.plot(epsilons)

        plt.subplot(143)
        plt.title('stats')
        plt.plot(self.stats)
        plt.ylim([-10,10])
        plt.legend([f'stat {i+1}: {np.round(self.env.stats[i],2)}' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Stat levels')

        plt.subplot(144)
        plt.title('Q-values')
        plt.plot(self.q_vals)
        # plt.ylim([-10,10])
        plt.legend([f'$Q_{i+1}$' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Q-values')

        plt.show()
        if self.blind:
          plt.figure()
          plt.plot(np.array(self.dqn.mask[0].detach().cpu().numpy()))
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