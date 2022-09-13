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
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        lr: float = 0.001, 
        explore = 1000,
        arbitrator_lr = 1e-3,
        hidden_size = 512,
        decision_process = 'gmQ',
        blind = False,
        switch_episode = 10000, 
        lr_actor = 0.003,
        lr_critic = 0.01,
        PPO_gamma = 0.99,
        K_epochs = 30,
        eps_clip = 0.2,
        action_std = 0.6,
        action_std_decay_rate = 0.05,
        min_action_std = 0.02,
        start_control = 0,
        norm_rewards = True,
        skip_unpreferred = False):
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
        self.channels = self.n_statedims #self.env.n_stats # obs_dim
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
        self.epsilon_reset = self.min_epsilon
        self.nonstationarity = None
        self.norm_rewards = norm_rewards
        self.updates_per_step = 1
        self.skip_unpreferred = skip_unpreferred
        # for PPO arbitrator
        self.has_continuous_action_space = True
        self.control_weights = []
        self.weighted_qs = []
        self.unweighted_qs = []
        # stat weighted
        self.stat_weights = []

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.modules = [Network(self.channels, self.action_dim, self.env.view_size, self.hidden_size, blind = self.blind).to(self.device) for i in range(self.numModules)]
        self.targets = [Network(self.channels, self.action_dim, self.env.view_size, self.hidden_size, blind = self.blind).to(self.device) for i in range(self.numModules)]
        # self.arbitrator = arbitrator_ceo(obs_dim, self.numModules, self.arbitrator_lr).to(self.device)
        self.arbitrator = arbitrator_ceo(self.numModules, self.action_dim, self.arbitrator_lr).to(self.device)

        ## PPO Arbitrator ##
        if self.decision_process == 'PPO_arbitrator':
          self.lr_actor = lr_actor
          self.lr_critic = lr_critic
          self.PPO_gamma = PPO_gamma
          self.K_epochs = K_epochs
          self.eps_clip = eps_clip 
          self.action_std = action_std
          self.action_std_decay_rate = action_std_decay_rate
          self.min_action_std = min_action_std
          self.start_control = start_control
          self.arbitrator = PPO(self.numModules, self.action_dim, self.lr_actor, self.lr_critic, self.PPO_gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, self.action_std)
        ## PPO Arbitrator ##

        for i in range(self.numModules):
          self.targets[i].load_state_dict(self.modules[i].state_dict()) 
          self.targets[i].eval() 
        
        # optimizer
        self.optimizers = [optim.Adam(modules.parameters(), lr=self.lr) for modules in self.modules]

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False



    def select_action(self, state: np.ndarray, frame) -> np.ndarray:
        """Select an action from the input state."""
        # modular policy
        # if frame % 7 == 0: self.current_module +=1
        # if self.current_module > self.numModules - 1: self.current_module = 0
        m = nn.Softmax(dim=0)

        prepared_state = torch.FloatTensor(state).to(self.device)
        ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
        preferred_actions = [np.argmax(modqs) for modqs in ModuleQValues]

        if self.epsilon > np.random.random() and not self.is_test and not self.decision_process == 'PPO_arbitrator':
            # selected_action = self.env.action_space.sample()
            selected_action = np.random.randint(self.action_dim)
            self.random_action = True
        else:
          self.random_action = False
          if self.decision_process == 'allQ':
            # ModuleQValues = np.array([self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)])
            ind = np.unravel_index(np.argmax(ModuleQValues, axis=None), ModuleQValues.shape)
            selected_action = ind[-1]
          # meanQs = np.sum(ModuleQValues,axis=1)
          # unhappy_module = np.argmin(meanQs)
          # finalQValues = ModuleQValues[unhappy_module] #unhappy takes control
          if self.decision_process == 'gmQ':
            finalQValues = sum(self.modules[i](prepared_state) for i in range(self.numModules)) ## This is GmQ
            # finalQValues = sum(self.modules[i](prepared_state)/torch.max(self.modules[i](prepared_state)) for i in range(self.numModules)) # centered GmQ
            # finalQValues = sum(m(self.modules[i](prepared_state)) for i in range(self.numModules)) ## pre-softmax to avoid domination
            selected_action = finalQValues.argmax().detach().cpu().numpy()

          if self.decision_process == 'variance':
            # ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
            variances = [np.var(qvals) for qvals in ModuleQValues]
            ind = np.argmax(variances)
            selected_action = ModuleQValues[ind].argmax()

          if self.decision_process == 'voting':
            action_slots = np.zeros(4)
            # ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
            ModuleActions = [np.argmax(qvals) for qvals in ModuleQValues]
            variances = [np.var(qvals) for qvals in ModuleQValues]
            for i in range(self.numModules):
              action_slots[ModuleActions[i]] += variances [i]
            ind = np.argmax(action_slots)
            selected_action = ModuleQValues[ind].argmax()

          if self.decision_process == 'max_abs':
            # ModuleQValues = [self.modules[i](prepared_state).detach().cpu().numpy() for i in range(self.numModules)]
            max_abs = [np.sum(np.abs(qvals)) for qvals in ModuleQValues]
            ind = np.argmax(max_abs)
            selected_action = ModuleQValues[ind].argmax()

          if self.decision_process == 'CEO':
            probs = self.arbitrator(prepared_state[:,:self.numModules]) # this is arbiQ
            m = Categorical(probs)
            arb_action = m.sample()
            action = torch.argmax(probs)
            self.arbitrator.saved_log_probs.append(m.log_prob(arb_action))
            finalQValues = self.modules[arb_action.item()](prepared_state)
            selected_action = finalQValues.argmax().detach().cpu().numpy()

          if self.decision_process == 'PPO_arbitrator':
            weights = self.arbitrator.select_action(state[:,:self.numModules])
            if frame < self.start_control: weights = np.ones(self.numModules)
            finalQValues = sum(weights[i]*self.modules[i](prepared_state) for i in range(self.numModules)) ## This is GmQ
            unweightedQValues = sum(self.modules[i](prepared_state) for i in range(self.numModules)) ## This is GmQ
            selected_action = finalQValues.argmax().detach().cpu().numpy()
            self.control_weights.append(np.round(weights,3))
            self.weighted_qs.append(finalQValues[0].detach().cpu().numpy())
            self.unweighted_qs.append(unweightedQValues[0].detach().cpu().numpy())

          if self.decision_process == 'stat_weighted':
            weights = [abs(self.env.set_point - stat) for stat in self.env.stats]
            weights = softmax(weights)
            finalQValues = sum(weights[i]*self.modules[i](prepared_state) for i in range(self.numModules)) ## This is GmQ
            # finalQValues = sum(weights[i]*F.softmax(self.modules[i](prepared_state),dim=1) for i in range(self.numModules)) ## This is GmQ
            selected_action = finalQValues.argmax().detach().cpu().numpy()
            self.stat_weights.append(np.round(weights,3))


        # if frame < self.explore_until:
        #   selected_action = np.random.randint(0, self.action_dim)

        
        if not self.is_test:
            self.transition = [state, selected_action]

        self.satisfied_modules = [pref == selected_action for pref in preferred_actions]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, sep_rewards = self.env.step(action)

        if not self.is_test:
            for i in range(self.numModules):
              if self.skip_unpreferred and not self.satisfied_modules[i] and not self.random_action: continue
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

            action = self.select_action(state,frame_idx)
            next_state, reward, done, mod_rewards = self.step(action)

            if self.decision_process == 'CEO':
              self.arbitrator.rewards.append(reward)

            if self.decision_process == 'PPO_arbitrator': 
              self.arbitrator.buffer.rewards.append(reward)
              self.arbitrator.buffer.is_terminals.append(done)

            state = next_state.unsqueeze(0)
            score += reward
            self.rewards.append(reward)
            self.mod_rewards.append(mod_rewards)
            self.stats.append(copy.deepcopy(self.env.stats))


            if self.store_value_maps: self.get_v_map(display=False) # and 9950 < frame_idx < 10200: self.get_v_map(display=False)


            self.cumulative_deviation += sum([abs(stat - 1) for stat in self.env.stats])
            for stat in self.env.stats:
              if abs(stat) > self.maximum_deviation: self.maximum_deviation = abs(stat)
            # if episode ends
            if done:
                state = self.env.reset().unsqueeze(0)
                scores.append(score)
                score = 0
                if self.decision_process == 'CEO':
                  self.arbitrator.update(self.device)
                if self.decision_process == 'PPO_arbitrator' and frame_idx > self.start_control:
                  self.arbitrator.update()
                   # if continuous action space; then decay action std of ouput action distribution
                  self.arbitrator.decay_action_std(self.action_std_decay_rate, self.min_action_std)

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            epsilons.append(self.epsilon)

            # if training is ready
            if all(len(mems) >= self.batch_size for mems in self.memories):
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
                
    def test(self, timesteps = 50,display = False) -> List[np.ndarray]:
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
            action = self.select_action(state,t)
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

          if self.norm_rewards:
            # reward /= reward.abs().max()
            reward = (reward - reward.mean()) / (reward.std()+ 1e-10)
          
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
            L1_loss = self.lambd*torch.sum(torch.abs(self.modules[i].mask)) # this is the 'penalty' for having numbers other than 0
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


    def display_q_map(self, stats_force = None):
      q_all = np.zeros((self.numModules,self.action_dim,self.env.size,self.env.size))
      move_back_loc = copy.copy(self.env.loc)
      actions = ['down','up','right','left']
      ## will take environment and agent as parameters
      for module in range(self.numModules):
        for i in range(1,self.env.size-1):
          for j in range(1,self.env.size-1):

            self.env.move_location([i,j])
            state = self.env.get_state()
            if stats_force is not None: state[:self.env.n_stats] = stats_force
            with torch.no_grad():
              q_vals = self.modules[module](torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
            # q_vals = (q_vals - np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) #normalizing step
            q_vals= (q_vals==np.max(q_vals)).astype(int) # sets top q-value to 1
            q_all[module,:,i,j] = q_vals

        fig, axes = plt.subplots(1,self.action_dim,figsize=(8,2))
        for i, ax in enumerate(axes):
          ax.imshow(q_all[module,i,:,:])
          ax.set_title(f'Q-{actions[i]}')
        plt.show()

      self.env.move_location(move_back_loc)

    def get_v_map(self, stats_force = None, display=True):
      v_map = np.empty((self.numModules,self.env.size,self.env.size))
      move_back_loc = copy.copy(self.env.loc)
      ## will take environment and agent as parameters
      for module in range(self.numModules):
        for i in range(1,self.env.size-1):
          for j in range(1,self.env.size-1):

            self.env.move_location([i,j])
            state = self.env.get_state()
            if stats_force is not None: state[:self.env.n_stats] = stats_force
            with torch.no_grad():
              q_vals = self.modules[module](torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
            v_map[module,i,j] = np.max(q_vals)

      if display:
        fig, axes = plt.subplots(1,self.numModules,figsize=(20,20))
        if self.numModules > 1:
          for i, ax in enumerate(axes):
            self.display_grid_with_text(v_map[i,1:-1,1:-1],ax)
            ax.set_title(f'Module {i+1}')
            # ax.imshow(v_map[i,1:-1,1:-1])
        else: self.display_grid_with_text(v_map[0,1:-1,1:-1],axes)
        plt.show()

      if not display:
        self.value_map_history.append(v_map[:,1:-1,1:-1])

      self.env.move_location(move_back_loc)

    def display_grid_with_text(self,grid,ax):
      grid = np.round(grid,2)
      # fig,ax = plt.subplots(figsize=(8,8))
      ax.imshow(grid)
      for (j,i),label in np.ndenumerate(grid):
          ax.text(i,j,label,ha='center',va='center',fontsize=10,fontweight='bold',color='r')
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
        clear_output(True)
        plt.figure(figsize=(30, 5))
        plt.subplot(151)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        # plt.plot(scores)
        # plt.plot(self.rewards)
        plt.plot(self.mod_rewards)
        # plt.ylim(-5,10)
        plt.subplot(152)
        plt.title('stats')
        plt.plot(self.stats)
        plt.ylim([-10,10])
        plt.legend([f'stat {i+1}: {np.round(self.env.stats[i],2)}' for i in range(self.env.n_stats)])
        plt.xlabel('Time step')
        plt.ylabel('Stat levels')

        if self.decision_process == 'PPO_arbitrator':
          plt.subplot(153)
          plt.title('Control weights')
          plt.plot(self.control_weights)
          # plt.ylim([-10,10])
          plt.legend([f'weight {i+1}: {np.round(self.control_weights[-1][i],3)}' for i in range(self.env.n_stats)])
          plt.xlabel('Time step')
          plt.ylabel('weight magnitude')

          plt.subplot(154)
          plt.title('Weighted Q-values')
          plt.plot(self.weighted_qs)
          # plt.ylim([-10,10])
          plt.legend([f'$Q_{i+1}$: {np.round(self.weighted_qs[-1][i],3)}' for i in range(4)])
          plt.xlabel('Time step')
          plt.ylabel('Q-value')

          plt.subplot(155)
          plt.title('Unweighted Q-values')
          plt.plot(self.unweighted_qs)
          # plt.ylim([-10,10])
          plt.legend([f'$Q_{i+1}$: {np.round(self.unweighted_qs[-1][i],3)}' for i in range(4)])
          plt.xlabel('Time step')
          plt.ylabel('Q-value')

        if self.decision_process == 'stat_weighted':
          plt.subplot(153)
          plt.title('Stat weights')
          plt.plot(self.stat_weights)
          # plt.ylim([-10,10])
          plt.legend([f'weight {i+1}: {np.round(self.stat_weights[-1][i],3)}' for i in range(self.env.n_stats)])
          plt.xlabel('Time step')
          plt.ylabel('weight magnitude')

        self.get_v_map()
        plt.show()
        if self.blind:
          plt.figure()
          plt.plot(np.array([x.mask[0].detach().cpu().numpy() for x in self.modules]).T)
          plt.show()
        # plt.subplot(132)
        # plt.title('loss')
        # plt.plot(losses[0])
