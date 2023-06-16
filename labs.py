import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
from torch.distributions import Normal
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class DuelingQNetwork(nn.Module):
    def __init__(self, action_space, hidden_size=64):
        super(DuelingQNetwork, self).__init__()

        # The input image is 84x84
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(8, 8), stride=(4, 4), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(7*7*32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(7*7*32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.n)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)  # Flatten the tensor
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.tensor(np.array(state)) for state in states]).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([torch.tensor(np.array(state)) for state in next_states]).float().to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones


class DuelingDQN_Agent:
    def __init__(self, observation_space, action_space, learning_rate=0.002, gamma=0.90, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, update_frequency=5):

        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DuelingQNetwork(action_space).to(self.device)
        self.target_network = DuelingQNetwork(action_space).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.network_losses = []

        # Initialize frame stack
        self.frame_stack = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)
        self.next_frame_stack = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)

    def update_q_network(self, batch):
        states, actions, rewards, next_states, dones = batch

        next_q_values = self.target_network(next_states).detach().max(1)[0]
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        predicted_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.functional.mse_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def choose_action(self):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.stack(self.frame_stack), dtype=torch.float32).unsqueeze(0).to(self.device)
                return int(self.q_network(state_tensor).argmax().item())

    def learn(self, observation, action, reward, next_observation, done):
        self.frame_stack.append(observation)
        self.next_frame_stack.append(next_observation)  # update the next_frame_stack
        self.replay_buffer.add(self.frame_stack.copy(), action, reward, self.next_frame_stack.copy(), done)
        if len(self.replay_buffer.buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            loss = self.update_q_network(batch)
            self.network_losses.append(loss)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.target_network.load_state_dict(self.q_network.state_dict())


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO_Agent:
    def __init__(self, observation_space, action_space, hidden_size=64, lr=3e-4, betas=(0.9, 0.999),
                 gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(observation_space.shape[0], action_space.shape[0], hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(observation_space.shape[0], action_space.shape[0], hidden_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.memory = Memory()

        self.MseLoss = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        dist, _ = self.policy_old(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(log_prob)

        return action.cpu().data.numpy().flatten()

    def learn(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Converting list to tensor
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            dist, state_value = self.policy(old_states)
            entropy = dist.entropy().mean()
            new_logprobs = dist.log_prob(old_actions).sum(dim=-1)  # sum over action dimension

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(new_logprobs - old_logprobs.detach().sum(dim=-1))  # sum over action dimension

            # Finding Surrogate Loss:
            surr1 = ratios * rewards
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * rewards
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_value, rewards) - 0.01 * entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear the memory
        self.memory.clear_memory()


def train_agent(agent, env, episodes, agent_type, max_iter=10000):
    rewards = []
    for episode in range(episodes):
        if agent_type == 'DuelingDQN':
            observation = env.reset().squeeze()
            agent.frame_stack.extend([observation] * 4)  # initialize the frame_stack
        else:
            observation = env.reset()[0]
        done = False
        episode_reward = 0
        iter_count = 0
        while iter_count <= max_iter and not done:
            if agent_type == 'DuelingDQN':
                action = agent.choose_action()  # Choose action based on current frame stack
                next_observation, reward, done, info = env.step(action)
                agent.learn(observation.squeeze(), action, reward, next_observation.squeeze(), done)  # Learn based on current frame stack
            else:
                action = agent.choose_action(observation)  # Choose action based on current frame stack
                next_observation, reward, done, _, _ = env.step(action)
                # Store the information in memory
                agent.memory.rewards.append(reward)
                agent.memory.is_terminals.append(done)
            observation = next_observation
            episode_reward += reward
            iter_count += 1

        if agent_type == 'PPO': # for PPO, learn at the end of each episode
            agent.learn()

        rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f'Episode: {episode}, Reward: {episode_reward}, Average Reward: {np.mean(rewards[-100:])}')
        if episode % 100 == 0:
            if agent_type == 'DuelingDQN':
                torch.save(agent.q_network.state_dict(), f'{agent_type}_episode_{episode}.pt')
            elif agent_type == 'PPO':
                torch.save(agent.policy.state_dict(), f'{agent_type}_episode_{episode}.pt')


def test_agent(agent, env, episodes, agent_type, video_path=None):
    for episode in range(episodes):
        if agent_type == 'DuelingDQN':
            observation = env.reset().squeeze()
            agent.frame_stack.extend([observation] * 4)  # initialize the frame_stack
        else:
            env = env.env
            observation = env.reset()[0]

        if video_path:
            recorder = VideoRecorder(env, path=f"{video_path}/{agent_type}_episode_{episode}.mp4")

        total_reward = 0
        done = False
        while not done:
            if agent_type == 'DuelingDQN':
                action = agent.choose_action()  # Choose action based on current frame stack
                next_observation, reward, done, info = env.step(action)
            else:
                action = agent.choose_action(observation)  # Choose action based on current observation
                next_observation, reward, done, _, _ = env.step(action)

            observation = next_observation
            total_reward += reward

            if video_path:
                env.render()  # Get the image of the current environment state
                recorder.capture_frame()  # Use the image as a frame in the video

        print(f'Test Episode {episode + 1}: Total reward = {total_reward}')

        if video_path:
            recorder.close()



