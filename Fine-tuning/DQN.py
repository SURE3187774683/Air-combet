import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import os

np.random.seed(1)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  #To prevent hash randomization, make the experiment reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()
BATCH_SIZE = 256
LR = 0.0001  #Learning rate
GAMMA = 0.90  #Discount factor, discount for future rewards
EPSILON_START = 0.90  #Initial epsilon in e-greedy policy
EPSILON_END = 0.01  # Terminating epsilon in e-greedy policy
EPSILON_DECAY = 100  #The decay rate of epsilon in e-greedy strategy
TARGET_UPDATE = 4  #Update frequency of the target network
MEMORY_CAPACITY = 1000000  #Experience playback capacity
HIDDEN_DIM1 = 32  #Network Hidden Layer 1
HIDDEN_DIM2 = 32  #Network Hidden Layer 2
STATE_DIM = 12   #State space dimension
ACTION_DIM = 7   #Action space dimension

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=32, hidden_dim2=32):
        """ Initialize the q network as a fully connected network
            state_dim: The number of characteristics entered is the state dimension of the environment
            action_dim: output action dimension
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)  # Input layer → Hidden Layer 1
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Hidden Layer 1 → Hidden Layer 2
        self.fc3 = nn.Linear(hidden_dim2, action_dim)   # Hidden Layer 2 → Output Layer

    def forward(self, x):
        # Activation function corresponding to each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQN(object):
    def __init__(self):
        # Neural network parameter setting
        self.policy_net = Net(STATE_DIM, ACTION_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2)
        self.target_net = Net(STATE_DIM, ACTION_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2)

        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # Copy parameters to target network
            target_param.data.copy_(param.data)

        # Buffer parameter settings
        self.learn_step_counter = 0  # Statistical Steps
        self.capacity = MEMORY_CAPACITY  #Experience playback capacity
        self.buffer = []  # buffer
        self.position = 0  # Location in buffer

        # E-greedy policy related parameters
        self.frame_idx = 0  #Attenuation count for epsilon

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  # optimizer

    def epsilon(self,frame_idx):
        #epsilon attenuation formula
        return EPSILON_END + \
                (EPSILON_START - EPSILON_END) * \
                math.exp(-1 * frame_idx / EPSILON_DECAY)

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):  #Decrease from initial value of 0.9
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()  #Select the action with the maximum Q value
        else:
            action = random.randrange(ACTION_DIM)
        return action

    def store_transition(self, state, action, reward, next_state,done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Insert a null value
        self.buffer[self.position] = (state, action, reward, next_state,done)
        self.position = (self.position + 1) % self.capacity

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:  # Do not update policy when a batch is not satisfied in memory
            return
        # sampling
        batch = random.sample(self.buffer, BATCH_SIZE)  # Random extraction and small batch transfer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)  # Decompression into a state, action, etc
        # Convert to Tensor
        state_batch = torch.tensor(state_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch, ).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch),dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch))
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # Calculate the Q (s_t, a) corresponding to the current state (s_t, a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()  # Calculate the Q value corresponding to the state (s_t_, a) at the next moment
        #Calculate the expected Q value. For the termination state, done_Batch [0]=1, corresponding expected_q_value equals reward
        expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  
        # Optimize update model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  #Clip to prevent gradient explosion
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # def save_checkpoint(self, checkpoint_file):
    #     torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)
    #
    # def load_checkpoint(self, checkpoint_file):
    #     self.load_state_dict(torch.load(checkpoint_file))