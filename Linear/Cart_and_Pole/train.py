'''
This file will have ALL the components for a Deep Q Learning Model: network and agent classes
'''
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import pandas as pd

class Linear_NN(nn.Module):
    
    def __init__(self, LR, INPUT_DIMS, OUTPUT_DIMS, HIDDEN_DIMS=256):
        '''
        3 Layer Linear Neural Network

        INPUT: 
            LR: learn rate
            INPUT_DIMS: list of single integer input dimensions
            HIDDEN_DIMS: integer hidden dimension.
            OUTPUT_DIMS: integer output dimensions
            LOSS: loss function. Declare as a class
        '''
        super(Linear_NN, self).__init__()

        self.input_dims = INPUT_DIMS
        self.output_dims = OUTPUT_DIMS
        self.hidden_dims = HIDDEN_DIMS

        self.fc1 = nn.Linear(*INPUT_DIMS, HIDDEN_DIMS)
        self.fc2 = nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS)
        self.fc3 = nn.Linear(HIDDEN_DIMS, OUTPUT_DIMS)

        self.lr = LR
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Forward pass of my Linear NN
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent():
    
    def __init__(self, INPUT_DIMS, N_ACTIONS, MAX_MEM=10_000, EPS_DEC=5e-4, EPS_MIN=0.01, BATCH_SIZE=64, GAMMA=0.99, LR=0.003):
        # 1. define hparams
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.lr = LR

        self.max_mem = MAX_MEM
        self.mem_counter = 0
        self.learn_counter = 0
        self.target_net_update = BATCH_SIZE

        self.eps = 1.0
        self.eps_dec = EPS_DEC
        self.eps_min = EPS_MIN

        self.action_space = np.arange(N_ACTIONS)
        # 2. define Q_net and target_net
        self.Q_net = Linear_NN(LR=LR, INPUT_DIMS=INPUT_DIMS, OUTPUT_DIMS=N_ACTIONS)
        self.Q_target = Linear_NN(LR=LR, INPUT_DIMS=INPUT_DIMS, OUTPUT_DIMS=N_ACTIONS)
        # 3. define memory
        self.state_memory = np.zeros((self.max_mem, *INPUT_DIMS), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem, *INPUT_DIMS), dtype=np.float32)
        self.action_memory = np.zeros((self.max_mem), dtype=np.int32)
        self.reward_memory = np.zeros((self.max_mem), dtype=np.float32)
        self.terminal_memory = np.zeros((self.max_mem), dtype=np.bool_)

    def store_transitions(self, state, new_state, action, reward, done):
        # store into memory and update mem_cntr
        index = self.mem_counter % self.max_mem

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            # choose best action
            observation = observation.astype(np.float32)
            state = T.tensor(observation).to(self.Q_net.device)
            all_actions = self.Q_net.forward(state)
            action = T.argmax(all_actions).item()
        else:
            # choose random action
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
        self.learn_counter += 1

        if self.mem_counter < self.batch_size:
            return
        # 1. batch the experience replay
        max_mem = min(self.mem_counter, self.max_mem)
        batch = np.random.choice(max_mem, size=self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_net.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_net.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_net.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_net.device)
        # 2. zero gradients
        self.Q_net.zero_grad()
        # 3. Calculate loss as MSE of Q1 and Q2
        # Q1-- q-values of the network for the actions taken
        # Q2-- q-values as calcuated by the Bellman equation (expected future reward)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        Q1 = self.Q_net.forward(state_batch)[batch_index, action_batch]
        Q_next = self.Q_target.forward(new_state_batch)
        Q_next[terminal_batch] = 0

        Q2 = reward_batch + self.gamma * T.max(Q_next, dim=1)[0]
        loss = self.Q_net.loss(Q1, Q2).to(self.Q_net.device)
        loss.backward()
        self.Q_net.optimizer.step()
        
        # 5. Update Epsilon
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min \
            else self.eps_min
        # 6. Update Target Network
        if ((self.learn_counter % self.target_net_update) == 0):
            self.Q_target = self.Q_net

def graph(data1, data2, game, save=True, force_graph=False):
    step_size = 50
    start = game - step_size
    end = game
    if (((game % step_size) == 0) | force_graph):
        plt.clf()
        plt.plot(data1)
        plt.plot(data2)
        plt.xlim(start, end)

        subset = data1[-step_size:]
        # plt.ylim(np.mean(subset) - 2.5 * np.sqrt(np.var(subset)), 
        #          np.mean(subset) + 2.5 * np.sqrt(np.var(subset)))
        plt.ylim(0, np.max(subset))
        plt.xlabel("Game Number")
        plt.ylabel("Score")
        plt.title("Scores over training time")
        plt.grid(True)
        plt.xticks(np.arange(start, end+1, (step_size//10)))
        plt.xticks(np.arange(start, end, (step_size//10)/5), minor=True)
        y_max = np.max(data1)

        plt.show(block=False)
        plt.pause(.1)
        
        if save:
            plt.savefig(f'saved_plots\\games_{start}_{end}') 

'''
Main method which trains a Deep Q Learning algorithm on the Cart and Pole problem from
Open AI gym. Trained on 1,000,000 epochs. Achieved a model which played Cart and Pole for 
600,000 iterations before I stopped play. 
'''
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    my_agent = Agent(INPUT_DIMS= [4], N_ACTIONS= 2, LR=0.001, EPS_MIN=0.1)

    score_history, mean_score_history = [0], [0]
    info_df = pd.DataFrame([[0,0,0,0,0]], columns=["Game", "Epoch", "Elapsed Epochs", "Score", "Mean Score"])
    
    obs, _ = env.reset()
    done = False
    score = 0
    game = 0
    for epoch in range(1_000_000):
        action = my_agent.choose_action(obs)
        new_obs, reward, done, _, _ = env.step(action)
        my_agent.store_transitions(state=obs, new_state=new_obs, reward=reward, 
                                    action=action, done=done)
        my_agent.learn()
        score += reward
        obs = new_obs

        if done:
            game += 1
            score_history.append(score)
            mean_score_history.append(np.mean(score_history[-51:-1]))
            print(f"Game={game}, Epoch={epoch}, Score={score_history[-1]}, Mean score={mean_score_history[-1]}")
            graph(score_history, mean_score_history, game)
            
            new_row = pd.DataFrame([{'Game': game, 'Epoch': epoch, 'Elapsed Epochs': epoch - info_df.iloc[-1, 1], 
                                     'Score': score_history[-1], 'Mean Score': mean_score_history[-1]}])
            info_df = pd.concat([info_df, new_row], ignore_index=True)

            obs, _ = env.reset()
            done = False
            score = 0

    graph(score_history, mean_score_history, game, force_graph=True)
    info_df.to_csv('saved_plots\\info.csv', index=False)
    T.save(my_agent.Q_net, 'saved_plots\\NN_model')
    T.save(my_agent.Q_net.state_dict(), 'saved_plots\\NN_model_params')
