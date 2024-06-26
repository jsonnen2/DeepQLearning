from train import Linear_NN
import torch as T
import gym
import numpy as np

'''
Load model and play 50 games. Compute average score
'''
# Load model
model = Linear_NN(LR=0.001, INPUT_DIMS=[4], OUTPUT_DIMS=2)
model.load_state_dict(T.load('models\\NN_model_params'))

score_history = []
env = gym.make('CartPole-v1')
for i in range(50):
    obs, _ = env.reset()
    score = 0
    done = False
    
    while not done:
        action_Q = model.forward(T.tensor(obs))
        action = T.argmax(action_Q).item()

        new_obs, reward, done, _, _ = env.step(action)
        score += reward
        obs = new_obs
        print(score)

    score_history.append(score)
    print(f"Game={i}, Score={score}")

print(np.mean(score_history))
