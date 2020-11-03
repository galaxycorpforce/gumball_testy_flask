from gumball_env import *
import random

env = GumballEnv()
config = {}
state, _, _, info = env.reset()
rewards = 0
steps = 0
sample_obs = state
print('sample_obs:\n')
print(sample_obs)
p1_rewards = []
p2_rewards = []
print('valid_actions', info['valid_actions'])
for i in range(3):
    player='p1'
    done = False
    state, _, _, _ = env.reset()
    while not done:
    #    action1 = env.sample_actions()
        action = env.sample_actions()
        obs, reward, done, info = env.step(action)
        if player == 'p1':
            p1_rewards.append(reward)
        else:
            p2_rewards.append(reward)
        player=info['player']
        steps += 1
    print('finished match:', i)

print('reward', rewards)
print('steps', steps)
print('p1_rewards', (p1_rewards))
print('p2_rewards', (p2_rewards))
print('sum p1_rewards', sum(p1_rewards))
print('sum p2_rewards', sum(p2_rewards))
