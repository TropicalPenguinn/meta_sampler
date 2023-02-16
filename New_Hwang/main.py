import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import os
from normalized_actions import NormalizedActions
from common.buffers import *
from dqn import Meta_Aent
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--test_interval', type=int, default=10000, metavar='N',
                    help='Test Steps')
parser.add_argument('--test_num', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=True)
args = parser.parse_args()

# Environment

import random

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Tesnorboard
i = 0
while True:
    if os.path.exists('run/{}/New_Hwang/{}'.format(args.env_name,i)):
        i += 1
    else:
        break
writer = SummaryWriter('run/{}/New_Hwang/{}'.format(args.env_name, i))
print("{},{}".format(i,args.seed))
# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Meta
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]
device=torch.device("cuda")
meta_agent = Meta_Aent(env, args, device, obs_dim + act_dim + 1, 2)
meta_memory = ReplayBuffer(obs_dim + act_dim + 1, 1, args.replay_size, device)

# Training Loop
total_numsteps = 0
updates = 0
step_number = 0
meta_action_num=0
test_env = list()
for i in range(args.test_num):
    test = gym.make(args.env_name)
    test.seed(args.seed + 10 * i)
    test_env.append(test)
scores_window = deque(maxlen=50)
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    previous_meta_state = None

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)

                updates += 1

        if len(meta_memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                meta_agent.update_parameters(meta_memory)

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        scores_window.append(reward)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        meta_state = np.concatenate((state, action, np.array([reward])), axis=0)
        meta_action = meta_agent.select_action(torch.Tensor(meta_state).to(device))

        if meta_action == 0:
            memory.push(state, action, reward, next_state, mask)
            meta_action_num +=1
        # meta sampler
        if previous_meta_state is not None:
            meta_memory.add(previous_meta_state,previous_meta_action,reward-previous_reward-2*np.mean(scores_window),meta_state,done)
        state = next_state
        previous_meta_action=meta_action
        previous_meta_state=meta_state
        previous_reward=reward




        if total_numsteps % args.test_interval == 0:
            org_state = state
            avg_reward = 0.
            for test in test_env:
                state = test.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = test.step(action)
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= args.test_num
            writer.add_scalar('score/score', avg_reward, total_numsteps)
            writer.add_scalar('score/percentage',meta_action_num/total_numsteps,total_numsteps)
            done = False
            state = org_state

    if total_numsteps > args.num_steps:
        break

env.close()

