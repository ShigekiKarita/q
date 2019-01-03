import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import DQN
from memory import ReplayMemory, Transition
from preprocess import FrameWindow, get_screen

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="CartPole-v0")
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--num_episodes", default=100000)
parser.add_argument("--num_frames", default=4)
args = parser.parse_args()

env = gym.make(args.task).unwrapped
# env.frameskip = 4

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env.reset()
plt.figure()
screen = get_screen(env)
print(screen.shape)
if screen.shape[1] == 1:  # gray
    plt.imshow(screen[0, 0].numpy(), #.squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
else:
    plt.imshow(screen.squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')

plt.title('Example extracted screen')
plt.show()
plt.savefig("extracted.png")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen(env).to(device)
_, screen_ch, screen_height, screen_width = init_screen.shape
n_action = env.action_space.n

policy_net = DQN(screen_ch * args.num_frames, screen_height, screen_width, n_action).to(device)
target_net = DQN(screen_ch * args.num_frames, screen_height, screen_width, n_action).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_action)]], device=device, dtype=torch.long)


episode_rewards = []
mean_rewards = []

def plot_rewards(reward):
    plt.figure(2)
    plt.clf()
    episode_rewards.append(reward)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards)
    # Take 100 episode averages and plot them too
    if len(episode_rewards) >= 100:
        mean_rewards.append(np.mean(episode_rewards[-100:]))
        plt.plot(mean_rewards)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(args.num_episodes):
    # Initialize the environment and state
    env.reset()
    sum_reward = 0
    frames = FrameWindow(args.num_frames, get_screen(env).to(device))
    state = frames.as_state()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        reward = 0
        for f in range(args.num_frames):
            _, reward_f, done, _ = env.step(action.item())
            frames.data[f] = get_screen(env).to(device)
            reward += reward_f
        if args.render:
            env.render()
        sum_reward += reward
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = frames.as_state()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            plot_rewards(sum_reward)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
