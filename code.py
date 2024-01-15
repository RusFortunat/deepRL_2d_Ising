# This will be my playground to see if physics-informed choice of reward function for 'smart' 
# 2d Ising model leads to the same physics as of a regular 2d Ising model.
# At the moment i will simply go with model A, i.e., spin-flip model with non-conserved magnetizaiton

import numpy as np
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our model neural network
class DQN(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

# structure of the Q table
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class that defines the Q table
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # deque is a more efficient form of a list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# get the state of surroundings, i.e., the info about nutrition field
def get_state(lattice, X, Y, Nx, Ny):
    # particle position
    nextX = X + 1 if X < Nx - 1 else 0
    prevX = X - 1 if X > 0 else Nx - 1
    nextY = Y + 1 if Y < Ny - 1 else 0
    prevY = Y - 1 if Y > 0 else Ny - 1
    spin0 = lattice[X][Y]
    spin1 = lattice[nextX][Y]
    spin2 = lattice[prevX][Y]
    spin3 = lattice[X][nextY]
    spin4 = lattice[X][prevY]
    state = [spin0, spin1, spin2, spin3, spin4]

    return state

# select the action, based on observation or take the random action; greedy epsilon policy implementation
def select_action(state): 
    global steps_done # count total number of steps to go from almost random exploration to more efficient actions
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row. Second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
    else:
        # select a random action; 
        rand_aciton = random.randint(0,1) # flip up or down randomly

    return torch.tensor([[rand_aciton]], device=device, dtype=torch.long)

# update the position of the particle
def step(lattice, X, Y, Nx, Ny, action):
    
    spin_old_value = lattice[X][Y]
    if action == 0:
        spin_new_value = 1
    else:
        spin_new_value = -1

    # compute reward
    nextX = X + 1 if X < Nx - 1 else 0
    prevX = X - 1 if X > 0 else Nx - 1
    nextY = Y + 1 if Y < Ny - 1 else 0
    prevY = Y - 1 if Y > 0 else Ny - 1
    deltaE = -1.0*(spin_new_value - spin_old_value)*(lattice[nextX][Y] + lattice[prevX][Y] + lattice[X][nextY] + lattice[X][prevY]) 
    reward = 0.0 
    if deltaE > 0:
        reward = -1.0*deltaE
    else:
        reward = 1.0

    # update state 
    lattice[X][Y] = spin_new_value

    new_state = get_state(lattice, X, Y, Nx, Ny)
    return reward, new_state

# update neural network parameters (perform backprop)
def optimize_model():
    if len(memory) < BATCH_SIZE: # execute 'optimize_model' only if #BATCH_SIZE number of updates have happened 
        return
    transitions = memory.sample(BATCH_SIZE) # draws a random set of transitions; the next_state for terminal transition will be NONE
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # turn [transition, (args)] array into [[transitions], [states], [actions], ... ]

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s != None, batch.next_state)), device=device, dtype=torch.bool) # returns a set of booleans
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # creates a list of non-empty next states
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # My understanding (for the original explanation check tutorial): 
    # Policy_net produces [[Q1,...,QN], ...,[]] (BATCH x N)-sized matrix, where N is the size of action space, 
    # and action_batch is BATCH-sized vector whose values are the actions that have been taken. 
    # Gather tells which Q from [Q1,...,QN] row to take, using action_batch vector, and returns BATCH-sized vector of Q(s_t, a) values
    state_action_values = policy_net(state_batch).gather(1, action_batch) # input = policy_net, dim = 1, index = action_batch

    # My understanding (for the original explanation check tutorial): 
    # Compute Q^\pi(s_t,a) values of actions for non_final_next_states by using target_net (old policy_net), from which max_a{Q(s_t, a)} are selected with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # target_net produces a vector of Q^pi(s_t+1,a)'s and max(1)[0] takes maxQ
    # Compute the expected Q^pi(s_t,a) values for all BATCH_SIZE (default=128) transitions
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# part of the code where all training is done
def do_training(num_train_episodes, Nx, Ny, Nt):
    for i_episode in range(num_train_episodes): # average the data over different independent runs
        print("Training episode ", i_episode)
        
        # we start from a disordered configuration each time, i.e., random initial conditions 
        lattice = np.random.randint(2, size=(Nx, Ny))
        for x in range(Nx): # must be a smarter way, but i'm lazy
            for y in range(Ny):
                if lattice[x][y] == 0:
                    lattice[x][y] = -1

        # main update loop; I use Monte Carlo random sequential updates here
        for t in range(Nt):
            # pick random spin
            X = random.randint(0, Nx-1)
            Y = random.randint(0, Ny-1)
            # update lattice state
            state = get_state(lattice, X, Y, Nx, Ny) # get the observation state
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action(state) # select action
            reward, next_state = step(lattice, X, Y, Nx, Ny, action) # update particle's position
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            score += reward

            memory.push(state, action, next_state, reward) # Store the transition in memory
            optimize_model() # Perform one step of the optimization (on the policy network)
            # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ)θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            # are we done with the episode?
            if score < - 2 * Nt: # stop episode if the system is doing very poorly
                print("The system hasn't learned shit after ", t, " timesteps. Start over..")
                rewards.append(score) 
                episode_durations.append(t)
                plot_durations()
                break   
          
    torch.save(target_net.state_dict(),PATH)
    plot_durations(show_result=True)

# after the training has been finished, we simulate the system's relaxational dynamics with fixed NN parameters
def post_training_simulation(Nx, Ny, sim_duration, n_observations, hidden_size, n_actions):
    print("The training is done, let's now see what are the results")
    ############# Load NN parameters ##############
    trained_NN = DQN(n_observations, hidden_size, n_actions).to(device)
    trained_NN.load_state_dict(torch.load(PATH))

    # first and second moments
    total_magnetization = np.zeros(sim_duration)
    correlation_function = np.zeros(sim_duration)

    # random initial conditions
    lattice = np.random.randint(2, size=(Nx, Ny))
    for x in range(Nx): # must be a smarter way, but i'm lazy
        for y in range(Ny):
            if lattice[x][y] == 0:
                lattice[x][y] = -1

    lattice_t0 =  np.copy(lattice) # to compute autocorrelation function

    for t in range(sim_duration):
        X = random.randint(0, Nx-1)
        Y = random.randint(0, Ny-1)
        state = get_state(lattice, X, Y, Nx, Ny) # get the observation state
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = select_action(state) # select action
        _, _ = step(lattice, X, Y, Nx, Ny, action) # update particle's position

        # compute first and second moments
        tot_mat = 0
        auto_corr = 0
        inv_size = 1.0/(Nx*Ny)
        for x in range(Nx): # must be a smarter way, but i'm lazy
            for y in range(Ny):
                tot_mat += lattice[x][y]*inv_size
                auto_corr += lattice[x][y]*lattice_t0[x][y]*inv_size
       
        total_magnetization[t] = tot_mat
        correlation_function[t] = auto_corr
        print("t = ", t, "; M = ", tot_mat, "; C = ", auto_corr)

    return total_magnetization, correlation_function

# Plotting part
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plots
def animate(frame):
    im.set_array(csave[frame+200])
    return im,

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() # clf -- clear current figure
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Durations')
    #plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            output = "/Users/Ruslan.Mukhamadiarov/Work/smartaxis/codepy/training_durations.png"
            plt.savefig(output, format = "png", dpi = 300)

# Main
if __name__ == '__main__':
    Jesse_we_need_to_train_NN = True
    PATH = "./NN_params.txt"
    ############# Model parameters for Machine Learning #############
    num_episodes = 500      # number of training episodes
    BATCH_SIZE = 100        # the number of transitions sampled from the replay buffer
    GAMMA = 0.99            # the discounting factor
    EPS_START = 0.9         # EPS_START is the starting value of epsilon; determines how random our action choises are at the beginning
    EPS_END = 0.05          # EPS_END is the final value of epsilon
    EPS_DECAY = 1000        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005             # TAU is the update rate of the target network
    LR = 1e-4               # LR is the learning rate of the AdamW optimizer
    n_observations = 5      # for simplicity, just choose a square 3x3 patch around the particle 
    n_actions = 2           # the particle can jump on any neighboring lattice sites, or stay put and eat
    hidden_size = 100       # hidden size of the network
    ############# Lattice simulation parameters #############
    Nx = 100                # Lx
    Ny = 100                # Ly
    Nt = 10000              # total number of timesteps

    ############# Do the training if needed ##############
    if Jesse_we_need_to_train_NN:
        policy_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(Nt) # the overall memory batch size 
        rewards = []
        episode_durations = []
        steps_done = 0 
        do_training() # training is hapenning for the c=1.0 field

    ############# Simulate the system relaxational dynamics after training ##############
    runs = 1
    sim_duration = 2000 #int(mean_duration) # was about 200 
    Nx, Ny = 200, 200 # because average life expectancy was about 200
    #for run in range(runs):
    total_magnetization, correlation_function = post_training_simulation(Nx, Ny, sim_duration, n_observations, hidden_size, n_actions)
    
    with open('Ouput.txt', 'w') as f:
        for t in range(sim_duration):
            output_string = str(t) + "\t" + str(total_magnetization[t]) + "\t" + str(correlation_function[t]) + "\n"
            f.write(output_string)
