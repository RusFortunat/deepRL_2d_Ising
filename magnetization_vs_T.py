# This will be my playground to see if physics-informed choice of reward function for 'smart' 
# 2d Ising model leads to the same physics as of a regular 2d Ising model.
# At the moment i will simply go with model A, i.e., spin-flip model with non-conserved magnetizaiton

from numba import njit
import math
import numpy as np
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

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

# get the state of surroundings, i.e., the info about nutrition field
def get_state(lattice, X, Y, Nx, Ny):
    # particle position
    nextX = X + 1 if X < Nx - 1 else 0
    prevX = X - 1 if X > 0 else Nx - 1
    nextY = Y + 1 if Y < Ny - 1 else 0
    prevY = Y - 1 if Y > 0 else Ny - 1
    #spin0 = lattice[X][Y]
    spin1 = lattice[nextX][Y]
    spin2 = lattice[prevX][Y]
    spin3 = lattice[X][nextY]
    spin4 = lattice[X][prevY]
#    state = spin1 + spin2 + spin3 + spin4
    state = [spin1, spin2, spin3, spin4]
    random.shuffle(state)

    return state

# post-training action selection
def select_action_post_training(state, T): 
    
    # interpret Q values as probabilities when simulating dynamics of the system 
    # in principle this could be easily extended to make this more general, but i am a lazy boi
    with torch.no_grad():
        Q_values = trained_NN(state)
        Q_values[0][0] /= T 
        Q_values[0][1] /= T 
        probs = torch.softmax(Q_values, dim=1) # converts logits to probabilities (torch object)
        dist = Categorical(probs) # feeds torch object to generate a list of probs (numpy object ?)
        action = dist.sample().numpy()[0] # sample list of probs and return the action
        
        return action

# after the training has been finished, we simulate the system's relaxational dynamics with fixed NN parameters
def post_training_simulation(Nx, Ny, sim_duration, T):

    # first and second moments
    total_magnetization = np.zeros(sim_duration)
    #correlation_function = np.zeros(sim_duration)

    # random initial conditions
    lattice = np.random.randint(2, size=(Nx, Ny))
    for x in range(Nx): # must be a smarter way, but i'm lazy
        for y in range(Ny):
            if lattice[x][y] == 0:
                lattice[x][y] = -1

    #lattice_t0 =  np.copy(lattice) # to compute autocorrelation function

    for t in range(sim_duration):
        #memory[:,:,t] = lattice # record the state of the lattice for the animation
        for n in range(Nx*Ny):
            X = random.randint(0, Nx-1)
            Y = random.randint(0, Ny-1)
            state = get_state(lattice, X, Y, Nx, Ny) # get the observation state
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action_post_training(state, T) # select action
            if action == 0:
                spin_new_value = 1
            else:
                spin_new_value = -1    
            # update state 
            lattice[X][Y] = spin_new_value

        # compute first and second moments
        tot_mat = 0
        #auto_corr = 0
        inv_size = 1.0/(Nx*Ny)
        for x in range(Nx): # must be a smarter way, but i'm lazy
            for y in range(Ny):
                tot_mat += lattice[x][y]*inv_size
                #auto_corr += lattice[x][y]*lattice_t0[x][y]*inv_size
       
        total_magnetization[t] = tot_mat
        #correlation_function[t] = auto_corr
        print("Temp = ", T, "; t = ", t, "; M = ", tot_mat)

    return total_magnetization

# Main
if __name__ == '__main__':
    
    n_observations = 4      # just give network a difference between positive and negative spins
    n_actions = 2           # the particle can jump on any neighboring lattice sites, or stay put and eat
    hidden_size = 16        # hidden size of the network
    PATH = "./NN_params.txt"

    trained_NN = DQN(n_observations, hidden_size, n_actions).to(device)
    trained_NN.load_state_dict(torch.load(PATH))
    sim_duration = 100 
    Nx, Ny = 200, 200 
    
    M_vs_T = np.zeros(50)

    for temp in range(50):
        T = temp + 1 # temperature
        total_magnetization = post_training_simulation(Nx, Ny, sim_duration, T)
        aver_magnetization = 0
        for i in range(10): # average over last 10 values
            aver_magnetization += 0.1*total_magnetization[sim_duration - i - 1]
        
        M_vs_T[temp] = aver_magnetization
        print("T = ", T, "; M = ", M_vs_T[temp])
    
    with open('M_vs_T.txt', 'w') as f:
        for t in range(50):
            output_string = str(t + 1) + "\t" + str(M_vs_T[t])  + "\n"
            f.write(output_string)
 
    print("Done!")