# This is the code for classic 2d Ising model, where I look at quenching dynamics form disordered to ordered state

from numba import njit
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# after the training has been finished, we simulate the system's relaxational dynamics with fixed NN parameters
#@njit
def simulation_loop(Nx, Ny, sim_duration, T):
    rng = np.random.default_rng(12345)
    # first and second moments
    total_magnetization = np.zeros(sim_duration)
    correlation_function = np.zeros(sim_duration)

    # random initial conditions
    lattice = np.random.randint(2, size=(Nx, Ny))
    for x in range(Nx): # must be a smarter way, but i'm lazy
        for y in range(Ny):
            if lattice[x][y] == 0:
                lattice[x][y] = -1
    print(lattice)

    lattice_t0 =  np.copy(lattice) # to compute autocorrelation function

    for t in range(sim_duration):
        memory[:,:,t] = lattice # record the state of the lattice for the animation
        for n in range(Nx*Ny):
            X = random.randint(0, Nx-1)
            Y = random.randint(0, Ny-1)
            spin_old_value = lattice[X][Y]
            nextX = X + 1 if X < Nx - 1 else 0
            prevX = X - 1 if X > 0 else Nx - 1
            nextY = Y + 1 if Y < Ny - 1 else 0
            prevY = Y - 1 if Y > 0 else Ny - 1
            deltaE = 2.0 * spin_old_value * (lattice[nextX][Y] + lattice[prevX][Y] + lattice[X][nextY] + lattice[X][prevY]) 
            prob = math.exp(-deltaE/T)
            dice = rng.random()
            if dice < prob: # flip spin
                spin_new_value = -1*spin_old_value
                lattice[X][Y] = spin_new_value
            #print("dE = ", deltaE, "; prob = ", prob, "; dice = ", dice, "; old spin = ", spin_old_value, "; new spin = ", lattice[X][Y])

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
        #print(lattice)
        

    return total_magnetization, correlation_function, memory


# Main
if __name__ == '__main__':
    Jesse_we_need_to_train_NN = True
    PATH = "./NN_params.txt"
    num_episodes = 100      # number of training episodes
    Nx = 100                # Lx
    Ny = 100                # Ly
    sim_duration = 500      # total number of timesteps
    T = 0.1                 # temperature
    runs = 1

    #for run in range(runs):
    memory = np.zeros((Nx, Ny, sim_duration), dtype=int)
    total_magnetization, correlation_function, memory = simulation_loop(Nx, Ny, sim_duration, T)
    
    filename = 'Ouput' + str(T) + '.txt'
    with open(filename, 'w') as f:
        for t in range(sim_duration):
            output_string = str(t) + "\t" + str(total_magnetization[t]) + "\t" + str(correlation_function[t]) + "\n"
            f.write(output_string)

    # animation
    fig = plt.figure(figsize=(24, 24))
    im = plt.imshow(memory[:, :, 1], interpolation="none", aspect="auto", vmin=0, vmax=1)

    def animate_func(i):
        im.set_array(memory[:, :, i])
        return [im]

    fps = 60

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=sim_duration,
        interval=1000 / fps,  # in ms
    )

    print("Saving animation...")
    filename_animation = "anim_T" + str(T) + ".mp4"
    anim.save(filename_animation, fps=fps, extra_args=["-vcodec", "libx264"])
    print("Done!")