import f90nml
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity
import reference_solution as refsol
import torch
import icepinn as ip

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Read in GI parameters
inputfile = "GI parameters - Reference limit cycle (for testing).nml"
GI=f90nml.read(inputfile)['GI']
nx_crystal = GI['nx_crystal']
L = GI['L']

# TODO - let user choose runtime when calling functions
# Define time constants
RUNTIME = 5
NUM_T_STEPS = RUNTIME+1

# Define x values for plotting
X_QLC = np.linspace(-L,L,nx_crystal)
t_points = np.linspace(0, RUNTIME, NUM_T_STEPS)
x, t = np.meshgrid(X_QLC, t_points)
TEST_SET = torch.tensor(np.column_stack((x.flatten(), t.flatten()))).to(device)

TITLE_DICT = {0: "Ntot", 1: "Nqll", 2: "N-ice"}

def animate_refsol(index, frame_interval = 50):
    """
    Animates reference solution.

    Args:
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
        frame_interval: time in ms that each frame should remain on screen
    """
    REFERENCE_SOLUTION = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS)

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Reference "+TITLE_DICT[index]) # set axes

    line = ax.plot(X_QLC, X_QLC, 'b', linewidth=1.0, label="Reference Value", zorder=0)[0]
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        ref_y = REFERENCE_SOLUTION[index][frame]

        # Update y-axis limits to fit the data
        ax.set_ylim(np.min(ref_y), np.max(ref_y))

        # set t values to graph for this frame    
        line.set_ydata(ref_y)

        return (line)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=frame_interval)
    # show me the money
    plt.show()

def animate_IcePINN(model_name, index, frame_interval = 50):
    """
    Animates PINN model output.

    Args:
        model_name: String name of folder model is stored in
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
        frame_interval: time in ms that each frame should remain on screen
    """
    loaded_model = ip.load_IcePINN(model_name)
    loaded_model.eval()

    # Get predictions from the network using test data
    pred = ip.enforced_model(TEST_SET, loaded_model)
    Ntot_pred = pred[:, 0]
    Nqll_pred = pred[:, 1]
    Nice_pred = Ntot_pred - Nqll_pred

    # Stack predictions to match expected output shape
    network_solution = torch.stack(
        [Ntot_pred, Nqll_pred, Nice_pred], 
        axis=0).reshape(3, NUM_T_STEPS, int(len(TEST_SET)/NUM_T_STEPS)
            ).cpu().detach().numpy()

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Predicted "+TITLE_DICT[index]) # set axes

    
    line = ax.plot(X_QLC, network_solution[index][0], 'r', linewidth=1.0, label="Network Prediction", zorder=0)[0]
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        net_y = network_solution[index][frame]

        # Update y-axis limits to fit the data
        ax.set_ylim(np.min(net_y), np.max(net_y))

        # set t values to graph for this frame    
        line.set_ydata(net_y)

        return (line)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=frame_interval)
    # show me the money
    plt.show()


def animate_both(model_name, index, frame_interval = 50):
    """
    Animates PINN model output superimposed on reference solution.

    Args:
        model_name: String name of folder model is stored in
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
        frame_interval: time in ms that each frame should remain on screen
    """
    loaded_model = ip.load_IcePINN(model_name)
    loaded_model.eval()

    # Get predictions from the network using test data
    pred = ip.enforced_model(TEST_SET, loaded_model)
    Ntot_pred = pred[:, 0]
    Nqll_pred = pred[:, 1]
    Nice_pred = Ntot_pred - Nqll_pred

    # Stack predictions to match expected output shape
    network_solution = torch.stack(
        [Ntot_pred, Nqll_pred, Nice_pred], 
        axis=0).reshape(3, NUM_T_STEPS, int(len(TEST_SET)/NUM_T_STEPS)
            ).cpu().detach().numpy() # Animation breaks on GPU; move to CPU

    # Generate reference solution
    REFERENCE_SOLUTION = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS)

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Predicted "+TITLE_DICT[index]) # set axes

    lines = ax.plot(X_QLC, REFERENCE_SOLUTION[index][0], 'b', linewidth=1.0, label="Reference Value", zorder=2)
    lines.append(ax.plot(X_QLC, network_solution[index][0], 'r', linewidth=1.0, label="Network Prediction", zorder=0)[0])
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        ref_y = REFERENCE_SOLUTION[index][frame]
        net_y = network_solution[index][frame]
        
        # Update y-axis limits to fit the data
        ax.set_ylim(np.min([np.min(net_y), np.min(ref_y)]), np.max([np.max(net_y), np.max(ref_y)]))

        # set t values to graph for this frame    
        lines[0].set_ydata(ref_y)
        lines[1].set_ydata(net_y)

        return (lines)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=frame_interval)
    # show me the money
    plt.show()
    