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
# NOTE: for animating model outputs, this must match what the model was trained on.
#   May want to save this in models in the future to amend this issue.
# Define time constants
RUNTIME = 10
NUM_T_STEPS = 100*RUNTIME+1

# Define x values for plotting
X_QLC = np.linspace(-L,L,nx_crystal)
t_points = np.linspace(0, RUNTIME, NUM_T_STEPS)
x, t = np.meshgrid(X_QLC, t_points)
TEST_SET = torch.tensor(np.column_stack((x.flatten(), t.flatten()))).to(device)

TITLE_DICT = {0: "N-tot", 1: "N-qll", 2: "N-ice"}

def save_ani_as_gif(animation, file_name, frame_interval = 20):
    """
    Saves animation to disk as a gif.

    Args:
        animation: a matplotlib.animation instance.
        file_name: Name of file to save animation as (excluding filetype).
        frame_interval: time in ms that each frame should remain on screen.
            Default is 20, which results in 60FPS animations.
            NOTE: This doesn't change the "resolution" of each frame or 
                the number of total frames. This only affects how fast 
                the frames are presented.
    """
    
    writer = ani.PillowWriter(fps=(1000//frame_interval))
    animation.save(file_name+".gif", writer=writer)
    print("Animation saved successfully as "+file_name)

def animate_refsol(index, frame_interval = 50, with_diffusion=True, file_name=None):
    """
    Animates reference solution.

    Args:
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
        frame_interval: time in ms that each frame should remain on screen
        with_diffusion: Should the reference solution include diffusion? True or False, True by default.
        file_name: Name of file to save animation as (excluding filetype), or None to not save animation.
            None by default. Animation is saved as a gif.
    """
    REFERENCE_SOLUTION = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS, with_diffusion=with_diffusion)

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

    if file_name is not None:
        save_ani_as_gif(animation, file_name, frame_interval)

    # show me the money
    plt.show()

def animate_IcePINN(model_name, index, frame_interval = 50, file_name=None):
    """
    Animates PINN model output.

    Args:
        model_name: String name of folder model is stored in
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
        frame_interval: time in ms that each frame should remain on screen
        file_name: Name of file to save animation as (excluding filetype), or None to not save animation.
            None by default. Animation is saved as a gif.
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

    if file_name is not None:
        save_ani_as_gif(animation, file_name, frame_interval)

    # show me the money
    plt.show()


def animate_together(model1_name, model2_name, index1, index2 = None, frame_interval = 50, file_name=None):
    """
    Animates PINN model output superimposed on reference solution.

    Args:
        model1_name: String name of folder model is stored in, or reference solution if None.
        model2_name: String name of folder model is stored in, or reference solution if None.
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice.
        index2: 0 for Ntot, 1 for Nqll, 2 for N-ice, or None to match index. None by default.
        frame_interval: time in ms that each frame should remain on screen. 50 by default.
        file_name: Name of file to save animation as (excluding filetype), or None to not save animation.
            None by default. Animation is saved as a gif.
    """
    # Yes I know there is some egregious code duplication and the next 20 lines should
    # go into a helper method. I realized once it was done and I'm not changing it.

    model1_solution = "foo"
    model2_solution = "bar"

    if index2 is None:
        index2 = index
    
    if model1_name is True or model1_name is False:
        # Generate reference solution
        model1_solution = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS, with_diffusion=model1_name)
        if model1_name is False:
            model1_name = "Reference "+TITLE_DICT[index1]+"(without diffusion)"
        else:
            model1_name = "Reference "+TITLE_DICT[index1]+"(with diffusion)"
    else:
        model1 = ip.load_IcePINN(model1_name)
        model1.eval()

        # Get predictions from model1 using test data
        pred1 = ip.enforced_model(TEST_SET, model1)
        Ntot_pred1 = pred1[:, 0]
        Nqll_pred1 = pred1[:, 1]
        Nice_pred1 = Ntot_pred1 - Nqll_pred1

        # Stack predictions to match expected output shape
        model1_solution = torch.stack(
            [Ntot_pred1, Nqll_pred1, Nice_pred1], 
            axis=0).reshape(3, NUM_T_STEPS, int(len(TEST_SET)/NUM_T_STEPS)
                ).cpu().detach().numpy() # Animation breaks on GPU; move to CPU
    
    if model2_name is True or model2_name is False:
        # Generate reference solution
        model2_solution = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS, with_diffusion=model2_name)
        if model2_name is False:
            model2_name = "Reference "+TITLE_DICT[index2]+"(without diffusion)"
        else:
            model2_name = "Reference "+TITLE_DICT[index2]+"(with diffusion)"
    else:
        model2 = ip.load_IcePINN(model2_name)
        model2.eval()

        # Get predictions from model2 using test data
        pred2 = ip.enforced_model(TEST_SET, model2)
        Ntot_pred2 = pred2[:, 0]
        Nqll_pred2 = pred2[:, 1]
        Nice_pred2 = Ntot_pred2 - Nqll_pred2

        # Stack predictions to match expected output shape
        model2_solution = torch.stack(
            [Ntot_pred2, Nqll_pred2, Nice_pred2], 
            axis=0).reshape(3, NUM_T_STEPS, int(len(TEST_SET)/NUM_T_STEPS)
                ).cpu().detach().numpy() # Animation breaks on GPU; move to CPU

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Predicted "+TITLE_DICT[index1]) # set axes

    lines = ax.plot(X_QLC, model1_solution[index1][0], 'b', linewidth=1.0, label=model1_name, zorder=2)
    lines.append(ax.plot(X_QLC, model2_solution[index2][0], 'r', linewidth=1.0, label=model2_name, zorder=0)[0])
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        ref_y = model1_solution[index1][frame]
        net_y = model2_solution[index2][frame]
        
        # Update y-axis limits to fit the data
        ax.set_ylim(np.min([np.min(net_y), np.min(ref_y)]), np.max([np.max(net_y), np.max(ref_y)]))

        # set t values to graph for this frame    
        lines[0].set_ydata(ref_y)
        lines[1].set_ydata(net_y)

        return (lines)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=frame_interval)
    
    if file_name is not None:
        save_ani_as_gif(animation, file_name, frame_interval)
    
    # show me the money
    plt.show()
    