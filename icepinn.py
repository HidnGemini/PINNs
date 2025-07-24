import f90nml
import numpy as np
from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity
import os
import reference_solution as refsol
from scipy.fft import rfft
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import icepinn as ip

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class SinActivation(nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
    def forward(self, x):
        return torch.sin(x)

class SinusoidalMappingLayer(nn.Module):
    """
    Custom-instantiated layer of sin activations, defined here: https://arxiv.org/pdf/2109.09338
    """
    def __init__(self, input_dim, num_features, sigma=1.0):
        super(SinusoidalMappingLayer, self).__init__()
        self.num_features = num_features
        # Initialize W_1 with Normal(0, sigma^2)
        self.W = nn.Parameter(torch.randn(num_features, input_dim) * sigma*sigma)
        # Initialize b_1 (phase lag) to zeros
        self.b = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # Compute W*x + b, note that we need to transpose W for correct dimensions
        x_proj = torch.matmul(x, self.W.t()) + self.b
        # Apply the sinusoidal mapping: sin(2*pi*(W*x + b))
        return torch.sin(2*np.pi*x_proj)

# Difference between nn._ and nn.functional._ is that functional is stateless: 
# https://stackoverflow.com/questions/63826328/torch-nn-functional-vs-torch-nn-pytorch
class IcePINN(nn.Module):
    """
    This IcePINN class can be instantiated as one of two options:
    - A standard feed-forward neural network
    - An "sf_PINN" with a layer of sin functions instantiated as specified in this paper:
        https://arxiv.org/pdf/2109.09338
    
    """
    def __init__(self, num_hidden_layers, hidden_layer_size, is_sf_PINN=False):
        super().__init__()
        self.is_sf = is_sf_PINN
        self.sml = SinusoidalMappingLayer(2, hidden_layer_size*3, sigma=1.0) # Consider fiddling with sigma
        self.post_sml = nn.Linear(hidden_layer_size*3, hidden_layer_size)
        
        self.sin = SinActivation()
        
        self.fc_in = nn.Linear(2, hidden_layer_size)
        self.post_fc_in = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.fc_hidden = nn.ModuleList()
        for _ in range(num_hidden_layers-2):
            self.fc_hidden.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            
        self.fc_out = nn.Linear(hidden_layer_size, 2)

    def forward(self, x):
        if self.is_sf:
            # Sinusoidal mapping of inputs: Increases initial gradient variability. https://arxiv.org/pdf/2109.09338 
            # This makes it less likely for the PINN to get stuck in local minima at the start of training.
            x = self.sml(x) 
            x = F.tanh(self.post_sml(x))
        else:
            x = F.tanh(self.fc_in(x))
            x = F.tanh(self.post_fc_in(x))

        for layer in self.fc_hidden:
            x = F.tanh(layer(x))

        x = self.fc_out(x)
        return x

class NtotPINN(nn.Module):
    """
    This NN only predicts Ntot internally. The forward method 
    returns a (batch_size, 2) tensor containing (Ntot, NqllEQ(Ntot))
    pairs derived from Nqll.

    NOTE: this is currently an untested proof-of-concept.
    """
    def __init__(self, num_hidden_layers, hidden_layer_size, is_sf_PINN=False):
        # same as IcePINN except out is just Ntot
        super().__init__()
        self.is_sf = is_sf_PINN
        self.sml = SinusoidalMappingLayer(2, hidden_layer_size*3) # Consider fiddling with sigma
        self.post_sml = nn.Linear(hidden_layer_size*3, hidden_layer_size)
        
        self.sin = SinActivation()
        
        self.fc_in = nn.Linear(2, hidden_layer_size)
        self.post_fc_in = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.fc_hidden = nn.ModuleList()
        for _ in range(num_hidden_layers-2):
            self.fc_hidden.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            
        self.fc_out = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        if self.is_sf:
            # Sinusoidal mapping of inputs: Increases initial gradient variability.
            # This makes it less likely for the PINN to get stuck in local minima at the start of training.
            x = self.sml(x) 
            x = F.tanh(self.post_sml(x))
        else:
            x = F.tanh(self.fc_in(x))
            x = F.tanh(self.post_fc_in(x))

        for layer in self.fc_hidden:
            x = F.tanh(layer(x))

        Ntot = self.fc_out(x)   # Ntot has shape (batch_size, 1)

        # Pass a 1D tensor (of shape (batch_size,)) into get_Nqll and then unsqueeze to restore the shape
        NqllEQ = get_Nqll(Ntot[:, 0]).unsqueeze(1)  
        x = torch.cat([Ntot, NqllEQ], dim=1)  # Concatenate to get a (batch_size, 2) tensor: (Ntot, NqllEQ)
        return x

def enforced_model(coords, model: IcePINN, hard_enforce_IC=True, factor=1.0):
    """
    Wrap IcePINN models in this function for training and evaluation to hard-enforce
    the initial condition using reparameterization. To gradually enforce IC during 
    curriculum learning over an adjustment period, set factor to a fraction representing
    how much of the adjustment period has elapsed. 

    Args:
      coords: Tensor of shape [batch_size, 2], where the first column is x and the second is t.
      model: Neural network that takes the full coordinate tensor [x, t] and outputs a tensor of shape [batch_size, 2],
             corresponding to [NN_tot, NN_qll].
      hard_enforce_IC: Boolean, whether the IC should be enforced. When False, output is like calling model(coords).
               True by default.
      factor: float fraction ranging from (0, 1]. Determines how much the IC is enforced, where 1 is full enforcement. 
              If no enforcement (factor=0) is desired, use enforce_IC=False instead. Used for gradual enforcement of IC during training.
              1.0 by default.
    
    Returns:
      Tensor of shape [batch_size, 2] where:
        - Column 0 is Ntot
        - Column 1 is Nqll
      If enforce_IC=True and factor=1.0, output satisfies Ntot(x, 0) = 1 and Nqll(x, 0) = get_Nqll(Ntot(x, 0)).
      If enforce_IC=True and factor<1.0, output will partially enforce these conditions.
    """        
    # Get the raw outputs from the network
    nn_out = model(coords)  # Expecting shape [batch_size, 2]
    
    if hard_enforce_IC:
        t = coords[:, 1:2]
        NN_tot = nn_out[:, 0:1]
        NN_qll = nn_out[:, 1:2]

        # Scale t values if necessary
        if factor != 1.0:
            t = 1+(t-1)*factor
        
        # Reparameterize to enforce the initial conditions
        # Ntot: initial condition is 1. factor = 1 when IC is fully enforced.
        Ntot = factor + t * NN_tot
        
        # Nqll: initial condition is get_Nqll(Ntot_initial)
        Nqll = (factor*get_Nqll(torch.tensor(1.0))) + t * NN_qll
        
        # Output as tensor with same shape as coords ([batch_size, 2])
        return torch.cat([Ntot, Nqll], dim=1)

    # If enforce is False, return output of model(coords)
    return nn_out

def get_Nqll(Ntot):
    """
    Gets the EQUILIBRIUM LEVEL of Nqll for a given Ntot.
    """
    # This torch.tensor wrapper is a band-aid solution to it throwing a fit
    # when I call this function with respect to device mismatching. Could probably fix.
    # Currently this solution causes a "PyTorch is unhappy" print warning.
    return torch.tensor(NBAR - NSTAR*torch.sin(2*np.pi*Ntot)).to(device)

def init_HE(m):
    # We don't want to interfere with custom initialization of SinusoidalMappingLayer
    if type(m) == nn.Linear:
	    nn.init.kaiming_normal_(m.weight)

def get_device():
    return device

def calc_loss_gradients(xs, ys, diffusion):
    """
    Computes gradients needed to compute collocation point loss. 
    Expects batched input.
    Args:
        xs: xs[0] = x, xs[1] = t
        ys: ys[0] = Ntot, ys[1] = Nqll

    Returns:
        (dNtot_dt, dNqll_dt, dNqll_dxx)
    """
    # batch_size = len(xs)
    xs.requires_grad_(True)  # Enable gradients for input (shape: (batch_size, 2))
    
    # ---- Compute First-Order Derivatives ----
    # We'll compute the following:
    #   Ntot_t = ∂Ntot/∂t, and
    #   Nqll_t = ∂Nqll/∂t
    #
    # Here, the input is [x, t] so t is at index 1.

    # Create grad_outputs tensor matching shape of model output
    grad_outputs = torch.zeros_like(ys).to(device)

    # 1. Compute gradient of Ntot:
    grad_outputs[:, 0] = 1.0  # We want gradients for the first output (Ntot)
    
    grad_Ntot = torch.autograd.grad(
        outputs=ys,
        inputs=xs,
        grad_outputs=grad_outputs,
        create_graph=True,      # I think graph is needed for backprop later
        retain_graph=True,
        materialize_grads=True      # default is false
    )[0]
    dNtot_t = grad_Ntot[:, 1]   # Extract ∂Ntot/∂t

    # 2. Compute gradient of Nqll:
    grad_outputs.zero_()      # Reset grad_outputs to zeros
    grad_outputs[:, 1] = 1.0  # Now we want gradients for the second output (Nqll)
    grad_Nqll = torch.autograd.grad(
        outputs=ys,
        inputs=xs,
        grad_outputs=grad_outputs,
        create_graph=True,      # Needed to compute the second-order derivative
        retain_graph=True,      # Retain graph for subsequent derivative computations
        materialize_grads=True 
    )[0]
    dNqll_t = grad_Nqll[:, 1]   # Extract ∂Nqll/∂t

    if diffusion:
        # ---- Compute Second-Order Derivative ----
        # Note that from grad_Nqll we already have ∂Nqll/∂x:
        dNqll_x = grad_Nqll[:, 0]

        # We take the gradient of Nqll_x with respect to the initial inputs.
        grad_Nqll_x = torch.autograd.grad(
            outputs=dNqll_x,
            inputs=xs,
            grad_outputs=torch.ones_like(dNqll_x),
            create_graph=True,      # I think graph is needed for backprop later
            retain_graph=True,
            materialize_grads=True 
        )[0]
        dNqll_xx = grad_Nqll_x[:, 0]  # Extract ∂²Nqll/∂x² (derivative with respect to x)
        dNqll_xx.detach()
    else:
        # No diffusion -> second derivative not needed.
        dNqll_xx = None

    # Return relevant gradients
    return dNtot_t.detach(), dNqll_t.detach(), dNqll_xx
    # TODO - verify why detach() is important here

def get_misc_params():
    # TODO - make sigma I calculation work for user-specified collocation points, ie. for random resampling
    # Supersaturation reduction at center
    c_r = GI['c_r']

    # Thickness of monolayers
    h_pr = GI['h_pr']
    h_pr_units = GI['h_pr_units']
    h_pr = AssignQuantity(h_pr,h_pr_units)
    h_pr.ito('micrometer')

    # Deposition velocity
    nu_kin = GI['nu_kin']
    nu_kin_units = GI['nu_kin_units']
    nu_kin = AssignQuantity(nu_kin,nu_kin_units)

    # Difference in equilibrium supersaturation between microsurfaces I and II
    sigma0 = torch.tensor(GI['sigma0']).to(device)

    # Supersaturation at facet corner
    sigmaI_corner = GI['sigmaI_corner']

    # Time constant for freezing/thawing
    tau_eq = GI['tau_eq']
    tau_eq_units = GI['tau_eq_units']
    tau_eq = AssignQuantity(tau_eq,tau_eq_units)

    # Compute omega_kin
    nu_kin_mlyperus = nu_kin/h_pr
    nu_kin_mlyperus.ito('1/microsecond')
    omega_kin = torch.tensor(nu_kin_mlyperus.magnitude * tau_eq.magnitude).to(device)

    # Compute sigmaI
    sigmaI = torch.tensor(sigmaI_corner*(c_r*(X_QLC/L)**2+1-c_r))
    # Concatenate a copy of sigmaI for each timestep to prevent shape inconsistencies later
    sigmaI = torch.cat([sigmaI]*NUM_T_STEPS).to(device)
    
    # sigma0, sigmaI, omega_kin = params
    return sigma0, sigmaI, omega_kin

def calc_QLC_rhs(Ntot, Nqll, dNqll_dxx):
    """
    This function computes the right-hand side of the two objective functions 
    that form the QLC system. Thus the outputs should theoretically be equal to 
    dNtot_dt and dNqll_dt. The outputs of this function are primarily used
    to compute collocation point loss. 
    
    If you want to disable diffusion, pass None to dNqll_dxx.

    This function is adapted from f1d_solve_ivp_dimensionless() in QLCstuff2.py
    
    Returns:
        [dNtot_dt, dNqll_dt], or a 2D tensor containing len(input tensor) (dNtot_dt, dNqll_dt) pairs
    """
    sigma0, sigmaI, omega_kin = ip.get_misc_params()

    # Ntot deposition
    m = (Nqll - (NBAR - NSTAR))/(2*NSTAR)
    sigma_m = (sigmaI - m * sigma0)
    dNtot_dt = omega_kin * sigma_m
    
    # If diffusion is active, add diffusion term to dNtot_dt
    if dNqll_dxx is not None:
        dNtot_dt += dNqll_dxx 
    # NQLL    
    dNqll_dt = dNtot_dt - (Nqll - (NBAR - NSTAR*torch.sin(2*np.pi*Ntot)))
    
    # Package for output
    return dNtot_dt, dNqll_dt

def calc_IcePINN_loss(
    model: IcePINN, 
    coords, 
    diffusion=True, 
    epoch=0, 
    enforce_IC=True, 
    hard_enforce_IC=True, 
    adjustment_period=0):

    """
    Custom loss function for IcePINNs. Given a set of training points, how incorrect
    is the network prediction?
I
    Args:
        model: IcePNN to be evaluated using 'coords'.
        coords: (x, t), or a tensor of shape [batch_size, 2] containing batch_size (x, t) coordinate pairs.
            These are the training points that 'model' is evaluated on.
        diffusion: True if diffusion is enabled, False otherwise. True by default.
        epoch: Number of training epochs elapsed thus far. 0 by default. Not relevant
            unless doing curriculum learning.
        enforce_IC: True if IC is being enforced, False otherwise. True by default.
        hard_enforce_IC: True if IC is being hard enforced, False if being soft enforced
            (which is the classical PINN approach). True by default.
        adjustment_period: Number of epochs it takes to have IC fully enforced. 0 by default. 
            Not relevant unless doing curriculum learning.

    Returns:
        (Ntot-loss, Nqll-loss), or a tensor of input shape containing (Ntot-loss, Nqll-loss) pairs.
    """
    adjustment_factor = 1.0
    if hard_enforce_IC and (epoch < adjustment_period):
        adjustment_factor = np.sqrt(epoch/adjustment_period)
    
    # model predicts output of training_set as batch
    ys = enforced_model(coords, model, hard_enforce_IC=hard_enforce_IC, factor=adjustment_factor)
    
    # Calculate and extract gradients
    dNtot_dt, dNqll_dt, dNqll_dxx = ip.calc_loss_gradients(coords, ys, diffusion)

    # Compute expected output
    Ntot, Nqll = ys[:, 0], ys[:, 1]
    dNtot_dt_rhs, dNqll_dt_rhs = ip.calc_QLC_rhs(Ntot, Nqll, dNqll_dxx)
    
    # dNtot_dt = Nqll*surface_diff_coefficient + w_kin*sigma_m
    # dNqll_dt = dNtot/dt - (Nqll - Nqll_eq)
    unsquared_loss = torch.stack(((dNtot_dt - dNtot_dt_rhs), (dNqll_dt - dNqll_dt_rhs)), axis=1)

    if enforce_IC and not hard_enforce_IC:
        # Soft IC enforcement: add penalty for deviation from IC.
        # NOTE: nx_crystal MUST be consistent with the number of points being sampled
        #   at t=0, or this will not penalize as intended. If implementing random 
        #   collocation points each epoch, will have to do dedicated loss calculation
        #   for a set of points at t=0 instead of re-using the first nx_crystal predictions.

        # Predictions at t=0
        Ntot_t0, Nqll_t0 = Ntot[0:nx_crystal], Nqll[0:nx_crystal]

        # Difference between predictions at t=0 and IC
        unsquared_IC_loss = torch.stack((Ntot_t0 - Ntot_init, Nqll_t0 - Nqll_init), axis=1)
        
        # Combine losses
        unsquared_loss = torch.concat((unsquared_IC_loss, unsquared_loss))
    
    # Return L2 loss (squared loss) as tensor of shape (len(coords), 2)
    # [dNtot_dt - dNtot_dt_rhs, dNqll_dt - dNqll_dt_rhs]
    return torch.square(unsquared_loss)

def train_IcePINN(
    model: IcePINN, 
    optimizer, 
    training_set, 
    epochs, 
    name, 
    print_every=1_000, 
    diffusion=True, 
    LR_scheduler=None, 
    enforce_IC=True,
    hard_enforce_IC=True,
    adjustment_period=0):
    """
    This function serves as the training loop for IcePINNs. The model weights
    are updated and stored within the model itself, so changes will be saved 
    even if this function is terminated prematurely. After the first epoch
    (or after the adjustment period has passed, if set),
    model weights will only be saved to disc if the Ntot loss or Nqll loss
    is lower than the lowest seen so far during this training run. 
    Model weights will not be saved during the adjustment period if one is set.

    Args:
        model: Instantiated IcePINN model to train.
        optimizer: Instantiated optimizer from the torch.optim library.
        training_set: Set of points to train 'model' on.
        epochs: Number of training epochs/iterations.
        name: String name of model. Used to name the directory in which 'model' is saved.
        print_every: How often training statistics are printed. 1000 by default.
        diffusion: Whether model is being trained with NQLL surface diffusion active.
            True by default. 
        LR_scheduler: Instantiated learning rate scheduler from the torch.optim.lr_scheduler
            library. Must be instantiated on same optimizer as 'optimizer' argument. None by default.
        enforce_IC: Whether the initial condition is being enforced in training or not. True by default.
        hard_enforce_IC: True if IC is being hard enforced, False if being soft enforced
            (which is the classical PINN approach). True by default. Ignored if enforce_IC is False.
        adjustment_period: Number of epochs it takes to have IC fully enforced. 0 by default. 
            Not relevant unless doing curriculum learning. 

    """

    save_path = './models/'+name

    # Create folder to store model in if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Commencing PINN training on {len(training_set)} points for {epochs} epochs.")

    if enforce_IC:
        save_path +='/params.pth'
        print(f"IC is enforced with an adjustment period of {adjustment_period}.")
    else:
        save_path +='/pre_IC_params.pth'
        print("IC is not enforced.")

    ## Concepts of a plan, but would require reworking load_model() and I can't deal
    # if enforce_IC:
    #     best_save_path +='/best_params.pth'
    #     last_save_path +='/last_params.pth'
    #     print(f"IC is enforced with an adjustment period of {adjustment_period}.")
    # else:
    #     best_save_path +='/best_pre_IC_params.pth'
    #     last_save_path +='/last_pre_IC_params.pth'
    #     print("IC is not enforced.")

    # Used to determine when to save model weights to disk
    min_Ntot_loss = 10e12
    min_Nqll_loss = 10e12
    best_model_Ntot_loss = 10e12
    best_model_Nqll_loss = 10e12
    best_model_epoch = -1

    start_time = time.time()    

    for epoch in range(epochs):
        # Zero accumulated gradients so they don't affect future computations
        optimizer.zero_grad() 

        # evaluate training loss
        loss = ip.calc_IcePINN_loss(
            model, 
            training_set,  
            diffusion=diffusion, 
            epoch=epoch,  
            enforce_IC = enforce_IC,
            hard_enforce_IC = hard_enforce_IC,
            adjustment_period = adjustment_period)
        Ntot_loss = torch.sum(loss[:, 0]).item()
        Nqll_loss = torch.sum(loss[:, 1]).item()

        # This heuristic for best model isn't perfect, but it's a good start
        # Models will not be saved during adjustment period
        if epoch >= adjustment_period and (Ntot_loss < min_Ntot_loss or Nqll_loss < min_Nqll_loss):
            # TODO: determine which model to save using a discrete testing set?

            # Save model (not including initial condition wrapper)
            torch.save(model.state_dict(), save_path)
            best_model_epoch = epoch
            if Ntot_loss < min_Ntot_loss:
                min_Ntot_loss = Ntot_loss
            else:
                min_Nqll_loss = Nqll_loss
            best_model_Ntot_loss = Ntot_loss
            best_model_Nqll_loss = Nqll_loss
        
        if LR_scheduler is not None:
            LR_scheduler.step(Ntot_loss+Nqll_loss)

        # Print training progress in [print_every] intervals
        if ((epoch+1) % print_every) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            this_epoch = epoch+1
            if (print_every % 1000) == 0:
                this_epoch = this_epoch // 1000
            else:
                this_epoch = this_epoch / 1000
            print(f"Epoch [{this_epoch}k/{epochs//1000}k] at {minutes}m {seconds}s: Ntot = {Ntot_loss:.3f}, Nqll = {Nqll_loss:.3f}, LR = {optimizer.param_groups[0]['lr']}")

        # Print time completion estimate and best model saved thus far
        # as each tenth of total training epochs elapses
        if (epoch+1) % (epochs//10) == 0:
            elapsed = time.time() - start_time
            tenths_elapsed = (epoch+1) // (epochs//10)
            tenths_remaining = 10 - tenths_elapsed
            time_remaining = (elapsed // tenths_elapsed) * tenths_remaining
            completion_estimate = (elapsed // tenths_elapsed) * 10

            c_minutes = int(completion_estimate // 60)
            c_seconds = int(completion_estimate % 60)
            r_minutes = int(time_remaining // 60)
            r_seconds = int(time_remaining % 60)
            if epochs - epoch+1 > epochs//10:
                print(f"Training {tenths_elapsed}/10ths complete! Completion estimate: {c_minutes}m {c_seconds}s | {r_minutes}m {r_seconds}s remaining.")
            
            if epoch > adjustment_period:
                print(f'Best model saved so far: Epoch {best_model_epoch+1}; Loss: {best_model_Ntot_loss:.3f} Ntot, {best_model_Nqll_loss:.3f} Nqll')
            
        if epoch+1 == adjustment_period:
            print("Adjustment period is complete! IC is now being fully enforced.")
            
        # backward and optimize
        loss.backward(torch.ones_like(loss)) # Computes loss gradients

        # Gradient clipping to mitigate exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

        optimizer.step() # Adjusts weights accordingly

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Training complete after {minutes} minutes and {seconds} seconds')
    print(f'Model {name} from epoch {best_model_epoch+1} has been saved.')
    print(f'Saved model Ntot loss: {best_model_Ntot_loss:.3f}.')
    print(f'Saved model Nqll loss: {best_model_Nqll_loss:.3f}.')

def load_IcePINN(model_name, pre_IC=False):
    """
    Loads a saved IcePINN from disc.
    Args:
        model_name: String name of folder model is stored in
        pre_IC: Boolean; should the best pre initial condition enforcement model
                be loaded? Default is False, which loads best model trained with IC enforced.
    
    Returns a freshly loaded IcePINN model instance.
    """
    path = './models/'+model_name
    if pre_IC:
        path +='/pre_IC_params.pth'
    else:
        path +='/params.pth'
    state_dict = torch.load(path)
    model_dimensions = state_dict['dimensions']
    is_sf_PINN = state_dict['is_sf_PINN']

    loaded_model = ip.IcePINN(
        num_hidden_layers=model_dimensions[0], 
        hidden_layer_size=model_dimensions[1], 
        is_sf_PINN=is_sf_PINN.item())

    # match buffers with the model being loaded
    loaded_model.register_buffer('dimensions', model_dimensions)
    loaded_model.register_buffer('is_sf_PINN', is_sf_PINN)

    loaded_model.load_state_dict(state_dict, strict=False) # takes the loaded dictionary, not the path file itself
    loaded_model.to(device)
    
    return loaded_model

#region Data preparation/pre-processing

# Read in GI parameters
inputfile = "GI parameters - Reference limit cycle (for testing).nml"
GI=f90nml.read(inputfile)['GI']
nx_crystal = GI['nx_crystal']
L = GI['L']
NBAR = GI['Nbar']
NSTAR = GI['Nstar']

# Define t range (needs to be same as training file)
RUNTIME = 2
NUM_T_STEPS = 100*RUNTIME + 1
#NUM_T_STEPS = RUNTIME*5 + 1

# Define initial conditions
Ntot_init = torch.ones(nx_crystal).to(device)
Nqll_init = get_Nqll(Ntot_init)

# Define x, t pairs for training
X_QLC = np.linspace(-L,L,nx_crystal)
t_points = np.linspace(0, RUNTIME, NUM_T_STEPS)
x, t = np.meshgrid(X_QLC, t_points)
training_set = torch.tensor(np.column_stack((x.flatten(), t.flatten()))).to(device)

#endregion



