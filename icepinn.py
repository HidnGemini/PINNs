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
    def __init__(self, num_hidden_layers, hidden_layer_size, is_sf_PINN=False):
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
            
        self.fc_out = nn.Linear(hidden_layer_size, 2)

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

        x = self.fc_out(x)
        return x

def enforced_model(coords, model: IcePINN):
    """
    Wrap IcePINN models in this function for training and evaluation to hard-enforce
    the initial condition using reparameterization.

    Args:
      coords: Tensor of shape [batch_size, 2], where the first column is x and the second is t.
      model: Neural network that takes the full coordinate tensor [x, t] and outputs a tensor of shape [batch_size, 2],
             corresponding to [NN_tot, NN_qll].
    
    Returns:
      Tensor of shape [batch_size, 2] where:
        - Column 0 is Ntot, satisfying Ntot(x, 0) = 1.
        - Column 1 is Nqll, satisfying Nqll(x, 0) = get_Nqll(Ntot(x, 0)).
    """    
    # Extract t from the coordinates tensor
    t = coords[:, 1:2]
    
    # Get the raw outputs from the network
    nn_out = model(coords)  # Expecting shape [batch_size, 2]
    NN_tot = nn_out[:, 0:1]
    NN_qll = nn_out[:, 1:2]
    
    # Reparameterize to enforce the initial conditions:
    # For Ntot: initial condition is 1.
    Ntot = 1.0 + t * NN_tot
    
    # For Nqll: initial condition is get_Nqll(1)
    Nqll = get_Nqll(1.0) + t * NN_qll
    
    # Concatenate the outputs to form a tensor with the same shape as coords ([batch_size, 2])
    return torch.cat([Ntot, Nqll], dim=1)

def get_Nqll(Ntot):
    return NBAR - NSTAR*np.sin(2*np.pi*Ntot)

def init_HE(m):
    # We don't want to interfere with custom initialization of SinusoidalMappingLayer
    if type(m) == nn.Linear:
	    nn.init.kaiming_normal_(m.weight)

def get_device():
    return device

def compute_loss_gradients(xs, ys, diffusion):
    """
    Computes gradients needed to compute collocation point loss. 
    Expects batched input with requires_grad=TRUE.
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

def f1d_solve_ivp_dimensionless(Ntot, Nqll, dNqll_dxx, scalar_params):
    """
    Adapted from QLCstuff2, this function computes the right-hand side of
    the two objective functions that make up the QLC system. If you want
    to disable diffusion, pass None to dNqll_dxx.
    
    Returns:
        [dNtot_dt, dNqll_dt]
    """
    sigma0, sigmaI, omega_kin = scalar_params

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

def calc_cp_loss(model: IcePINN, coords, params, epochs, epoch, print_every, print_gradients, diffusion):
    """Calculates collocation point loss.

    Args:
        coords: (x, t), or a tensor of size [2, batch_size] containing batch_size (x, t) coordinate pairs
        params: output of get_misc_params()

    Returns:
        (Ntot-loss, Nqll-loss), or a tensor of input shape containing (Ntot-loss, Nqll-loss) pairs.
    """
    # model predicts output of training_set as batch
    ys = enforced_model(coords, model)
    
    # Calculate and extract gradients
    dNtot_dt, dNqll_dt, dNqll_dxx = ip.compute_loss_gradients(coords, ys, diffusion)
    if (((epoch+1) % print_every) == 0) and print_gradients:
        print(f"Gradients: {dNtot_dt}, {dNqll_dt}, {dNqll_dxx}.")

    # Compute expected output
    Ntot, Nqll = ys[:, 0], ys[:, 1]
    dNtot_dt_rhs, dNqll_dt_rhs = ip.f1d_solve_ivp_dimensionless(Ntot, Nqll, dNqll_dxx, params)
    
    # dNtot_dt = Nqll*surface_diff_coefficient + w_kin*sigma_m
    # dNqll_dt = dNtot/dt - (Nqll - Nqll_eq)
    cat_test = torch.stack((dNtot_dt - dNtot_dt_rhs, dNqll_dt - dNqll_dt_rhs), axis=1)
    
    # Return squared loss as tensor of shape (len(coords), 2)
    # [dNtot_dt - dNtot_dt_rhs, dNqll_dt - dNqll_dt_rhs]
    return torch.square(cat_test)

def train_IcePINN(model: IcePINN, optimizer, training_set, epochs, name, print_every, print_gradients=False, diffusion=True, LR_scheduler=None):

    save_path = './models/'+name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Commencing PINN training on {len(training_set)} points for {epochs} epochs.")

    # Retrieve miscellaneous params for loss calculation
    params = ip.get_misc_params()
    min_Ntot_loss = 10e12
    min_Nqll_loss = 10e12
    best_model_Ntot_loss = 10e12
    best_model_Nqll_loss = 10e12
    best_model_epoch = -1

    start_time = time.time()    

    for epoch in range(epochs):
        # Zero accumulated gradients so they don't affect future computations
        optimizer.zero_grad() 

        # evaluate collocation point loss
        loss = ip.calc_cp_loss(model, training_set, params, epochs, epoch, print_every, print_gradients, diffusion)
        Ntot_loss = torch.sum(loss[:, 0]).item()
        Nqll_loss = torch.sum(loss[:, 1]).item()

        if LR_scheduler is not None:
            LR_scheduler.step(Ntot_loss+Nqll_loss)

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

        if ((epoch+1) % (epochs//10)) == 0:
            elapsed = time.time() - start_time
            tenths_elapsed = (epoch+1) // (epochs//10)
            tenths_remaining = 10 - tenths_elapsed
            time_remaining = (elapsed // tenths_elapsed) * tenths_remaining
            minutes = int(time_remaining // 60)
            seconds = int(time_remaining % 60)
            if tenths_remaining is not 0:
                print(f"Training {tenths_elapsed}/10ths complete! Estimated time remaining: {minutes}m {seconds}s")
            
        # backward and optimize
        loss.backward(torch.ones_like(loss)) # Computes loss gradients
        optimizer.step() # Adjusts weights accordingly

        # This heuristic for best model isn't perfect, but it's a good start
        if Ntot_loss < min_Ntot_loss or Nqll_loss < min_Nqll_loss:
            # Save model (not including initial condition wrapper)
            torch.save(model.state_dict(), save_path+'/params.pth')
            best_model_epoch = epoch
            if Ntot_loss < min_Ntot_loss:
                min_Ntot_loss = Ntot_loss
            else:
                min_Nqll_loss = Nqll_loss
            best_model_Ntot_loss = Ntot_loss
            best_model_Nqll_loss = Nqll_loss

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Training complete after {minutes} minutes and {seconds} seconds')
    print(f'Model {name} from epoch {best_model_epoch+1} has been saved.')
    print(f'Saved model Ntot loss: {best_model_Ntot_loss:.3f}.')
    print(f'Saved model Nqll loss: {best_model_Nqll_loss:.3f}.')

def load_IcePINN(model_name):
    """
    Loads a saved IcePINN.
    Args:
        model_name: String name of folder model is stored in
    """
    path = './models/'+model_name+'/params.pth'
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
RUNTIME = 5
NUM_T_STEPS = RUNTIME + 1
#NUM_T_STEPS = RUNTIME*5 + 1

# Define initial conditions
Ntot_init = np.ones(nx_crystal)
Nqll_init = get_Nqll(Ntot_init)

# Define x, t pairs for training
X_QLC = np.linspace(-L,L,nx_crystal)
t_points = np.linspace(0, RUNTIME, NUM_T_STEPS)
x, t = np.meshgrid(X_QLC, t_points)
training_set = torch.tensor(np.column_stack((x.flatten(), t.flatten()))).to(device)

#endregion



