from typing import Optional, Sequence, Tuple, Union

import numpy as np
from deepchem.models.torch_models import Logistic
import torch.nn as nn
import torch
from deepchem.utils.dft_utils import Mol
from deepchem.models.torch_models import TorchModel





class DM21Model(TorchModel):
    """
    
    References
    ----------
    ..[1] Roothaan equations. (2022, July 15).
    In Wikipedia. https://en.wikipedia.org/wiki/Roothaan_equations

    ..[2] Orbital overlap. (2024, January 4).
    In Wikipedia. https://en.wikipedia.org/wiki/Orbital_overlap#Overlap_matrix
    """
    def __init__(self, hidden_size: int = 256, n_layers: int = 6, **kwargs):
        model = DM21(hidden_size=hidden_size, n_layers=n_layers)
        super(DM21Model, self).__init__(model, L2Loss(), **kwargs)


class DM21(nn.Module):
    """DM21 accurately models complex systems such as hydrogen chains,
    charged DNA base pairs and diradical transition states. It extends
    DFT (Density Functional Theory) which is a well established method
    for investigating electronic structure of many-body systems.

    Density Functional Theory (DFT) is a quantum mechanical method used
    to investigate the electronic structure (principally the ground state)
    of many-body systems, particularly atoms, molecules, and condensed
    phases. The main idea behind DFT is to describe the complex system
    of interacting particles (usually electrons) using the electron
    density, a function of spatial coordinates, rather than the many-body
    wavefunction.

    DeepMind 21 specifically address two long-standing problems with
    traditional functionals:

    1. The delocalization error: Most existing density functional
    approximations prefer electron densities that are unrealistically
    spread out over several atoms or molecules rather than being
    correctly localized around a single molecule or atom.

    2. Spin symmetry breaking: When describing the breaking of chemical
    bonds, existing functionals tend to unrealistically prefer configurations
    in which a fundamental symmetry known as spin symmetry is broken.

    These longstanding challenges are both related to how functionals
    behave when presented with a system that exhibits
    “fractional electron character.” By using a neural network to
    represent the functional and tailoring our training dataset to
    capture the fractional electron behaviour.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.dft.dm21 import DM21
    >>> model = DM21()
    >>> input = torch.rand((100, 11))
    >>> output = model(input)
    >>> output.shape
    torch.Size([100, 3])

    References
    ----------
    .. [1] Simulating matter on the quantum scale with AI. (2024, May 14). Google DeepMind.
        https://deepmind.google/discover/blog/simulating-matter-on-the-quantum-scale-with-ai/
    .. [2] James Kirkpatrick et al. ,Pushing the frontiers of density functionals by solving
        the fractional electron problem. Science374,1385-1389(2021).DOI:10.1126/science.abj6511
    .. [3] Density functional theory. (2024, April 24).
        In Wikipedia. https://en.wikipedia.org/wiki/Density_functional_theory

    """

    def __init__(self, hidden_size: int = 256, n_layers: int = 6):
        """Initialise the DeepMind 21 Model.

        Parameters
        ----------
        hidden_size: int (default 256)
            Size of Linear/Dense (Fully Connected) Layers to use in the model.
        n_layers: int (default 6)
            Number of Linear/Dense (Fully Connected) Layers to use in the model.

        """
        super(DM21, self).__init__()

        self.hidden_size: int = hidden_size
        self.n_layers: int = n_layers

        # Layer Initialisation
        self.lin_tanh = nn.Linear(11, self.hidden_size)
        self.lin_elu = nn.ModuleList()
        for i in range(self.n_layers):
            self.lin_elu.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.lin_elu.append(nn.LayerNorm(self.hidden_size))
        self.final = nn.Linear(self.hidden_size, 3)
        self.acti_tanh = nn.Tanh()
        self.acti_elu = nn.ELU()
        self.acti_scaled_sigmoid = Logistic(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Method for the DeepMind 21 Model.

        Parameters
        ----------
        x: torch.Tensor
            A torch tensor with 11 features.

        Returns
        -------
        torch.Tensor
            Predicted Output of the model.

        """
        x = torch.log(torch.abs(x) + torch.tensor([1e-4], device=x.device))
        x = self.acti_tanh(self.lin_tanh(x))
        for i in range(self.n_layers):
            x = self.acti_elu(self.lin_elu[i](x))
            x = self.lin_elu[2*i+1](x)
        x = self.acti_scaled_sigmoid(self.final(x))
        return x


class DM21Features:
    grid_coords: np.ndarray
    grid_weights: np.ndarray
    rho_a: np.ndarray
    rho_b: np.ndarray
    tau_a: np.ndarray
    tau_b: np.ndarray
    norm_grad_rho_a: np.ndarray
    norm_grad_rho_b: np.ndarray
    norm_grad_rho: np.ndarray
    hfxa: np.ndarray
    hfxb: np.ndarray


def construct_function_inputs(
    mol: Mol,
    dms: Union[np.ndarray, Sequence[np.ndarray]],
    spin: int,
    coords: np.ndarray,
    weights: np.ndarray,
    rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    ao: Optional[np.ndarray] = None
):
    if spin == 0:
        # RKS
        rhoa = rho / 2
        rhob = rho / 2
    else:
        # UKS
        rhoa, rhob = rho
    
    # Local HF features.
    exxa, exxb = [], []
    fxxa, fxxb = [], []
    """
    for omega in sorted(self._omega_values):
      hfx_results = compute_hfx_density.get_hf_density(
          mol,
          dms,
          coords=coords,
          omega=omega,
          deriv=1,
          ao=ao)
      exxa.append(hfx_results.exx[0])
      exxb.append(hfx_results.exx[1])
      fxxa.append(hfx_results.fxx[0])
      fxxb.append(hfx_results.fxx[1])
    exxa = np.stack(exxa, axis=-1)
    fxxa = np.stack(fxxa, axis=-1)
    """