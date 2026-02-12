from typing import Union, Tuple

import torch

from deepchem.utils.dft_utils.data.datastruct import AtomZsType, AtomPosType
from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
from deepchem.utils.dft_utils.system.base_system import BaseSystem


class PWSol(BaseSystem):
    def __init__(
        self,
        lattice: Lattice,
        e_cut: float,
        k_vector: torch.Tensor,
        
    ):
        """
        Parameters
        ----------
        ecut: torch.FloatTensor
            
        """

        g_cut = (2 * e_cut) ** 0.5
        g_vector, weights = lattice.get_gvgrids(g_cut)

        q = g_vector + k_vector

        k_energy = 0.5 * torch.sum(q * q, dim = 1)

        mask = k_energy <= e_cut
        self.g_vector = g_vector[mask]
        self.k_energy = k_energy[mask]

if __name__ == "__main__":
    lattice = Lattice(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64))
    pwsol = PWSol(lattice, 20, torch.ones([3]) * 2)
    print(pwsol.g_vector)
    print(pwsol.k_energy)
