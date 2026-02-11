from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional, Union, Dict
from deepchem.utils.dft_utils import LibcintWrapper
from deepchem.utils.dft_utils.hamilton.intor.molintor import _check_and_set
from deepchem.utils.dft_utils.hamilton.intor.utils import NDIM


@dataclass
class PBCIntOption:
    """Configuration class for periodic boundary condition (PBC) integral parameters.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import PBCIntOption
    >>> pbc = PBCIntOption()
    PBCIntOption(precision=1e-10, kpt_diff_tol=1e-08)

    Attributes
    ----------
    precision: float (default 1e-8)
        Precision of the integral to limit the lattice sum.       
    kpt_diff_tol : float (default 1e-6)
        Difference between k-points to be regarded as the same.

    """
    precision: float = 1e-8
    kpt_diff_tol: float = 1e-6

    @staticmethod
    def get_default(
        lattsum_opt: Optional[Union[PBCIntOption,
                                    Dict]] = None) -> PBCIntOption:
        """Get the default PBCIntOption object.

        Parameters
        ----------
        lattsum_opt: Optional[Union[PBCIntOption, Dict]]
            The lattice sum option. If it is a dictionary, then it will be
            converted to a PBCIntOption object. If it is None, then just use
            the default value of PBCIntOption.

        Returns
        -------
        PBCIntOption
            The default PBCIntOption object.

        """
        if lattsum_opt is None:
            return PBCIntOption()
        elif isinstance(lattsum_opt, dict):
            return PBCIntOption(**lattsum_opt)
        else:
            return lattsum_opt


# helper functions
def get_default_options(options: Optional[PBCIntOption] = None) -> PBCIntOption:
    """if options is None, then set the default option.
    otherwise, just return the input options.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_default_options
    >>> get_default_options()
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)

    Parameters
    ----------
    options: Optional[PBCIntOption]
        Input options object

    Returns
    -------
    PBCIntOption
        The options object

    """
    if options is None:
        options1 = PBCIntOption()
    else:
        options1 = options
    return options1


def get_default_kpts(kpts: Optional[torch.Tensor], dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
    """if kpts is None, then set the default kpts (k = zeros)
    otherwise, just return the input kpts in the correct dtype and device

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_default_kpts
    >>> get_default_kpts(torch.tensor([[1, 1, 1]]), torch.float64, 'cpu')
    tensor([[1., 1., 1.]], dtype=torch.float64)

    Parameters
    ----------
    kpts: Optional[torch.Tensor]
        Input k-points
    dtype: torch.dtype
        The dtype of the kpts
    device: torch.device
        Device on which the tensord are located. Ex: cuda, cpu

    Returns
    -------
    torch.Tensor
        Default k-points

    """
    if kpts is None:
        kpts1 = torch.zeros((1, NDIM), dtype=dtype, device=device)
    else:
        kpts1 = kpts.to(dtype).to(device)
        assert kpts1.ndim == 2
        assert kpts1.shape[-1] == NDIM
    return kpts1


def _check_and_set_pbc(wrapper: LibcintWrapper, other: Optional[LibcintWrapper]) -> LibcintWrapper:
    """Check and set the `other` parameter for PBC integrals.

    This function verifies that the `other` parameter is compatible with the `wrapper`
    for periodic boundary condition calculations, then returns the appropriate
    `other` parameter (sets to `wrapper` if it is `None`).

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.hamilton.intor.pbcintor import _check_and_set_pbc
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
    >>> # Create a shared lattice
    >>> a = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    >>> lattice = Lattice(a)
    >>> 
    >>> # Create two atoms with shared basis
    >>> pos1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    >>> pos2 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    >>> basis = loadbasis("1:STO-3G", dtype=torch.float64, requires_grad=False)
    >>> atom1 = AtomCGTOBasis(atomz=1, bases=basis, pos=pos1)
    >>> atom2 = AtomCGTOBasis(atomz=1, bases=basis, pos=pos2)
    >>> # Create a single wrapper and get subsets (ensures same parent)
    >>> combined_wrapper = LibcintWrapper([atom1, atom2], spherical=True, lattice=lattice)
    >>> wrapper = combined_wrapper[:1]  # First atom
    >>> other_wrapper = combined_wrapper[1:]  # Second atom
    >>> # Case 1: other is None - returns wrapper
    >>> result1 = _check_and_set_pbc(wrapper, None)
    >>> print(result1 is wrapper)
    True
    >>> # Case 2: other is provided and compatible (same lattice)
    >>> result2 = _check_and_set_pbc(wrapper, other_wrapper)
    >>> print(result2 is other_wrapper)
    True
    >>> print(result2.lattice is wrapper.lattice)
    True
    >>> # Case 3: other has different lattice (raises AssertionError)
    >>> a_diff = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    >>> lattice_diff = Lattice(a_diff)
    >>> wrapper_diff = LibcintWrapper([atom2], spherical=True, lattice=lattice_diff)


    Parameters
    ----------
    wrapper : LibcintWrapper
        Primary wrapper object containing lattice information.
    other : Optional[LibcintWrapper]
        Secondary wrapper object to be checked for compatibility.

    Returns
    -------
    LibcintWrapper
        The validated `other` parameter.

    Raises
    ------
    AssertionError
        If the lattice of `other` is not the same as `wrapper`.
    """
    other1 = _check_and_set(wrapper, other)
    assert other1.lattice is wrapper.lattice
    return other1

