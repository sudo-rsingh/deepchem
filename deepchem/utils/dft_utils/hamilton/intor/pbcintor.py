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

    This dataclass encapsulates parameters that control the behavior of lattice
    sums and k-point handling in periodic systems. It provides sensible defaults
    while allowing customization for specific calculation requirements.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import PBCIntOption
    >>> 
    >>> # Create with default values
    >>> pbc = PBCIntOption()
    >>> print(pbc.precision, pbc.kpt_diff_tol)
    1e-08 1e-06
    >>> 
    >>> # Create with custom precision
    >>> pbc_custom = PBCIntOption(precision=1e-10, kpt_diff_tol=1e-8)
    >>> print(pbc_custom)
    PBCIntOption(precision=1e-10, kpt_diff_tol=1e-08)
    >>> 
    >>> # Get default instance
    >>> default_pbc = PBCIntOption.get_default()
    >>> print(default_pbc)
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)
    >>> 
    >>> # Create from dictionary
    >>> pbc_dict = PBCIntOption.get_default({"precision": 1e-9, "kpt_diff_tol": 1e-7})
    >>> print(pbc_dict)
    PBCIntOption(precision=1e-09, kpt_diff_tol=1e-07)

    Attributes
    ----------
    precision : float, default 1e-8
        Target precision for lattice sum convergence. This parameter determines
        when to truncate infinite lattice sums based on the contribution magnitude.
        Smaller values yield higher accuracy but increased computational cost.
        
    kpt_diff_tol : float, default 1e-6
        Tolerance for treating two k-points as equivalent. During k-point sampling,
        points whose difference is below this threshold are considered the same
        to avoid numerical issues and redundant calculations.

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
    """Get default PBC integral options or return provided options.

    This function provides a convenient way to ensure that valid PBC integral
    options are available. If no options are provided, it returns a default
    PBCIntOption instance with standard parameters. If options are provided,
    they are returned unchanged.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_default_options, PBCIntOption
    >>> 
    >>> # Get default options
    >>> default_opts = get_default_options()
    >>> print(default_opts)
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)
    >>> 
    >>> # Use custom options
    >>> custom_opts = PBCIntOption(precision=1e-10, kpt_diff_tol=1e-8)
    >>> returned_opts = get_default_options(custom_opts)
    >>> print(returned_opts is custom_opts)
    True
    >>> 
    >>> # Pass None to get defaults
    >>> opts_from_none = get_default_options(None)
    >>> print(opts_from_none.precision)
    1e-08

    Parameters
    ----------
    options : Optional[PBCIntOption], default None
        Input options object. If None, a default PBCIntOption is returned.

    Returns
    -------
    PBCIntOption
        The provided options object or a new default instance.

    """
    if options is None:
        options1 = PBCIntOption()
    else:
        options1 = options
    return options1


def get_default_kpts(kpts: Optional[torch.Tensor], dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
    """Get default k-points or ensure existing k-points have correct format.

    This function ensures that k-points are properly formatted for PBC calculations.
    If no k-points are provided, it returns a default k-point at the origin (Gamma point).
    If k-points are provided, it ensures they have the correct dtype and device.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import get_default_kpts
    >>> 
    >>> # Get default k-points (Gamma point)
    >>> default_kpts = get_default_kpts(None, torch.float64, torch.device('cpu'))
    >>> print(default_kpts)
    tensor([[0., 0., 0.]], dtype=torch.float64)
    >>> 
    >>> # Convert existing k-points to correct format
    >>> input_kpts = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    >>> converted_kpts = get_default_kpts(input_kpts, torch.float64, torch.device('cpu'))
    >>> print(converted_kpts)
    tensor([[1., 1., 1.]], dtype=torch.float64)
    >>> 
    >>> # Multiple k-points
    >>> multi_kpts = torch.tensor([[0, 0, 0], [0.5, 0, 0]], dtype=torch.float64)
    >>> result = get_default_kpts(multi_kpts, torch.float64, torch.device('cpu'))
    >>> print(result.shape)
    torch.Size([2, 3])

    Parameters
    ----------
    kpts : Optional[torch.Tensor]
        Input k-points tensor of shape (n_kpts, 3). If None, returns Gamma point.
    dtype : torch.dtype
        The desired data type for the k-points tensor.
    device : torch.device
        Device on which the tensor should be located (e.g., 'cpu', 'cuda').

    Returns
    -------
    torch.Tensor
        K-points tensor with shape (n_kpts, 3) on the specified device with
        the specified dtype. If kpts is None, returns a single k-point at origin.

    Raises
    ------
    AssertionError
        If the provided kpts tensor is not 2-dimensional or doesn't have
        the correct last dimension (3).

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
    >>> 
    >>> # Import required components (avoiding circular imports)
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
    >>> 
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
    >>> 
    >>> # Create a single wrapper and get subsets (ensures same parent)
    >>> combined_wrapper = LibcintWrapper([atom1, atom2], spherical=True, lattice=lattice)
    >>> wrapper = combined_wrapper[:1]  # First atom
    >>> other_wrapper = combined_wrapper[1:]  # Second atom
    >>> 
    >>> # Case 1: other is None - returns wrapper
    >>> result1 = _check_and_set_pbc(wrapper, None)
    >>> print(result1 is wrapper)
    True
    >>> 
    >>> # Case 2: other is provided and compatible (same lattice)
    >>> result2 = _check_and_set_pbc(wrapper, other_wrapper)
    >>> print(result2 is other_wrapper)
    True
    >>> print(result2.lattice is wrapper.lattice)
    True
    >>> 
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

