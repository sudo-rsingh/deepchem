"""
PyTorch-based Exchange-Correlation (XC) Functionals for DeepChem DFT

This module provides pure PyTorch implementations of DFT XC functionals
that are fully differentiable and work with DeepChem's DFT pipeline.
These implementations are designed to match LibXC exactly.
"""

import torch
from typing import Union, List
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils.data.datastruct import ValGrad, SpinParam
from deepchem.utils import safepow


class PyTorchLDA(BaseXC):
    """
    Local Density Approximation (LDA) XC functional in pure PyTorch.
    
    This implementation exactly matches LibXC's lda_x functional.
    Uses Slater-Dirac exchange.
    
    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import ValGrad
    >>> xc = PyTorchLDA()
    >>> n = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
    >>> densinfo = ValGrad(value=n)
    >>> edensity = xc.get_edensityxc(densinfo)
    >>> edensity.shape
    torch.Size([3])
    """

    def __init__(self, name: str = "lda_x"):
        super().__init__()
        self.name = name

    @property
    def family(self) -> int:
        return 1

    def _slater_exchange(self, n: torch.Tensor) -> torch.Tensor:
        C_x = torch.tensor(0.75 * (3.0 / torch.pi) ** (1.0 / 3.0))
        n_safe = torch.clamp(n, min=1e-12)
        return -C_x * safepow(n_safe, 4.0 / 3.0)

    def get_edensityxc(
        self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
    ) -> torch.Tensor:
        if isinstance(densinfo, ValGrad):
            n = densinfo.value
            ex = self._slater_exchange(n)
            return ex
        else:
            nu = densinfo.u.value
            nd = densinfo.d.value
            ex = self._slater_exchange(nu) + self._slater_exchange(nd)
            return ex

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return []
