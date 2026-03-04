import torch
from typing import Union, List
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils.data.datastruct import ValGrad, SpinParam
from deepchem.utils import safepow


class PyTorchLDA(BaseXC):
    """
    Local Density Approximation (LDA) XC functional.

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
        """Initialize the PyTorchLDA XC functional.

        Parameters
        ----------
        name: str (default "lda_x")
            The name of the LDA functional.
        """
        super().__init__()
        self.name = name
        self.functional = getattr(self, self.name)

    @property
    def family(self) -> int:
        """Returns the family identifier number of the XC.

        Returns
        -------
        int
            It returns 1 for LDA based on BaseXC.
        """
        return 1

    def lda_x(self, n: torch.Tensor) -> torch.Tensor:
        """Calculate the LDA exchange energy density.

        Parameters
        ----------
        n: torch.Tensor
            Electron density of the system.

        Returns
        -------
        torch.Tensor
            Exchange energy density.
        """
        C_x = torch.tensor(0.75 * ((3.0 / torch.pi)**(1.0 / 3.0)))
        n_safe = torch.clamp(n, min=1e-12)
        return -C_x * safepow(n_safe, 4.0 / 3.0)

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the xc energy density (energy per unit volume).

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            The density information.
            If the XC is unpolarized, then densinfo is ValGrad.
            If the XC is polarized, then densinfo is SpinParam[ValGrad].
            The ValGrad contains the value and gradient of the density.
            The SpinParam[ValGrad] contains the value and gradient of the density
            for each spin channel.

        Returns
        -------
        torch.Tensor
            The energy density of the XC.
        """
        if isinstance(densinfo, ValGrad):
            n = densinfo.value
            ex = self.functional(n)
            return ex
        else:
            nu = densinfo.u.value
            nd = densinfo.d.value
            ex = self.functional(nu) + self.functional(nd)
            return ex

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """List tensor names that affect the output of the method.

        Parameters
        ----------
        methodname: str
            Name of the method of the class.
        prefix: str, optional
            Prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.
        """
        return []
