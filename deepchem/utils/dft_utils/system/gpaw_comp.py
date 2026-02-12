from ase import Atoms
from gpaw import GPAW, PW
import numpy as np

# same cubic cell
atoms = Atoms(
    symbols='H',
    positions=[[0,0,0]],
    cell=np.eye(3),
    pbc=True
)

calc = GPAW(
    mode=PW(544),        # ≈ 20 Ha
    kpts=[2,2,2],        # your k-vector
    txt=None
)

atoms.calc = calc
calc.initialize(atoms)

pd = calc.wfs.pd   # plane-wave descriptor

G = pd.G_Qv        # GPAW G vectors
k = pd.K_qv[0]     # k vector

print("GPAW G vectors:")
print(G[:10])

print("GPAW |k+G|^2:")
q = G + k
print(np.sum(q*q, axis=1)[:10])
