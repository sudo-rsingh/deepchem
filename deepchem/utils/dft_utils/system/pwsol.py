
from typing import Union, Tuple, List, Any
import torch
import math


class AtomsTorch:
    """Torch version of Atoms object that holds all system and cell parameters."""

    def __init__(self, atom, pos, a, ecut, Z, s, f, device=None, rdtype=torch.float64):
        self.device = device or torch.device("cpu")
        self.rdtype = rdtype
        self.cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64

        self.atom = atom
        self.pos = torch.as_tensor(pos, dtype=self.rdtype, device=self.device)
        self.a = torch.as_tensor(a, dtype=self.rdtype, device=self.device)
        self.ecut = torch.as_tensor(ecut, dtype=self.rdtype, device=self.device)
        self.Z = torch.as_tensor(Z, dtype=self.rdtype, device=self.device)
        self.s = torch.as_tensor(s, dtype=torch.long, device=self.device)
        self.f = torch.as_tensor(f, dtype=self.rdtype, device=self.device)

        self.initialize()

    def initialize(self):
        M, N = self._get_index_matrices()
        self._set_cell(M)
        self._set_G(N)

    def _get_index_matrices(self):
        s0, s1, s2 = [int(x) for x in self.s]
        n_tot = s0 * s1 * s2

        ms = torch.arange(n_tot, dtype=self.rdtype, device=self.device)

        m1 = torch.floor(ms / (s2 * s1)) % s0
        m2 = torch.floor(ms / s2) % s1
        m3 = ms % s2
        M = torch.stack((m1, m2, m3), dim=1)

        n1 = m1 - (m1 > s0 / 2) * s0
        n2 = m2 - (m2 > s1 / 2) * s1
        n3 = m3 - (m3 > s2 / 2) * s2
        N = torch.stack((n1, n2, n3), dim=1)

        return M.to(self.rdtype), N.to(self.rdtype)

    def _set_cell(self, M):
        self.Natoms = len(self.atom)
        self.Nstate = len(self.f)

        self.pos = torch.atleast_2d(self.pos)
        self.Z = self.Z.to(self.rdtype)

        R = self.a * torch.eye(3, dtype=self.rdtype, device=self.device)
        self.R = R
        self.Omega = torch.abs(torch.det(R))

        Sinv = torch.diag(1.0 / self.s.to(self.rdtype))
        self.r = M @ Sinv @ R.T

    def _set_G(self, N):
        Rinv = torch.linalg.inv(self.R)
        G = 2 * math.pi * (N @ Rinv)
        self.G = G

        G2 = torch.linalg.norm(G, dim=1) ** 2
        self.G2 = G2

        active_mask = (2 * self.ecut >= G2)
        self.active = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
        self.G2c = G2[self.active]

        phase = torch.exp(-1j * (G @ self.pos.T).to(self.cdtype))
        self.Sf = torch.sum(phase, dim=1)


class PWBasis:
    def __init__(self, atoms, device=None, dtype=torch.complex128):
        self.atoms = atoms
        self.device = device or torch.device("cpu")
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32

        self.Omega = torch.as_tensor(atoms.Omega, dtype=self.rdtype, device=self.device)
        self.G2 = torch.as_tensor(atoms.G2, dtype=self.rdtype, device=self.device)
        self.G2c = torch.as_tensor(atoms.G2c, dtype=self.rdtype, device=self.device)
        self.s = tuple(int(x) for x in atoms.s)
        self.n = int(torch.prod(torch.tensor(self.s)))
        self.active = torch.as_tensor(atoms.active, dtype=torch.long, device=self.device)
        self.Nstate = getattr(atoms, "Nstate", None)

    def O(self, W): return O_torch(self, W)
    def L(self, W): return L_torch(self, W)
    def Linv(self, W): return Linv_torch(self, W)
    def I(self, W): return I_torch(self, W)
    def J(self, W): return J_torch(self, W)
    def Idag(self, W): return Idag_torch(self, W)
    def Jdag(self, W): return Jdag_torch(self, W)


def _to_complex(W, basis):
    return torch.as_tensor(W, dtype=basis.cdtype, device=basis.device)


def O_torch(basis, W):
    W = _to_complex(W, basis)
    return basis.Omega * W


def L_torch(basis, W):
    W = _to_complex(W, basis)
    G2 = basis.G2c[:, None] if W.shape[0] == basis.G2c.shape[0] else basis.G2[:, None]
    return -basis.Omega * G2 * W


def Linv_torch(basis, W):
    W = _to_complex(W, basis)
    out = torch.zeros_like(W)
    G2 = basis.G2[:, None] if W.ndim > 1 else basis.G2
    denom = G2 * (-basis.Omega)
    mask = G2 != 0
    out[mask] = W[mask] / denom[mask]
    out[0] = 0
    return out


def I_torch(basis, W):
    W = _to_complex(W, basis)
    n = basis.n

    if W.shape[0] != basis.G2.shape[0]:
        shape = (n,) if W.ndim == 1 else (n, basis.Nstate)
        Wfft = torch.zeros(shape, dtype=basis.cdtype, device=basis.device)
        Wfft[basis.active] = W
    else:
        Wfft = W

    if W.ndim == 1:
        return torch.fft.ifftn(Wfft.reshape(basis.s)).reshape(-1) * n
    else:
        return torch.fft.ifftn(Wfft.reshape((*basis.s, basis.Nstate)), dim=(0,1,2)).reshape(n, basis.Nstate) * n


def J_torch(basis, W):
    W = _to_complex(W, basis)
    n = basis.n
    if W.ndim == 1:
        return torch.fft.fftn(W.reshape(basis.s)).reshape(-1) / n
    else:
        return torch.fft.fftn(W.reshape((*basis.s, basis.Nstate)), dim=(0,1,2)).reshape(n, basis.Nstate) / n


def Idag_torch(basis, W):
    return J_torch(basis, W)[basis.active] * basis.n


def Jdag_torch(basis, W):
    return I_torch(basis, W) / basis.n


def coulomb(atoms, op):
    Vcoul = -4 * torch.pi * atoms.Z[0] / atoms.G2
    Vcoul = Vcoul.clone()
    Vcoul[0] = 0
    return op.J(Vcoul * atoms.Sf)


def get_Eewald(atoms: Any, gcut: float = 2.0, gamma: float = 1e-8, dtype=torch.float64):
    Z = torch.as_tensor(atoms.Z, dtype=dtype)
    R = torch.as_tensor(atoms.R, dtype=dtype)
    Omega = float(atoms.Omega)
    pos = torch.as_tensor(atoms.pos, dtype=dtype)
    N = int(atoms.Natoms)

    gexp = -math.log(gamma)
    nu = torch.tensor(0.5 * math.sqrt(gcut**2 / gexp), dtype=dtype)

    Eewald = -nu / torch.sqrt(torch.tensor(torch.pi, dtype=dtype)) * torch.sum(Z**2)
    Eewald += (-math.pi * torch.sum(Z).item() ** 2) / (2.0 * (nu**2).item() * Omega)
    return Eewald


def get_n_total(atoms, op, Y):
    Yrs = op.I(Y)
    n = atoms.f * torch.real(Yrs.conj() * Yrs)
    return torch.sum(n, dim=1)


def sqrtm(A):
    evals, evecs = torch.linalg.eig(A)
    return evecs @ torch.diag(torch.sqrt(evals)) @ torch.linalg.inv(evecs)


def orth(op, W):
    U = sqrtm(W.conj().T @ op.O(W))
    return W @ torch.linalg.inv(U)


def H(op, W, phi, vxc, Vreciproc):
    Veff = Vreciproc + op.Jdag(op.O(op.J(vxc) + phi))
    return -0.5 * op.L(W) + op.Idag(Veff[:, None] * op.I(W))


def Q(inp, U):
    mu, V = torch.linalg.eig(U)
    mu = mu[:, None]
    denom = torch.sqrt(mu) @ torch.ones((1, len(mu)), dtype=mu.dtype)
    denom2 = denom + denom.conj().T
    return V @ ((V.conj().T @ inp @ V) / denom2) @ V.conj().T


def get_grad(atoms, op, W, phi, vxc, Vreciproc):
    F = torch.diag(atoms.f).to(op.cdtype)
    HW = H(op, W, phi, vxc, Vreciproc)
    WHW = W.conj().T @ HW
    OW = op.O(W)
    U = W.conj().T @ OW
    invU = torch.linalg.inv(U)
    U12 = sqrtm(invU)
    Ht = U12 @ WHW @ U12
    return (HW - (OW @ invU) @ WHW) @ (U12 @ F @ U12) + OW @ (U12 @ Q(Ht @ F - F @ Ht, U))


def get_phi(op, n):
    return -4 * torch.pi * op.Linv(op.O(op.J(n)))


def lda_x(n):
    f = -3 / 4 * (3 / (2 * torch.pi)) ** (2 / 3)
    rs = (3 / (4 * torch.pi * n)) ** (1 / 3)
    ex = f / rs
    vx = 4 / 3 * ex
    return ex, vx


def lda_c_chachiyo(n):
    a = -0.01554535
    b = 20.4562557
    rs = (3 / (4 * torch.pi * n)) ** (1 / 3)
    ec = a * torch.log(1 + b / rs + b / rs**2)
    vc = ec + a * b * (2 + rs) / (3 * (b + b * rs + rs**2))
    return ec, vc


def get_E(scf):
    return (
        get_Ekin(scf.atoms, scf.op, scf.Y)
        + get_Ecoul(scf.op, scf.n, scf.phi)
        + get_Exc(scf.op, scf.n, scf.exc)
        + get_Een(scf.n, scf.pot)
        + scf.Eewald
    )


def get_Ekin(atoms, op, W):
    F = torch.diag(atoms.f).to(op.cdtype)
    return torch.real(-0.5 * torch.trace(F @ W.conj().T @ op.L(W)))


def get_Ecoul(op, n, phi):
    n = n.to(op.cdtype)
    return torch.real(0.5 * n @ op.Jdag(op.O(phi)))


def get_Exc(op, n, exc):
    n = n.to(op.cdtype)
    return torch.real(n @ op.Jdag(op.O(op.J(exc))))


def get_Een(n, Vreciproc):
    n = n.to(Vreciproc.dtype)
    return torch.real(Vreciproc.conj().T @ n)


def scf_step(scf):
    scf.Y = orth(scf.op, scf.W)
    scf.n = get_n_total(scf.atoms, scf.op, scf.Y)
    scf.phi = get_phi(scf.op, scf.n)
    x, c = lda_x(scf.n), lda_c_chachiyo(scf.n)
    scf.exc = x[0] + c[0]
    scf.vxc = x[1] + c[1]
    return get_E(scf)


def sd(scf, Nit, etol=1e-6, beta=1e-5):
    Elist = []
    for i in range(Nit):
        E = scf_step(scf)
        Elist.append(E)
        print(f"Nit: {i+1}\tEtot: {E:.6f} Eh", end="\r")
        if i > 1 and abs(Elist[i-1] - Elist[i]) < etol:
            print("\nSCF converged.")
            return E
        g = get_grad(scf.atoms, scf.op, scf.W, scf.phi, scf.vxc, scf.pot)
        scf.W = scf.W - beta * g
    print("\nSCF not converged!")
    return E


def pseudo_uniform(size, seed=1234):
    U = torch.zeros(size, dtype=torch.complex128)
    mult, mod = 48271, (2**31) - 1
    x = (seed * mult + 1) % mod
    for i in range(size[0]):
        for j in range(size[1]):
            x = (x * mult + 1) % mod
            U[i, j] = x / mod
    return U


class SCF:
    def __init__(self, atoms):
        self.atoms = atoms
        self.op = PWBasis(atoms)
        self.pot = coulomb(self.atoms, self.op)
        self._init_W()

    def run(self, Nit=1001, etol=1e-6):
        self.Eewald = get_Eewald(self.atoms)
        return sd(self, Nit, etol)

    def _init_W(self, seed=1234):
        W = pseudo_uniform((len(self.atoms.G2c), self.atoms.Nstate), seed)
        self.W = orth(self.op, W)


if __name__ == "__main__":
    sol = AtomsTorch(["H"], [[0, 0, 0]], 16, 16, [1], [60, 60, 60], [1])
    print("Natoms:", sol.Natoms)
    print("Omega:", sol.Omega)
    print("G count:", sol.G.shape[0])
    print(sol.G2c.shape[0])

    import time
    start = time.perf_counter()
    SCF(sol).run()
    print(" {:.6f} seconds".format(time.perf_counter() - start))

