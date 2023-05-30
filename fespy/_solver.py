import numpy as np
from scipy.constants import gas_constant
from scipy.optimize import least_squares
from ._thermo import *
import typing

__all__ = ['find_DHres_kDH', ]

# setting R to kJ/(mol * K)
R = gas_constant * 1e-3


def find_DHres_kDH(kfexp, kuexp, nres, temp=298, k0=1e7, DSres=16.5e-3) -> typing.Tuple[float, float]:
    """
    Root finder which determines the enthalpy per residue contribution (DHres)
    and curvature (kappaDH) of the protein.

    Parameters
    ----------
    kfexp: float
        Folding rate of the protein domain.
    kuexp: float
        Unfolding rate of the protein domain.
    nres: int
        Number of residues in the protein domain.
    temp: float
        Temperature of the folding experiment in Kelvin. Default set to
        298K.
    k0: float
        Pre-exponential for determining (un)folding rates via Kramer's
        approximation. Default set to 1e7 per sec.
    DSres: float
        The entropic penalty of fixing a residue to its native conformation.
        Default set to 16.5 J/(mol * K)

    Returns
    -------
    rtype: typing.Tuple[float, float]
        Returns the enthalpy per residue contribution (DHres) and curvature
        (kappaDH) of the protein.
    """
    nat = np.linspace(0, 1, 101)
    lbound = int(0.1 * nat.size)
    hbound = int(0.3 * nat.size)

    # function for solver
    def _(x) -> float:
        # determine free energy profile and find DGts_u and DGts_n
        DG = gen_free_energy(nat, nres, *x, temp, DSres)
        barrier_idx = DG[hbound:-lbound].argmax() + hbound
        DGts_u = DG[barrier_idx] - DG[:barrier_idx].min()
        DGts_n = DG[barrier_idx] - DG[barrier_idx:].min()

        # determine folding kinetics
        kf = k0 * np.exp(-DGts_u/(R * temp))
        ku = k0 * np.exp(-DGts_n/(R * temp))

        # compute stability penalty
        p = np.log((DGts_n - DGts_u)/(-R * temp * np.log(kuexp / kfexp))) ** 2

        # solve objective function
        return (np.log(kf / kfexp) ** 2) + (np.log(ku / kuexp) ** 2) + p

    # solve for DHres and kDH
    DHres, kDH = least_squares(
        fun=_, x0=(5.5, 3), bounds=(0, np.inf)).x
    return DHres, kDH
