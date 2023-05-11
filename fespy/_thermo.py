import numpy as np
from scipy.constants import gas_constant

__all__ = ['gen_enthalpy', 'gen_entropy', 'gen_free_energy']

# setting R to kJ/(mol * K)
R = gas_constant * 1e-3


def gen_enthalpy(nat, nres, DHres, kDH) -> np.ndarray:
    """
    Determine the 1D enthalpy function of the single domain protein.
    Assumed to decay like a markov chain.

    Parameters
    ----------
    nat: np.ndarray
        The nativeness reaction coordinate.
    nres: int
        Number of residues in the protein.
    DHres: float
        Average enthalpic contribution of each amino acid.
    kDH: float
        The coefficient that determines the decay of the enthalpy function.
        Essentially 'the rate of making intramolecular interactions'.

    Returns
    -------
    rtype: np.ndarray
        Returns the enthalpy function as a function of nativeness.
    """
    x = np.exp(-kDH)
    DH = nres * DHres
    return DH * ((1 - x**(1 - nat)) / (1 - x))


def gen_entropy(nat, nres, DSres=16.5e-3) -> np.ndarray:
    """
    Determine the 1D entropy function of the single domain protein.
    Assumes all residues contribute equally.

    Parameters
    ----------
    nat: np.ndarray
        The nativeness reaction coordinate.
    nres: int
        Number of residues in the protein.
    DSres: float
        The entropic penalty of fixing a residue to its native conformation.
        Default set to 16.5 J/(mol * K)

    Returns
    -------
    rtype: np.ndarray
        Returns the configurational entropy of the protein as a function of
        nativeness.
    """
    DS_mix = -R * nres * np.nan_to_num(
        (nat * np.log(nat)) + ((1 - nat) * np.log(1 - nat)),
        nan=0
    )
    DS_dif = nres * DSres * (1 - nat)

    return DS_dif + DS_mix


def gen_free_energy(nat, nres, DHres, kDH, temp=298, DSres=16.5e-3) -> np.ndarray:
    """
    Determine the 1D free energy function of the single domain protein.
    Assumes all residues contribute equally in enthalpy and entropy.

    Parameters
    ----------
    nat: np.ndarray
        The nativeness reaction coordinate.
    nres: int
        Number of residues in the protein.
    DHres: float
        Average enthalpic contribution of each amino acid.
    kDH: float
        The coefficient that determines the decay of the enthalpy function.
        Essentially the rate of 'making intramolecular interactions'.
    temp: float
        Temperature of the folding experiment in Kelvin. Default set to
        298K.
    DSres: float
        The entropic penalty of fixing a residue to its native conformation.
        Default set to 16.5 J/(mol * K)

    Returns
    -------
    rtype: np.ndarray
        Returns the free energy surface of the protein as a function of
        nativeness.
    """
    return gen_enthalpy(nat, nres, DHres, kDH) - (temp * gen_entropy(nat, nres, DSres))
