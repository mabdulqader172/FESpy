"""
_fes.py

TODO: make FES oop object; requires:
    - all_atom: atom-level sub-object which includes the following
        1. van der Waal atom pairings
        2. Hydrogen Bonding Pairings
        3. Electrostatic Pairings (based on pH)
        4. Bonded Pairings
        ** All pairings within same residue are not included
        Included: all r_min values for all atomic pairings.
    - alpha_carbon: alpha-carbon level description which includes
        1. ca_ca pairings
        2. DSSP pairings
        ** All pairings intended to include DSSP pairings
    # - General Thermodynamics
    #     1. Enthalpy: DHres, DH(n), and kappaDH
    #     2. Entropy: DSres, DS(n)
    #     3. Free Energy: DG(n), temp
    # - General Kinetics
    #     1. Folding Kinetics (kf, ku) and (pf, pu)
    #     2. Stability
    #     3. PreExponential (k0)
    - Structural Parameters
        1. Topology: CO, ACO
        2. Global Charge given pH
        3. DSSP
        4. Sequence Length
"""
import mdtraj
import numpy as np
from scipy.constants import gas_constant
from ._thermo import *
from ._solver import *

__all__ = ['FES']

# setting R to kJ/(mol * K)
R = gas_constant * 1e-3


class FES:
    """
    FES: Free Energy Surface

    OOP interface for calculating the 1D projection of an energy surface
    for single domain proteins. The model assumes kinetics can be
    determined via diffusion on the energy surface.

    Attributes
    ----------
    """
    def __init__(self, kfexp, kuexp, pdb, pH=7.4, temp=298, k0=1e7, lnat=101) -> None:
        """
        Parses structure and sequence data from the Protein Data Bank
        file passed along with the experimental folding kinetics
        obtained from research.
        """
        self.t: mdtraj.Trajectory = mdtraj.load_pdb(pdb, frame=0)
        self.n_residues = self.t.n_residues
        self.n_atoms = self.t.n_atoms

        # General Kinetics
        self.kf = kfexp
        self.ku = kuexp
        self.k0 = k0
        self.pH = pH
        self.temp = temp
        self.stab = -R * self.temp * np.log(self.ku/self.kf)
        self.pf = 1/(1 + (self.ku/self.kf))
        self.pu = 1 - self.pf

        # General Thermodynamics
        self.nat = np.linspace(0, 1, lnat)
        self.DHres, self.kDH = find_DHres_kDH(
            self.kf, self.ku, self.n_residues, self.temp, self.k0)
        self.DS = gen_entropy(self.nat, self.n_residues)
        self.DH = gen_enthalpy(
            self.nat, self.n_residues, self.DHres, self.kDH)
        self.DG = self.DH - (self.temp * self.DS)

