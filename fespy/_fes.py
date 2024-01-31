"""
_fes.py

Main OOP definition of the FES object.
"""
import mdtraj
import typing
import numpy as np
from scipy.constants import gas_constant
from ._thermo import *
from ._solver import *
from ._all_atom import *
from ._calpha import *
from ._prep import prep_structure

__all__ = ['FES']

# setting R to kJ/(mol * K)
R = gas_constant * 1e-3


class FES:
    """
    FES: Free Energy Surface

    OOP interface for calculating the 1D projection of an energy surface
    for single domain proteins. The model assumes kinetics can be
    determined via diffusion on the energy surface.
    """
    def __init__(self, kfexp, kuexp, pdb, pH=7.4, temp=298, k0=1e7, lnat=101):
        """
        Parses structure and sequence data from the Protein Data Bank
        file passed along with the experimental folding kinetics
        obtained from research.

        Parameters
        ----------
        kfexp: float
            Folding rate of the protein domain.
        kuexp: float
            Unfolding rate of the protein domain.
        pdb: str
            File path of protein databank file.
        pH: float
            pH of the protein.
        temp: float
            Temperature of the folding experiment in Kelvin. Default set to
            298K.
        k0: float
            Pre-exponential for determining (un)folding rates via Kramer's
            approximation. Default set to 1e7 per sec.
        lnat: int
            The size of the mesh for the nativeness reaction coordinate. Default
            set to 101 points.
        """
        self.traj: mdtraj.Trajectory = prep_structure(pdb_file=pdb, pH=pH)
        self.n_residues = self.traj.n_residues
        self.n_atoms = self.traj.n_atoms
        self.pH = pH
        self.temp = temp
        self.id = ''.join((str(pdb).split("/")[-1]).split(".")[:-1])

        # General Kinetics
        self.kf = kfexp
        self.ku = kuexp
        self.k0 = k0
        self.pf = 1 / (1 + (self.ku / self.kf))
        self.pu = 1 - self.pf

        # General Thermodynamics
        self.nat = np.linspace(0, 1, lnat)
        self.DHres, self.kDH = find_DHres_kDH(
            self.kf, self.ku, self.n_residues, self.temp, self.k0)
        self.DS = gen_entropy(self.nat, self.n_residues)
        self.DH = gen_enthalpy(
            self.nat, self.n_residues, self.DHres, self.kDH)
        self.DG = self.DH - (self.temp * self.DS)
        self.stab = -R * self.temp * np.log(self.ku / self.kf)

        # All atom description
        self.atom = AllAtom(traj=self.traj, pH=self.pH, temp=self.temp)

        # Alpha Carbon description
        self.c_alpha = CAlpha(traj=self.traj)
        (self.co,
         self.aco,
         self.tcd,
         self.lro) = self._get_topology()

    def _get_topology(self) -> typing.Tuple[float, float, float, float]:
        """
        Returns the topology parameters derived by previous research from following
        groups: Baker Lab (CO, ACO), Gromiha Lab (LRO), Zhou Lab (TCD). Each follow
        the criterion of the original papers (ACO, CO, TCD: 2 residues away and
        within 6 Angstroms distance; LRO: 12 residues away and 8 angstroms away).

        Returns
        -------
        rtype: typing.Tuple[float, float, float, float]
            Contact order (CO), absolute contact order (ACO), total contact distance
            (TCD), and long range order (LRO) of the protein.
        """
        # get the non-bonded distances and conditions
        cond = (self.atom.r < 0.6) & (self.atom.seqdist > 2)

        # compute CO, ACO, and TCD
        co = self.atom.seqdist[cond].sum() / (self.n_residues * self.atom.seqdist[cond].size)
        aco = self.n_residues * co
        tcd = self.atom.seqdist[cond].sum() / (self.n_residues ** 2)

        # condition for LRO
        cond = (self.c_alpha.r < 0.8) & (self.c_alpha.seqdist >= 12)
        lro = self.c_alpha.seqdist[cond].size / self.n_residues

        return co, aco, tcd, lro
