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
        self.dASA_p, self.dASA_ap = self._asa()

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

    def _asa(self) -> typing.Tuple[float, float]:
        """
        Calculate the change in solvent accessible surface area in angstrom
        square.

        Returns
        -------
        rtype: typing.Tuple[float, float]
            Return solvent accessible surface area for polar and apolar
            components, respectively.
        """
        max_ap_lib = {
            'ALA': 86.7, 'GLN': 59.7, 'LEU': 154.1, 'SER': 40.0,
            'ARG': 96.9, 'GLU': 64.0, 'LYS': 125.2, 'THR': 89.7,
            'ASN': 37.2, 'GLY': 42.0, 'MET': 121.4, 'TRP': 197.9,
            'ASP': 36.6, 'HIS': 108.7, 'PHE': 191.2, 'TYR': 158.8,
            'CYS': 34.8, 'ILE': 150.1, 'PRO': 130.0, 'VAL': 133.6
        }
        max_pl_lib = {
            'ALA': 62.3, 'GLN': 165.3, 'LEU': 46.9, 'SER': 125.0,
            'ARG': 177.1, 'GLU': 160.0, 'LYS': 112.8, 'THR': 94.3,
            'ASN': 164.8, 'GLY': 62.0, 'MET': 102.6, 'TRP': 87.1,
            'ASP': 156.4, 'HIS': 115.3, 'PHE': 48.8, 'TYR': 104.2,
            'CYS': 143.2, 'ILE': 46.9, 'PRO': 49.0, 'VAL': 51.4
        }

        # get non_hydrogen structure
        non_hydrogen_structure = self.traj.atom_slice(
            self.traj.topology.select('(element != H) and (element != D)')
        )
        max_ap, max_pl = np.array([
            (max_ap_lib[res.name], max_pl_lib[res.name])
            for res in non_hydrogen_structure.topology.residues
        ]).sum(axis=0)

        # get the asa, with the index
        asa, idx = mdtraj.shrake_rupley(
            non_hydrogen_structure, mode='atom', n_sphere_points=960, get_mapping=True
        )

        # identify the apolar atoms
        is_apolar = np.array([
            non_hydrogen_structure.topology.atom(i).name.upper()[0] == 'C'
            for i in idx
        ]).astype(bool)

        # apply boolean to get p and ap areas
        asa_ap = (asa[0, is_apolar].sum()) * 100
        asa_pl = (asa[0].sum() - asa_ap) * 100

        return max(max_pl - asa_pl, 0), max(max_ap - asa_ap, 0)
