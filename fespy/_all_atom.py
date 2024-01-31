"""
_all_atom.py

Submodule that computes all heavy atom level information of the protein structure obtained by FES.
"""
import mdtraj
import numpy as np
from scipy.constants import epsilon_0, e, N_A
from ._prep import *
from itertools import combinations
import typing

__all__ = ['AllAtom']


class AllAtom:
    """
    All atom object which determines full atom library for any
    given single domain protein.
    """
    def __init__(self, traj, pH, temp) -> None:
        """
        Wrangle atom level information on the single domain protein.
        Properties include van der waals, hydrogen bonding, charge,
        and bonded pairings along with the atom information of each
        pair.

        Parameters
        ----------
        traj: mdtraj.Trajectory
            Trajectory object to extract all this data.
        pH: float
            pH of the protein.
        """
        # general position and topology data
        self.xyz = traj.xyz[0]
        self.seq = ''.join(traj.topology.to_fasta())
        self.topology: mdtraj.Topology = traj.topology

        # van der Waal radii in nanometer
        self.rw = {
            'C': 0.165, 'O': 0.150,
            'N': 0.147, 'S': 0.175,
        }

        # bonded data
        (self.bond,
         self.bond_key,
         self.bond_type) = self._bonded()

        # non-bonded (van der Waal) data
        (self.r,
         self.r_min,
         self.r_cut,
         self.seqdist,
         self.non_bond,
         self.non_bond_key,
         self.non_bond_type) = self._non_bonded(traj)

        # electrostatic data
        self.charge_dict = determine_charges(
            pH=pH,
            first_res=self.seq[0],
            last_res=self.seq[-1],
        )
        self.epsilon = compute_dielectric(temp=temp) * epsilon_0
        self.lam_d = compute_debye_length(
            ionic_strength=compute_ionic_strength(conc=0.05, pH=pH),
            temp=temp,
        )
        (self.is_coulomb,
         self.q1q2,
         self.coulomb_potential) = self._electrostatic()

        # hydrogen bond data
        (self.is_hbond,
         self.hbond_theta) = self._hbonds(traj)

    def _check_pairs(self, i, j) -> bool:
        """
        Check if atom pairing is not bonded nor from same residue.

        Parameters
        ----------
        i: int
            Atom index i
        j: int
            Atom index j

        Returns
        -------
        rtype: bool
            If True atom pairing will be used, else abandoned.
        """
        # check if in same residue
        if self.topology.atom(i).residue.index == self.topology.atom(j).residue.index:
            return False

        # check if any one is a hydrogen
        if (self.topology.atom(i).name.upper()[0] == 'H') or (
                self.topology.atom(j).name.upper()[0] == 'H'):
            return False

        # Determine if bonded
        if np.any(np.isin(self.bond, (i, j)).all(axis=1)):
            return False

        # if all conditions met return true
        return True

    def _bonded(self) -> list[np.ndarray]:
        """
        Find all the non-hydrogen bonds in the protein and the
        atom types involved in the pairings.

        Returns
        -------
        rtype: list[np.ndarray]
            Return the atom bond index and atom bond types.
        """
        bond_data = np.array([
            [a.index,
             b.index,
             f'{a.name.upper()}-{a.residue.name.upper()}-{a.residue.index}',
             f'{b.name.upper()}-{b.residue.name.upper()}-{b.residue.index}',
             ''.join([a.name.upper()[0], b.name.upper()[0]])
             ] for a, b in self.topology.bonds
        ])
        return [bond_data[:, :2].astype(int), bond_data[:, 2:], bond_data[:, -1]]

    def _non_bonded(self, traj) -> list[np.ndarray]:
        """
        Find all the non-hydrogen bonds in the protein and the
        atom types involved in the pairings.

        Returns
        -------
        rtype: list[np.ndarray]
            Return the atom bond index and atom bond types.
        """
        # extract all the non-bonded data
        non_bond_data = np.array([
            (i,
             j,
             '{}-{}-{}'.format(
                 self.topology.atom(i).name.upper(),
                 self.topology.atom(i).residue.name.upper(),
                 self.topology.atom(i).residue.index
             ),
             '{}-{}-{}'.format(
                 self.topology.atom(j).name.upper(),
                 self.topology.atom(j).residue.name.upper(),
                 self.topology.atom(j).residue.index
             ),
             ''.join([self.topology.atom(j).name.upper()[0], self.topology.atom(j).name.upper()[0]]),
             np.abs(self.topology.atom(i).residue.index - self.topology.atom(j).residue.index))
            for i, j in combinations(np.arange(self.topology.n_atoms), 2) if self._check_pairs(i, j)
        ])

        # split into 1D array attributes
        pair_idx = non_bond_data[:, :2].astype(int)
        pair_key = non_bond_data[:, 2:-2]
        pair_type = non_bond_data[:, -2]
        seqdist = non_bond_data[:, -1].astype(int)
        r_min = np.array([self.rw[a[0]] + self.rw[b[0]] for a, b in pair_type])
        r_cut = 2.5 * r_min
        r = mdtraj.compute_distances(traj, atom_pairs=pair_idx)[0]

        return [r, r_min, r_cut, seqdist, pair_idx, pair_key, pair_type]

    def _electrostatic(self, shift=True) -> list[np.ndarray]:
        """
        Determine the electrostatics of the protein. The atom pairing,
        atom pairing type, point charge, and distance between each pairing
        will be determined for the protein.

        Parameters
        ----------
        shift: bool
            Apply shift function to taper electrostatic effects to 0 outside
            of Debye length. Default is True.

        Returns
        -------
        rtype: list[np.ndarray]
            Returns which pairings are involved in electrostatics, the charge, and potential
            in (kJ/mol).
        """
        # determine the point charge of each atom type
        charged_resid = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS']

        # selection algebra for finding the subset of atoms in the protein
        charge_atom_idx = self.topology.select(
            f'((resname {" ".join(charged_resid)}) and (name {" ".join(list(self.charge_dict.keys())[:-2])})) \
            or (name N and resid 0) or (name OXT and resid {self.topology.n_residues - 1})'
        )
        charge_pairs = np.array([
            (i, j) for i, j in combinations(charge_atom_idx, 2)
            if self._check_pairs(i, j)
        ])
        is_coulomb = np.array([
            np.any(np.isin(charge_pairs, _).all(axis=1))
            for _ in self.non_bond
        ])

        # get index of charged atom pairings
        q1q2 = np.zeros(self.r.shape)
        _ = np.arange(0, is_coulomb.shape[0])
        coulomb_idx = np.array([
            _[np.isin(self.non_bond, arr).all(axis=1)]
            for arr in charge_pairs
        ]).reshape(-1).astype(int)

        # add charge to the charged pairings
        q1q2[coulomb_idx] = np.array([
            (self.charge_dict[self.topology.atom(i).name] * e) *
            (self.charge_dict[self.topology.atom(j).name] * e)
            for i, j in charge_pairs
        ])

        # apply shift function
        if shift:
            shift_c = (1 - ((self.r / self.lam_d) ** 2)) ** 2
            shift_c[self.r > self.lam_d] = 0
            is_coulomb[self.r > self.lam_d] = False
        else:
            shift_c = np.ones(self.r.shape)

        # calculate the molar coulomb constant in terms of (kJ * m) / (C^2 * mol)
        ke = (1e-3 * N_A) / (4 * np.pi * self.epsilon)

        # compute electrostatic potential and apply shift function
        potential = ke * (q1q2 / (self.r * 1e-9)) * np.exp(-self.r / self.lam_d)
        coulomb_potential = shift_c * potential

        # return electrostatic data
        return [is_coulomb, q1q2, coulomb_potential]

    def _hbonds(self, traj) -> list[np.ndarray]:
        """
        Find all the hydrogen bonds in the protein along with the
        angle each bond is producing.

        Parameters
        ----------
        traj: mdtraj.Trajectory
            Trajectory object for determining the atoms involved in
            hydrogen bonding.

        Returns
        -------
        rtype: list[np.ndarray]
            Returns the angles, distances, and pairings each
            hydrogen bond is involved in.
        """
        donor, hdrgn, acptr = np.array([
            (i, j, k) for i, j, k in zip(*mdtraj.wernet_nilsson(traj=traj)[0].T)
            if self._check_pairs(i, k) and ~np.any(
                np.isin(self.non_bond[self.is_coulomb], (i, k)).all(axis=1)
            )
        ]).T
        hbond_pairs = np.array([donor, acptr]).T
        is_hbond = np.array([
            np.any(np.isin(hbond_pairs, _).all(axis=1))
            for _ in self.non_bond
        ])

        # get index of atom_pairings
        _ = np.arange(is_hbond.shape[0])
        theta_idx = np.array([
            _[np.isin(self.non_bond, arr).all(axis=1)]
            for arr in hbond_pairs
        ]).reshape(-1)

        # determine theta, -1 means non-hbond pairing
        theta = -1 * np.ones(self.r.shape)
        a = self.xyz[donor] - self.xyz[hdrgn]
        b = self.xyz[acptr] - self.xyz[hdrgn]
        theta[theta_idx] = np.arccos([
            ai.dot(bi) / np.sqrt(ai.dot(ai)) / np.sqrt(bi.dot(bi)) * (180 / np.pi)
            for ai, bi in zip(a, b)
        ])

        # return all hbond data
        return [is_hbond, theta]
