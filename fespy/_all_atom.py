import mdtraj
import numpy as np
from scipy.constants import epsilon_0, e, N_A
from ._prep import determine_charges, compute_dielectric
from itertools import combinations
import typing

__all__ = ['AllAtom']


class AllAtom:
    """
    All atom object which determines full atom library for any
    given single domain protein.

    Attributes
    ----------

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
        self.topology: mdtraj.Topology = traj.topology

        # van der Waal radii in nanometer
        self.rw = {
            'C': 0.165, 'O': 0.150,
            'N': 0.147, 'S': 0.175,
        }

        # bonded data
        (self.bond,
         self.bond_type) = self._bonds()

        # hydrogen bond data
        (self.hbond,
         self.hbond_type,
         self.hbond_dist,
         self.hbond_theta,
         self.hbond_rmin,
         self.hbond_seqdist) = self._hbonds(traj)

        # van der Waal data
        (self.vdw,
         self.vdw_type,
         self.vdw_dist,
         self.vdw_rmin,
         self.vdw_seqdist) = self._van_der_waal(traj)

        # electrostatic data
        self.charge_dict = determine_charges(pH=pH)
        self.epsilon_r = compute_dielectric(temp=temp)
        (self.coulomb,
         self.coulomb_type,
         self.coulomb_potential,
         self.coulomb_charge,
         self.coulomb_dist,
         self.coulomb_rmin,
         self.coulomb_seqdist) = self._electrostatic(traj)

        # key all pairings
        self.non_bonded, self.non_bonded_key = self.determine_keys()

    def determine_keys(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Determine the key matrix for all atom pairings available in the protein.

        Returns
        -------
        rtype: typing.Tuple[np.ndarray, np.ndarray]
            Array of key typings for search algorithms.
        """
        pair_matrix = np.unique(
            np.vstack((self.vdw, self.coulomb, self.hbond)),
            axis=0
        )

        # make key matrix
        key_matrix = list()
        for i, j in pair_matrix:
            _key = [False, False, False]

            # determine the key combo
            if np.any((self.hbond == (i, j)).all(axis=1)):
                _key[1] = True
            else:
                _key[0] = True
            if np.any((self.coulomb == (i, j)).all(axis=1)):
                _key[2] = True

            # store key
            key_matrix.append(_key)

        return pair_matrix, np.array(key_matrix)

    def _bonds(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Find all the non-hydrogen bonds in the protein and the
        atom types involved in the pairings.

        Returns
        -------
        rtype: typing.Tuple[np.ndarray, np.ndarray]
            Return the atom bond index and atom bond types.
        """
        bond_data = np.array([
            [a.index, b.index, a.name.upper(), b.name.upper()]
            for a, b in self.topology.bonds
        ])
        # non_hydrogen_cond = (bond_data[:, -2] != 'H') & (bond_data[:, -1] != 'H')

        return bond_data[:, :2].astype(int), bond_data[:, 2:].astype(str)
        # return bond_data[non_hydrogen_cond, :2].astype(int), bond_data[non_hydrogen_cond, 2:].astype(str)

    def _hbonds(self, traj) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        rtype: typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Returns the angles, distances, and pairings each
            hydrogen bond is involved in.
        """
        # donor, hdrgn, acptr = mdtraj.baker_hubbard(traj).T
        donor, hdrgn, acptr = mdtraj.wernet_nilsson(traj=traj)[0].T

        # determine theta
        a = self.xyz[donor] - self.xyz[hdrgn]
        b = self.xyz[acptr] - self.xyz[hdrgn]
        hbond_theta = np.arccos([
            ai.dot(bi) / np.sqrt(ai.dot(ai)) / np.sqrt(bi.dot(bi))
            for ai, bi in zip(a, b)
        ])

        # determine atomic distances between each hbond
        hbond_dist = mdtraj.compute_distances(
            traj, atom_pairs=np.vstack((donor, acptr)).T)[0]

        # sort the donor and acceptor pairings by idx
        hbond = np.sort(np.vstack((donor, acptr)).T, axis=1)

        # get the hbond types
        hbond_type = np.array([
            (self.topology.atom(i).name.upper()[0],
             self.topology.atom(j).name.upper()[0])
            for i, j in hbond
        ])
        hbond_seqdist = np.array([
            np.abs(self.topology.atom(i).residue.index - self.topology.atom(j).residue.index)
            for i, j in hbond
        ])

        # get hbond sigma
        hbond_rmin = np.array([self.rw[a] + self.rw[b] for a, b in hbond_type])

        # return all hbond data
        return hbond, hbond_type, hbond_dist, hbond_theta, hbond_rmin, hbond_seqdist

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
        if np.any((self.bond == (i, j)).all(axis=1)):
            return False

        # Determine if hydrogen bonded
        if np.any((self.hbond == (i, j)).all(axis=1)):
            return False

        # if all conditions met return true
        return True

    def _van_der_waal(self, traj) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Determine all the atom pairings, atom types of each pairing,
        distance (in nanometer), and sigma (for lennard-jones
        calculations) for all van der Waal pairs in the protein.

        Parameters
        ----------
        traj: mdtraj.Trajectory
            Trajectory object for determining the atoms involved in
            van der Waal interactions.

        Returns
        -------
        rtype: typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Returns atom idx of each pair, atom types in pair, distance,
            and sigma for each van der Waals interaction.
        """
        # determine all the vdw pairings and atom types of each pairing
        vdw_data = np.array([
            (i, j,
             self.topology.atom(i).name.upper()[0], self.topology.atom(j).name.upper()[0],
             np.abs(self.topology.atom(i).residue.index - self.topology.atom(j).residue.index))
            for i, j in combinations(np.arange(self.topology.n_atoms), 2) if self._check_pairs(i, j)
        ])
        vdw = vdw_data[:, :2].astype(int)
        vdw_type = vdw_data[:, 2:-1]
        vdw_seqdist = vdw_data[:, -1].astype(int)

        # get distance parameters
        vdw_dist = mdtraj.compute_distances(traj, atom_pairs=vdw_data[:, :2])[0]
        vdw_rmin = np.array([self.rw[a] + self.rw[b] for a, b in vdw_type])

        return vdw, vdw_type, vdw_dist, vdw_rmin, vdw_seqdist

    def _electrostatic(self, traj) -> typing.Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Determine the electrostatics of the protein. The atom pairing,
        atom pairing type, point charge, and distance between each pairing
        will be determined for the protein.

        Parameters
        ----------
        traj: mdtraj.Trajectory
            Trajectory object for determining the atoms involved in
            van der Waal interactions.

        Returns
        -------
        rtype: typing.Tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray np.ndarray]
            Returns the coulomb pairings, atom typings, potential (kJ/mol), q1q1, and r (nm).
        """
        # determine the point charge of each atom type
        charged_resid = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS']

        # selection algebra for finding the subset of atoms in the protein
        atom_idx = self.topology.select(
            f'((resname {" ".join(charged_resid)}) and (name {" ".join(list(self.charge_dict.keys())[:-2])})) or \
            (name N and resid 0) or (name OXT and resid {self.topology.n_residues - 1})'
        )
        f = lambda i, j: (self.topology.atom(i).residue.index != self.topology.atom(j).residue.index) \
            and not (np.any((self.bond == (i, j)).all(axis=1)))

        # get all the atom pairings pairing types for charged atoms
        coulomb_data = np.array([
            (i, j,
             self.topology.atom(i).name.upper()[0], self.topology.atom(j).name.upper()[0],
             np.abs(self.topology.atom(i).residue.index - self.topology.atom(j).residue.index))
            for i, j in combinations(atom_idx, 2) if f(i, j)
        ])

        # get pairing and distance parameters
        coulomb_dist = mdtraj.compute_distances(traj, atom_pairs=coulomb_data[:, :2])[0]
        coulomb = coulomb_data[:, :2].astype(int)
        coulomb_type = coulomb_data[:, 2:-1]
        coulomb_seqdist = coulomb_data[:, -1].astype(int)

        # r_min
        coulomb_rmin = np.array([(self.rw[a] + self.rw[b]) for a, b in coulomb_type])

        # product charges
        coulomb_charge = np.array([
            (self.charge_dict[self.topology.atom(i).name] * e) * (self.charge_dict[self.topology.atom(j).name] * e)
            for i, j in coulomb
        ])

        # calculate the molar coulomb constant in terms of (J * m) / (C^2 * mol)
        ke = N_A / (4 * np.pi * epsilon_0 * self.epsilon_r)

        # coulomb potential calculation for each atom pairing.
        coulomb_potential = np.array([  # (J * m * C^2) / (C^2 * mol * m) --> (J / mol)
            ke * (q1q2 / d) for d, q1q2 in zip(coulomb_dist * 1e-9, coulomb_charge)
        ]) * 1e-3  # (J / mol) --> (kJ / mol)

        return coulomb, coulomb_type, coulomb_potential, coulomb_charge, coulomb_dist, coulomb_rmin, coulomb_seqdist
