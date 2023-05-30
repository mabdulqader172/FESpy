import mdtraj
import numpy as np
from itertools import combinations

__all__ = ['CAlpha']


class CAlpha:
    def __init__(self, traj) -> None:
        """
        Wrangle alpha carbon level information on the single domain protein.
        Properties include ca-ca distances, DSSP sequence, and the DSSP
        pairings of each pair.

        Parameters
        ----------
        traj: mdtraj.Trajectory
            Trajectory object to extract all this data.
        """
        # Distances (nm) and residue pairings
        d, self.residue_pair = mdtraj.compute_contacts(
            traj=traj, contacts=list(combinations(np.arange(traj.n_residues), 2)), scheme='ca')
        self.distances = d[0]

        # DSSP and dssp pairings by residue
        self.dssp = mdtraj.compute_dssp(traj=traj, simplified=True)[0]
        self.dssp_pair = np.array([
            (self.dssp[i], self.dssp[j]) for i, j in self.residue_pair
        ])

        # Sequence distance for topology measurements
        self.seqdist = np.abs(self.residue_pair[:, 0] - self.residue_pair[:, 1]).astype(int)
