"""
_prep.py

Submodule with structure and charge determination.
"""
import pdbfixer
from openmm import unit
import mdtraj
import numpy as np

__all__ = ['prep_structure', 'determine_charges', 'compute_dielectric']


def prep_structure(pdb_file, pH=7.0) -> mdtraj.Trajectory:
    """
    Fix and prepares the structure via pdbfixer and preps
    the structure for analysis of the FES module.

    Parameters
    ----------
    pdb_file: str
        File path of databank file.
    pH: float
        pH of the protein.

    Returns
    -------
    rtype: mdtraj.Trajectory
        Return the mdtraj trajectory of the pdb file after
        structure is fixed.
    """
    # make fixer object and find missing residues that are only
    # in the middle of the chain.
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in keys:
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    # replace non-natural residues with natural occurring ones
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)

    # add missing atoms and hydrogen based off pH
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    # pass resultant structures to mdtraj
    return mdtraj.Trajectory(
        xyz=np.array(fixer.positions.value_in_unit(unit.nanometer)),
        topology=mdtraj.Topology.from_openmm(fixer.topology)
    )


def determine_charges(pH) -> dict:
    """
    Determine the partial charges of all charged
    residues in the protein.

    Parameters
    ----------
    pH: float
        pH of the protein in its given environment.

    Returns
    -------
    rtype: dict
        Float dictionary with charges for each atom type.
    """
    pKa = {
        'OD1': 3.65, 'OD2': 3.65,  # ASP
        'OE1': 4.25, 'OE2': 4.25,  # GLU
        'NZ': 10.53,  # LYS
        'NE': 12.48, 'NH1': 12.48, 'NH2': 12.48,  # ARG
        'ND1': 6.00, 'NE2': 6.00,  # HIS
        'OXT': 3.60,  # C-TERM
        'N': 8.00  # N-TERM
    }
    pKa_div = {
        'OD1': 2, 'OD2': 2,  # ASP
        'OE1': 2, 'OE2': 2,  # GLU
        'NZ': 1,  # LYS
        'NE': 3, 'NH1': 3, 'NH2': 3,  # ARG
        'ND1': 2, 'NE2': 2,  # HIS
        'OXT': 1,  # C-TERM
        'N': 1  # N-TERM
    }
    fN = lambda pk: 1 / (1 + 10 ** (pH - pk)) if np.abs(1 / (1 + 10 ** (pH - pk))) > 0.05 else 0
    fO = lambda pk: -1 / (1 + 10 ** -(pH - pk)) if np.abs(-1 / (1 + 10 ** -(pH - pk))) > 0.05 else 0

    # determine the point charge of each atom type
    return {
        _: fN(pKa[_]) / pKa_div[_] if _[0] == 'N' else fO(pKa[_]) / pKa_div[_]
        for _ in pKa.keys()
    }


def compute_dielectric(temp) -> float:
    """
    Determine the relative permittivity of protein
    in response to temperature. It should be noted
    that the mode
    Parameters
    ----------
    temp: float
        Temperature of the folding experiment in Kelvin.

    Returns
    -------
    rtype: float
        Returns the relative permittivity
    """
    fit = np.array([
        1.14427144e-08, -1.59366817e-05,  8.98556111e-03,
        -2.68039610e+00, 4.10734135e+02
    ])

    return np.array([
        a * (temp ** i) for i, a in enumerate(fit[::-1])
    ]).sum()
