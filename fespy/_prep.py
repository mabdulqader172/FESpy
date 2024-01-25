"""
_prep.py

Submodule with structure and charge determination.
"""
import pdbfixer
from openmm import unit
import mdtraj
import numpy as np
from scipy.constants import (
    Boltzmann as kB,
    N_A,
    elementary_charge as ele,
    epsilon_0
)

__all__ = [
    'prep_structure', 'determine_charges',
    'compute_dielectric', 'compute_ionic_strength',
    'compute_debye_length'
]


def prep_structure(pdb_file, pH=7.0) -> mdtraj.Trajectory:
    """
    Fix and prepares the structure via pdbfixer and preps
    the structure for analysis of the FES module.

    Parameters
    ----------
    pdb_file: str
        File path of protein databank file.
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


def determine_charges(pH, first_res, last_res) -> dict:
    """
    Determine the partial charges of all charged
    residues in the protein.

    Parameters
    ----------
    pH: float
        pH of the protein in its given environment.
    first_res: str
        The first residue single amino acid code. Used for determining
        the charge of the first residue amine group.
    last_res: str
        The last residue single amino acid code. Used for determining
        the charge of the last residue carboxyl group.


    Returns
    -------
    rtype: dict
        Float dictionary with charges for each atom type.
    """
    amine_pka = {
        'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67,
        'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60,
        'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80,
        'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15,
        'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.11,
    }
    carboxyl_pka = {
        'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19,
        'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
        'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02,
        'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
        'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.20,
    }
    pKa = {
        'OD2': 3.65,  # ASP
        'OE2': 4.25,  # GLU
        'NZ': 10.53,  # LYS
        'NH1': 12.48,  # ARG
        'ND1': 6.00,  # HIS
        'OXT': carboxyl_pka[last_res],  # C-TERM
        'N': amine_pka[first_res]  # N-TERM
    }
    fN = lambda pk: 1 / (1 + 10 ** (pH - pk)) if np.abs(1 / (1 + 10 ** (pH - pk))) > 0.05 else 0
    fO = lambda pk: -1 / (1 + 10 ** -(pH - pk)) if np.abs(-1 / (1 + 10 ** -(pH - pk))) > 0.05 else 0

    # determine the point charge of each atom type
    return {
        _: fN(pKa[_]) if _[0] == 'N' else fO(pKa[_])
        for _ in pKa.keys()
    }


def compute_dielectric(temp) -> float:
    """
    Determine the relative permittivity of protein
    in response to temperature. It should be noted
    that the mode.

    Parameters
    ----------
    temp: float
        Temperature of the folding experiment in Kelvin.

    Returns
    -------
    rtype: float
        Returns the relative permittivity
    """
    coef_ = np.array([
        -1.37468329e-06, 2.06837364e-03, -1.22690384e+00, 2.96744195e+02
    ])
    X = np.array([temp ** i for i in np.arange(0, 4)[::-1]])
    return (X * coef_).sum().round(1)


def compute_ionic_strength(
        conc, pH, buffer='phosphate', salt_conc=0, salt='NaCl',
) -> float:
    """
    Function to compute ionic strength of a known solution.

    Parameters
    ----------
    conc: float
        The concentration of the buffer in Molar (M).
    pH: float
        pH of the solution, must be a value between 0 and 14.
    buffer: str
        One of three buffer types 'tris', 'phosphate', or 'hepes'.
    salt_conc: float
        The concentration of the added salt in Molar (M).
    salt: str
        The type of salt added options are 'NaCl' and 'MgCl2'.

    Returns
    -------
    rtype: float
        Returns the ionic strength of the buffer.
    """
    pKa_lib = {
        'tris': (8.07, [-1, 0]),
        'phosphate': (7.20, [-2, -1]),
        'hepes': (7.48, [-1, 0])
    }
    salt_lib = {
        'NaCl': [(salt_conc, 1), (salt_conc, -1)],
        'MgCl2': [(salt_conc, 1), (2 * salt_conc, -1)],
    }

    # debug
    if buffer not in pKa_lib.keys():
        raise Exception('Incorrect Buffer Option')
    if pH < 0 or pH > 14:
        raise Exception('Incorrect pH Option')

    # effect of salt
    salt_str = np.array([
        c * (i ** 2) for c, i in salt_lib[salt]
    ]).sum()

    # get the pKa and charges of each species
    pKa, (base_charge, acid_charge) = pKa_lib[buffer]

    # compute ionic strength of base and acid
    if (base_charge, acid_charge) == (-1, 0):
        b_str = conc * (base_charge ** 2)
        s_str = conc * (np.abs(base_charge) ** 2)
        return (b_str + s_str + salt_str) / 2

    # for polyprotic acids
    # determine the fraction of populations use Henderson-Hasselbalch
    eq = 10 ** (pH - pKa)
    p_a = 1 / (1 + eq)
    p_b = 1 - p_a
    c_a = p_a * conc
    c_b = p_b * conc
    b_str_ = (c_a * (acid_charge ** 2)) + (c_b * (base_charge ** 2))
    s_str_ = (c_a * np.abs(acid_charge)) + (c_b * np.abs(base_charge))
    return (b_str_ + s_str_ + salt_str) / 2


def compute_debye_length(ionic_strength, temp=298) -> float:
    """
    Function to compute the Debye length of a protein in nanometers
    This function requires the ionic strength in M units and temp in
    kelvin.

    Parameters
    ----------
    ionic_strength: float
        The ionic strength of the solution.
    temp: float
        The temperature of the solution.

    Returns
    -------
    rtype: float
        Returns the debye length in nm.
    """
    # first turn ionic strength into number/m3 units
    i = (N_A * ionic_strength) * 1e3

    # permittivity calculation
    epsilon = compute_dielectric(temp) * epsilon_0
    kappa = np.sqrt((epsilon * kB * temp) / (2 * (ele ** 2) * i))

    # return in nm
    return (kappa * 1e9).round(3)
