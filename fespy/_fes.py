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
    - General Thermodynamics
        1. Enthalpy: DHres, DH(n), and kappaDH
        2. Entropy: DSres, DS(n)
        3. Free Energy: DG(n), temp
    - General Kinetics
        1. Folding Kinetics (kf, ku) and (pf, pu)
        2. Stability
        3. PreExponential (k0)
    - Structural Parameters
        1. Topology: CO, ACO
        2. Global Charge given pH
"""


class FES:
    """
    FES: Free Energy Surface

    OOP interface for calculating 1D projection of energy surface
    for single domain proteins. The model assumes kinetics can be
    determined via diffusion on the energy surface.

    Attributes
    ----------
    """
    def __init__(self) -> None:
        """

        """
        pass
