# FESpy - A Python Module for Computing Energy Surfaces
The following module is a key package used in my PhD research where a series of physical information is computed using
the folding kinetics of a protein ($k_f$ the folding rate and $k_u$ the unfolding rate), pH, temperature, and the PDB (protein data bank)
file of the structure of the protein.

## Properties of the `FES` Object
The python modules `openmm`, `mdtraj`, and `scipy` are key dependencies used in `FESpy` as they are needed for computing the
following information:
- **One Dimensional Free Energy Surface** based on the model proposed in the 
[following publications](https://doi.org/10.1039/C1CP20402E). Where a function for $\Delta G$, $\Delta H$ and $\Delta S$
is computed along a single reaction coordinate *Nativeness* which is how folded the protein is (1 is fully folded, 0 is 
unfolded).
- **Topology Parameters** which are single arbitrary values that gives you global information about the protein's secondary
structure. The computed parameters are [contact order (CO)](https://doi.org/10.1006/JMBI.1998.1645), 
[absolute contact order (ACO)](https://doi.org/10.1110/PS.0302503), [total contact distance (TCD)](https://doi.org/10.1016/S0006-3495(02)75410-6),
and [long-range order (LRO)](https://doi.org/10.1006/JMBI.2001.4775).
- **Alpha Carbon Description** of the protein. This includes the physical $C_\alpha$ - $C_\alpha$ distances along with the 
distance of each alpha carbon pairing along the primary amino acid sequence.
- **All Atom Description** of the protein. This includes the all heavy atom descriptors like atom pair distances, charges,
the type of atom paring (van der Waals, hydrogen-bonding, electrostatic). All distances and pairing information are computed
using `mdtraj` library.
