import numpy as np
from pysktb import *
import pickle


def tblg_nm(n=9, m=8):
    import numpy as np
    from pymatgen.core import Structure
    """Get the pymatgen structure of the twisted bilayer graphene for given n and m.
    angle is given by arccos[ 0.5*(n**2+4*n*m+m**2) / (n**2+n*m+m**2)  ]
    Args:
        n (int, optional): sc 1
        m (int, optional): sc 2
    Returns:
        TYPE: pymatgen structure object
    """
    def R(t):  # Rotation matrix
        c, s = np.cos(t), np.sin(t)
        R = np.array(((c, -s), (s, c)))
        R = np.vstack([R, [0, 0]])
        R = np.hstack([R, [[0], [0], [1]]])
        return R
    t = np.arccos(0.5 * (n**2+4*n*m+m**2) / (n**2+n*m+m**2))
    lattice_lower = [3*R(t/2)@i for i in np.array([[1.0, 0.0, 0],
                                                   [0.5, np.sqrt(3.0)/2.0, 0], [0, 0, 10]])]  # lower lattice
    lattice_upper = [3*R(-t/2)@i for i in np.array([[1.0, 0.0, 0],
                                                    [0.5, np.sqrt(3.0)/2.0, 0], [0, 0, 10]])]  # top lattice
    lower = Structure(lattice=lattice_lower, species=["C", "C"], coords=[
                      [1./3., 1./3., 0.5], [2./3., 2./3., 0.5]])
    upper = Structure(lattice=lattice_upper, species=["C", "C"], coords=[
                      [1./3., 1./3., 0.5], [2./3., 2./3., 0.5]])

    sc = np.eye(3)
    sc[0][0] = n
    sc[0][1] = m
    sc[1][0] = -m
    sc[1][1] = n+m
    # supercell transfornation matrix for lower system
    lower.make_supercell(sc)

    sc = np.eye(3)
    sc[0][0] = m
    sc[0][1] = n
    sc[1][0] = -n
    sc[1][1] = m+n
    # supercell transfornation matrix for upper system
    upper.make_supercell(sc)

    d = 1  # distance between layers
    pos = []
    for i in upper:
        pos.append(i.coords+[0, 0, d])
    for i in lower:
        pos.append(i.coords+[0, 0, -d])
    species = ["C"]*len(pos)
    tblg = Structure(lattice=upper.lattice, coords=pos,
                     species=species, coords_are_cartesian=True)
    print("angle of twist = {}".format(np.rad2deg(
        np.arccos(0.5*(n**2+4*n*m+m**2)/(n**2+n*m+m**2)))))
    return tblg  # return the twisted bilayer structure


def ham_tblg(struc):
    atoms = []
    for i in struc:
        atoms.append(Atom(i.species_string, i.frac_coords))
    for i in atoms:
        i.set_orbitals(["s"])

    d = 0
    lattice = Lattice(struc.lattice.matrix, 1)
    interactions = {"C": {"e_s": 0}, "CC": {"V_sss": -.5}}

    nn = 1.75
#     nn=np.unique(np.sort(struc.distance_matrix.flatten()))[1]
    bond = {"CC": {"NN": nn}}

    s = Structure(lattice, atoms, bond_cut=bond)
    ham = Hamiltonian(s, interactions, numba=1)
    return ham


print("Starting to make the structure...\n")
tblg_structure = tblg_nm(28, 27)
print("Done...\n")

print("Starting Hamiltonian construction...\n")
ham = ham_tblg(tblg_structure)
print("Done...\n")

print("Saving...\n")
with open('tblg.pickle', 'wb') as handle:
    pickle.dump(ham, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Done...\n")
