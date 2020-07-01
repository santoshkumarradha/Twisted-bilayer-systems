import numpy as np
from pymatgen.core import Structure
import numba as nb



@nb.njit()
def get_nm(theta=1.2,error=1e-3):
    """returns the pair of n,m for a given angle
    
    Args:
        theta (float): angle to find n,m
        error (float, optional): error to which searchm defailt 1e-3
    
    Returns:
        TYPE: [n,m],angle
    """
    k=False
    for n in range(1,2000):
        for m in range(1,2000):
            ang=np.rad2deg(np.arccos(  0.5*(n**2+4*n*m+m**2)/(n**2+n*m+m**2) ))
            if np.abs(ang-theta)<error:
                nm=[n,m]
                k=True
                break
            if k:break
        if k:break
    return nm,ang


def tblg(angle=3.,error=1e-3,verbose=True):
    """get the pymatgen structure of TBLG for given angle
    
    Args:
        angle (float, optional): angle of twist
        verbose (bool,optional): verbosity
    """
    nm,angle=get_nm(angle,error)
    if verbose:
        print("Angle = {}".format(np.round(angle)))
    return tblg_nm(n=nm[0],m=nm[1])


def tblg_nm(n=9,m=8):
    """Get the pymatgen structure of the twisted bilayer graphene for given n and m.
    angle is given by arccos[ 0.5*(n**2+4*n*m+m**2) / (n**2+n*m+m**2)  ]
    
    Args:
        n (int, optional): sc 1
        m (int, optional): sc 2
    
    Returns:
        TYPE: pymatgen structure object
    """
    def R(t): #Rotation matrix
        c, s = np.cos(t), np.sin(t)
        R = np.array(((c, -s), (s, c)))
        R=np.vstack([R,[0,0]]);R=np.hstack([R,[[0],[0],[1]]])
        return R
    t=np.arccos(0.5* (n**2+4*n*m+m**2) / (n**2+n*m+m**2)  )
    lattice_lower=[3*R(t/2)@i for i in np.array([[1.0,0.0,0],[0.5,np.sqrt(3.0)/2.0,0],[0,0,10]])]  # lower lattice
    lattice_upper=[3*R(-t/2)@i for i in np.array([[1.0,0.0,0],[0.5,np.sqrt(3.0)/2.0,0],[0,0,10]])]  # top lattice
    lower=Structure(lattice=lattice_lower,species=["C","C"],coords=[[1./3.,1./3.,0.5],[2./3.,2./3.,0.5]]) 
    upper=Structure(lattice=lattice_upper,species=["C","C"],coords=[[1./3.,1./3.,0.5],[2./3.,2./3.,0.5]])

    sc=np.eye(3)
    sc[0][0]=n;sc[0][1]=m
    sc[1][0]=-m;sc[1][1]=n+m
    lower.make_supercell(sc) # supercell transfornation matrix for lower system

    sc=np.eye(3)
    sc[0][0]=m;sc[0][1]=n
    sc[1][0]=-n;sc[1][1]=m+n
    upper.make_supercell(sc) # supercell transfornation matrix for upper system


    d=1  # distance between layers
    pos=[]
    for i in upper:pos.append(i.coords+[0,0,d])
    for i in lower:pos.append(i.coords+[0,0,-d])
    species=["C"]*len(pos)
    tblg=Structure(lattice=upper.lattice,coords=pos,species=species,coords_are_cartesian=True)
    # print("angle of twist = {}".format(np.rad2deg(np.arccos(  0.5*(n**2+4*n*m+m**2)/(n**2+n*m+m**2)   ))))
    return tblg #return the twisted bilayer structure