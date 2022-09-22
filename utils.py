import os, getopt

import numpy as np
from openbabel import openbabel as ob


def get_options(argv):
   opts, args = getopt.getopt(argv, "hl:t:f:j:s:u:", ["help", "ligand=", "target=", "forceField=", "trajectories=", "steps=", "temperature="])
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -l <ligand> -t <target>')
         sys.exit()
      elif opt in ("-l", "--ligand"):
         ligand = arg
      elif opt in ("-t", "--target"):
         target = arg
      elif opt in ("-f", "--forceField"):
         ff = arg
      elif opt in ("-j", "--trajectories"):
         trajectories = arg
      elif opt in ("-s", "--steps"):
         steps = arg
      elif opt in ("-u", "--temperature"):
         temperature = arg

   return target, ligand, ff, trajectories, steps, temperature

def readfile(f):
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("sdf")

    conv.ReadFile(mol, f)

    return mol

def readfile_xyz(f):
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("xyz")

    conv.ReadFile(mol, f)

    return mol

def writefile(mol, f):
    conv = ob.OBConversion()
    conv.SetOutFormat("sdf")

    conv.WriteFile(mol, f)

def writefile_xyz(mol, f):
    conv = ob.OBConversion()
    conv.SetOutFormat("xyz")

    conv.WriteFile(mol, f)




def too_close(mol, startid, endid):
    com1 = np.array([0.0, 0.0, 0.0])
    com2 = np.array([0.0, 0.0, 0.0])

    atom = ob.OBAtom()

    for i in range(1, startid-1):
        atom = mol.GetAtom(i)
        tmp = np.array([ atom.GetX(), atom.GetY(), atom.GetZ() ])
        com1 += tmp

    com1 /= (startid-1)

    atom = ob.OBAtom()
    for i in range(startid, endid):
        atom = mol.GetAtom(i)
        tmp = np.array([ atom.GetX(), atom.GetY(), atom.GetZ() ])
        com2 += tmp

    com2 /= (endid - startid)

    dist = np.linalg.norm(com1-com2)

    if dist > 1.0:
        return True
    else:
        False

def check_num_mols(mol):
    mols = mol.Separate()

    if len(mols) == 2:
        return True
    else:
        return False


def get_com(mol):
    com = np.array([0.0, 0.0, 0.0])
    atom = ob.OBAtom()

    for i in range(mol.NumAtoms()):
        atom = mol.GetAtom(i+1)
        tmp = np.array([ atom.GetX(), atom.GetY(), atom.GetZ() ])
        com += tmp

    com /= mol.NumAtoms()

    return com

def move_molecule(mol, move, startid=1, endid=-1):
    atom =ob.OBAtom()

    if endid == -1:
        endid = mol.NumAtoms() + 1

    for i in range(startid, endid):
    #for i in range(startid-1, endid):
        atom = mol.GetAtom(i)
        temp = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        temp += move
        atom.SetVector(temp[0], temp[1], temp[2])

def minimize_molecule(mol, force_field):
    ff = ob.OBForceField.FindForceField(force_field)
    ff.Setup(mol)
    ff.ConjugateGradients(2000)
    e = ff.Energy()

    ff.GetCoordinates(mol)
    mol.SetEnergy(e)

    return e, mol

def rotate(V, J, T):
    x = V[0]
    y = V[1]
    z = V[2]

    u = J[0]
    v = J[1]
    w = J[2]

    norm = np.sqrt(u*u + v*v + w*w)
    inv_norm_sqrt = 1.0 / (norm*norm)

    sint = np.sin(T)
    cost = np.cos(T)

    a = (u * (u*x + v*y + w*z) + (x * (v*v + w*w) - u * (v*y + w*z)) * cost + norm * (-w*y + v*z) * sint) * inv_norm_sqrt
    b = (v * (u*x + v*y + w*z) + (y * (u*u + w*w) - v * (u*x + w*z)) * cost + norm * ( w*x - u*z) * sint) * inv_norm_sqrt
    c = (w * (u*x + v*y + w*z) + (z * (u*u + v*v) - w * (u*x + v*y)) * cost + norm * (-v*x + u*y) * sint) * inv_norm_sqrt

    rotated = np.array([a, b, c])

    return rotated

def rotate_molecule(mol, direction, theta, startid=1, endid=-1):
    if endid == -1:
        endid = mol.NumAtoms() + 1

    com = np.array([0.0, 0.0, 0.0])
    atom = ob.OBAtom()

    for i in range(startid, endid):
        atom = mol.GetAtom(i)
        com += np.array([atom.GetX(), atom.GetY(), atom.GetZ()])

    com /= (endid - startid)

    for i in range(startid, endid):
        atom = mol.GetAtom(i)
        temp = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        temp -= com
        temp = rotate(temp, direction, theta)
        temp += com
        atom.SetVector(temp[0], temp[1], temp[2])

def random_vector():
    rng = np.random.default_rng()
    vec = rng.random((3,))
    x = np.random.choice([-1., 1.], size=1)
    y = np.random.choice([-1., 1.], size=1)
    z = np.random.choice([-1., 1.], size=1)

    vec_tmp = np.array([ x*vec[0], y*vec[1], z*vec[2] ])
    vec = vec_tmp.reshape(3,)

    return vec

def uniform1():
    rng = np.random.default_rng()
    return rng.random()

def random_angle():
    rng = np.random.default_rng()
    return 90./180.*np.pi * rng.random()

def random_length():

    rng = np.random.default_rng()
    #return 2 * rng.random()
    ranInt = rng.integers(low=0, high=30, size=1)


    if ranInt == 0:
        return 0
    else:
        return ranInt / 100.

def set_conformations(mol, force_field):
    ff = ob.OBForceField.FindForceField(force_field)
    ff.Setup(mol)

    rmsd_cutoff   = 0.5
    energy_cutoff = 50.0
    conf_cutoff   = 10000000
    verbose       = False

    ff.DiverseConfGen(rmsd_cutoff, conf_cutoff, energy_cutoff, verbose)
    ff.GetConformers(mol)


    return mol

def plus_equal_mols(target, ligand, numTot):
    builder  = ob.OBBuilder()
    atom     = ob.OBAtom()
    atom2    = ob.OBAtom()

    if target.NumAtoms() < numTot:
        mol_new  = ob.OBMol()
        mol_new = target
        mol_new.SetCoordinates(target.GetCoordinates())
        numAtoms = target.NumAtoms()

        atom0      = []
        atom1      = []
        bond_order = []

        mol_bonds = ob.OBMolBondIter(ligand)

        for bond in mol_bonds:
            atom0.append(bond.GetBeginAtomIdx())
            atom1.append(bond.GetEndAtomIdx())
            bond_order.append(bond.GetBondOrder())

        for i in range(ligand.NumAtoms()):
            atom = ligand.GetAtom(i+1)
            mol_new.AddAtom(atom)

        for i in range(len(atom0)):
            builder.Connect(mol_new, numAtoms+atom0[i], numAtoms+atom1[i], bond_order[i])

    else:
        mol_new  = ob.OBMol()
        mol_new = target
        mol_new.SetCoordinates(target.GetCoordinates())
        numAtoms = target.NumAtoms() - ligand.NumAtoms()
        for i in range(ligand.NumAtoms()):
            atom = ligand.GetAtom(i+1)
            X = atom.GetX()
            Y = atom.GetY()
            Z = atom.GetZ()

            atom2 = mol_new.GetAtom(numAtoms+i+1)
            atom2.SetVector(X, Y, Z)

    return mol_new


