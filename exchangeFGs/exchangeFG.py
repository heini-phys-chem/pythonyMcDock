#!/usr/bin/env python3

import sys, os
import numpy as np

import openbabel
from openbabel import openbabel as ob


def readfile(f):
    '''
    input:   filename (sdf)
    returns: openbabel mol object
    '''
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("sdf")

    conv.ReadFile(mol, f)

    return mol

def writefile(mol, f):
    '''
    input:   mol object, filename (sdf)
    returns: -
    '''
    conv = ob.OBConversion()
    conv.SetOutFormat("sdf")

    conv.WriteFile(mol, f)

def get_com(mol):
    '''
    get center of mass
    '''
    com = np.array([0.0, 0.0, 0.0])
    atom = ob.OBAtom()

    for i in range(mol.NumAtoms()):
        atom = mol.GetAtom(i+1)
        tmp = np.array([ atom.GetX(), atom.GetY(), atom.GetZ() ])
        com += tmp

    com /= mol.NumAtoms()

    return com


def move_molecule(mol, move, startid=1, endid=-1):
    '''
    move ligand using move vector
    '''
    atom =ob.OBAtom()

    if endid == -1:
        endid = mol.NumAtoms() + 1

    for i in range(startid, endid):
        atom = mol.GetAtom(i)
        temp = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        temp += move
        atom.SetVector(temp[0], temp[1], temp[2])


def minimize_molecule(mol, force_field):
    '''
    optimizes molecule using a force field
    returns optimized geometry and energy
    '''
    ff = ob.OBForceField.FindForceField(force_field)
    ff.Setup(mol)
    ff.ConjugateGradients(2000)
    e = ff.Energy()

    ff.GetCoordinates(mol)
    mol.SetEnergy(e)

    return e, mol

def exchange_FG(mol, FG):
    atom_target = ob.OBMol()
    atom_FG     = ob.OBMol()

    atom_target = mol.GetAtom(9)
    coords = np.array( [ atom_target.GetX(), atom_target.GetY(), atom_target.GetZ() ] )

    mol.DeleteAtom(atom_target)
    numAtoms = mol.NumAtoms()
#    writefile(mol, "test.sdf")

    builder = ob.OBBuilder()

    FG_bonds = ob.OBMolBondIter(FG)

    atom0      = []
    atom1      = []
    bond_order = []

    for bond in FG_bonds:
        atom0.append(bond.GetBeginAtomIdx())
        atom1.append(bond.GetEndAtomIdx())
        bond_order.append(bond.GetBondOrder())

    for i in range(FG.NumAtoms()):
        atom_FG = FG.GetAtom(i+1)
        mol.AddAtom(atom_FG)

    for i in range(len(atom0)):
        builder.Connect(mol, numAtoms+atom0[i], numAtoms+atom1[i], bond_order[i])

    builder.Connect(mol, 1, numAtoms+1, 1)

    return mol

def main():
    filename    = sys.argv[1]
    force_field = sys.argv[2]

    FG = ["oh.sdf", "ch3.sdf", "nh2.sdf"]

    mol_oh  = readfile(FG[0])
    mol_ch3 = readfile(FG[1])
    mol_nh2 = readfile(FG[2])

    mol = readfile(filename)
    com = get_com(mol)

    move_molecule(mol, com)

    e, mol = minimize_molecule(mol, "UFF")

    print("First energy: {:.2f}".format(e))
    mol = exchange_FG(mol, mol_nh2)

    e, mol = minimize_molecule(mol, "UFF")
    print("FG energy: {:.2f}".format(e))

    writefile(mol, "test.sdf")


if __name__ == '__main__':
    main()
