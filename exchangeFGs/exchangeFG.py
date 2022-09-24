#!/usr/bin/env python3

import sys, os
import random
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

def exchange_FG(mol_sub, FG, atom1Idx, atom2Idx):
    mol_tmp = ob.OBMol()
    mol_tmp = mol_sub
    mol_tmp.SetCoordinates(mol_sub.GetCoordinates())

    atom_target = ob.OBMol()
    atom_FG     = ob.OBMol()

    atom_target = mol_tmp.GetAtom(atom2Idx)
    coords = np.array( [ atom_target.GetX(), atom_target.GetY(), atom_target.GetZ() ] )

    mol_tmp.DeleteAtom(atom_target)
    numAtoms = mol_tmp.NumAtoms()
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
        mol_tmp.AddAtom(atom_FG)

    for i in range(len(atom0)):
        builder.Connect(mol_tmp, numAtoms+atom0[i], numAtoms+atom1[i], bond_order[i])

    builder.Connect(mol_tmp, atom1Idx, numAtoms+1, 1)

    return mol_tmp

def get_XH_bonds(mol):
    atom1 = ob.OBAtom()
    atom2 = ob.OBAtom()
    bond  = ob.OBBond()
    numAtoms = mol.NumAtoms()

    CH_bonds = []

    for i in range(numAtoms):
        for j in range(numAtoms):
            if i == j: continue

            atom1 = mol.GetAtom(i+1)
            atom2 = mol.GetAtom(j+1)

            if atom1.GetBond(atom2) != None and atom2.GetAtomicNum() == 1:
                #print("Bond Found between {} and {}".format(atom1.GetIdx(), atom2.GetIdx()))
                CH_bonds.append( [atom1.GetIdx(), atom2.GetIdx()] )

    return CH_bonds


def main():
    filename    = sys.argv[1]
    force_field = sys.argv[2]

    FG = ["oh.sdf", "ch3.sdf", "nh2.sdf"]


    for i in range(50):
        mol = readfile(filename)
        com = get_com(mol)

        move_molecule(mol, com)

        e, mol = minimize_molecule(mol, "UFF")
        XH_bonds = get_XH_bonds(mol)


        mol_FG = ob.OBMol()
        mol_FG = readfile(random.choice(FG))
        mol_sub = ob.OBMol()
        mol_sub = mol
        mol_sub.SetCoordinates(mol.GetCoordinates())

        atoms = random.choice(XH_bonds)

        print("First energy: {:.2f}".format(e))
        mol_sub = exchange_FG(mol_sub, mol_FG, atoms[0], atoms[1])

        e, mol = minimize_molecule(mol_sub, "UFF")
        print("FG energy: {:.2f}".format(e))

        writefile(mol, "test_{}.sdf".format(i))


if __name__ == '__main__':
    main()
