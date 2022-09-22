#!/usr/bin/env python3

import sys, os
import warnings

import numpy as np
import openbabel
from openbabel import openbabel as ob

from utils import *

warnings.filterwarnings("ignore")

def main():
    print("\nUsing OpenBabel version: {}\n------------------------------\n".format(openbabel.__version__))
    f_target, f_ligand, force_field, trajectories, steps, temperature = get_options(sys.argv[1:])


    target = readfile(f_target)
    ligand = readfile(f_ligand)
    print(" [+] Read molecules")

    # center ligand molecule
    com = get_com(ligand)
    print(" [+] get center of mass (ligand)")

    move_molecule(ligand, -com)
    print(" [+] move molecule to center (ligand)")

    # center target molecule
    com = get_com(target)
    print(" [+] get center of mass (target)")

    move_molecule(target, -com)
    print(" [+] move molecule to center (target) \n")

    # get energy of target molecule
    ea, target = minimize_molecule(target, force_field)
    eb, ligand = minimize_molecule(ligand, force_field)

    numTot = target.NumAtoms() + ligand.NumAtoms()
    e_low  = np.Inf
    eb_min = np.Inf

    ff = ob.OBForceField.FindForceField(force_field)
#    conv = ob.OBConversion()
#    conv.SetInAndOutFormats("xyz", "xyz")
#    os.system("rm -f conformers.xyz")
    print("Perfomring rotor search for ligand molecule      file: {}".format(f_ligand))
    ligand = set_conformations(ligand, force_field)

    numConfs = ligand.NumConformers()

    for c in range(1, numConfs+1):
        ligand.SetConformer(c)

        eb, ligand = minimize_molecule(ligand, force_field)

        if eb < eb_min:
            eb_min = eb

        print("Rotamere {}     E = {:.4f} kcal/mol".format(str(c), eb))

    print("Lowers energy conformation E = {:.4f} kcal/mol".format(eb_min))
    print("Running {} trajectories for {} steps".format(trajectories, steps))
    print("MC temperature (tau) = {}".format(temperature))




    print("\nConformation:       Trajectory:         Acceptance rate:    Final Ebind:")
    print("---------------------------------------------------------------------------")

    for c in range(1, numConfs+1):
        ligand.SetConformer(c)

        eb, ligand = minimize_molecule(ligand, force_field)

        for n in range(int(trajectories)):
            # translate molecule
            direction = random_vector()
            com = get_com(ligand)
            temp = direction * 4.0 - com
            move_molecule(ligand, temp)
            rot = random_vector()
            theta = random_angle()# * np.pi
            rotate_molecule(ligand, rot, theta)

            # roll back to rejected MC moves
            mol_ligand = ob.OBMol()
            mol_old    = ob.OBMol()
            mol_ligand = plus_equal_mols(target, ligand, numTot)
            mol_old    = mol_ligand
            mol_old.SetCoordinates(mol_ligand.GetCoordinates())

            ff.Setup(mol_ligand)

            e = ff.Energy()

            energy_old = e
            delta_e = 0.0
            accept = 0

            startid = mol_ligand.NumAtoms() - ligand.NumAtoms() + 1
            endid   = mol_ligand.NumAtoms() + 1

            # MC simulation
            for step in range(int(steps)):
                move = random_vector()
                move *= random_length()
                move_molecule(mol_ligand, move, startid=startid, endid=endid)

                rot   = random_vector()
                theta = random_angle()
                rotate_molecule(mol_ligand, rot, theta, startid=startid, endid=endid)

                ff.SetCoordinates(mol_ligand)

                e = ff.Energy()

                delta_e = e - energy_old
                range_test = check_num_mols(mol_ligand)

                # Metropolis-Hastings MC criterian, accept ...
                threash_hold = uniform1()
                if np.exp(-delta_e / float(temperature)) >= threash_hold:# and range_test:
                    mol_old.SetCoordinates(mol_ligand.GetCoordinates())
                    energy_old = e
                    accept += 1
                else:
                    mol_ligand.SetCoordinates(mol_old.GetCoordinates())
                    e = energy_old


            ec, mol_ligand = minimize_molecule(mol_ligand, force_field)
            #ff.SetCoordinates(mol_ligand)
            #ec = ff.Energy()

            e_bind = ec - (ea + eb)

            acceptance_ratio = accept * 100.0 / (float(steps) + 1)
            print("{} / {}\t\t\t{} / {}\t\t\t{:.2f}\t\t{:.2f}\tkcal/mol".format(c, numConfs, n+1, trajectories, acceptance_ratio, e_bind), end = '')


            writefile_xyz(mol_ligand, "out_{}_{}.xyz".format(c, n))


            if (e_bind < e_low):
                print("     <----- New Lowest")
                e_low = e_bind
                writefile_xyz(mol_ligand, "min.xyz")
            else:
                print()

    os.system("cat out_*.xyz > out.xyz")
    os.system("rm -rf out_*.xyz")

    print("\n\n [+] Optimization done")

if __name__ == '__main__':
    main()