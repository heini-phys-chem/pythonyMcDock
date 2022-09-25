#!/usr/bin/env python3

import sys, os
import random
import warnings

import numpy as np
import openbabel
from openbabel import openbabel as ob

from utils import *

# Suppress RuntimeWarning from first Metropolis-Hastings MC criterian, accept
warnings.filterwarnings("ignore")
os.system("rm -f out.xyz min.xyz")

def main():
    print("\nUsing OpenBabel version: {}\n------------------------------\n".format(openbabel.__version__))
    # Read cmd line arguments
    f_target, f_ligand, force_field, trajectories, steps, temperature, mutations = get_options(sys.argv[1:])

    FG = ["oh.sdf", "ch3.sdf", "nh2.sdf"]
    for k in range(int(mutations)):
        # Read in molecules
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


        XH_bonds = get_XH_bonds(target)
        mol_FG = ob.OBMol()
        mol_FG = readfile(random.choice(FG))
        mol_sub = ob.OBMol()
        mol_sub = target
        mol_sub.SetCoordinates(target.GetCoordinates())
        atoms = random.choice(XH_bonds)
        mol_sub = exchange_FG(mol_sub, mol_FG, atoms[0], atoms[1])

        # get energy of target molecule
        ea, target = minimize_molecule(mol_sub, force_field)
        eb, ligand = minimize_molecule(ligand, force_field)

        numTot = target.NumAtoms() + ligand.NumAtoms()
        e_low  = np.Inf
        eb_min = np.Inf

        # Construct force field object
        ff = ob.OBForceField.FindForceField(force_field)


        # Perfomr conformer search
        print(" Perfomring rotor search for ligand molecule      file: {}".format(f_ligand))
        ligand = set_conformations(ligand, force_field)

        numConfs = ligand.NumConformers()

        for c in range(1, numConfs+1):
            ligand.SetConformer(c)

            eb, ligand = minimize_molecule(ligand, force_field)

            if eb < eb_min:
                eb_min = eb

            print(" Rotamere {}     E = {:.4f} kcal/mol".format(str(c), eb))

        print(" Lowers energy conformation E = {:.4f} kcal/mol".format(eb_min))
        print(" Running {} trajectories for {} steps".format(trajectories, steps))
        print(" MC temperature (tau) = {}".format(temperature))


        # Start the MC simulation
        print("\n Conformation:         Trajectory:         Acceptance rate:               Final Ebind:")
        print(" -------------------------------------------------------------------------------------")

        # Loop over conformers
        for c in range(1, numConfs+1):
            ligand.SetConformer(c)

            eb, ligand = minimize_molecule(ligand, force_field)

            # For every conformer loop over number of trajectories
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

                # MC simulation for every trajectory
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

                e_bind = ec - (ea + eb)

                acceptance_ratio = accept * 100.0 / (float(steps) + 1)
                acceptance_ratio = round(acceptance_ratio, 2)
                print(" {} / {}\t\t\t{} / {}\t\t\t{}\t%\t\t{:4.2f}\tkcal/mol".format(c, numConfs, str(n+1).zfill(2), trajectories, str(acceptance_ratio).zfill(2), e_bind), end = '')


                writefile_xyz(mol_ligand, "out_{}_{}_{}.xyz".format(c, n, k))


                if (e_bind < e_low):
                    print("     <----- New Lowest")
                    e_low = e_bind
                    writefile_xyz(mol_ligand, "min.xyz")
                else:
                    print()

        os.system("cat out_*.xyz >> out.xyz")
        os.system("rm -rf out_*.xyz")

        print("\n\n [+] Optimization done")

if __name__ == '__main__':
    main()
