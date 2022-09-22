# pythonyMcDock

mcdock.cpp from Andersx (https://github.com/andersx/mcdock.git) rewritten in python.

## Install mcDock.py

Clone the repository:
```
{
git clone https://github.com/heini-phys-chem/pythonyMcDock.git
}
```
This code uses the python wrapper (openbabel, NOT pybel) from the OpenBabel code.
Therefore, openbabel need to be installed. One way is to use `conda` to install OpenBabel:
```
{
conda create --name openbabel
conda activate openbabel
conda install -c conda-forge openbabel
}
```
## Running mcDock.py

```
{
./mcDock.py --ligand small_conformer.sdf --target glucosepane.sdf  --forceField UFF --trajectories 15 --steps 200 --temperature 0.3
}
```
