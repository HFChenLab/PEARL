# PEARL
Personalized Energy Adaptation through Reweighting Learning (PEARL) Force Field for Intrinsically Disordered Proteins
# Usage
The DeepReweighting method is employed to optimize LJ parameters and RNA-specific CMAP parameters. For further details, please refer to this repository: https://github.com/Junjie-Zhu/DeepReweight. 

For optimizing CMAP parameters of proteins, please refer to the usage examples below:

`python3 DeepReweighting_protein_CMAP_system.py --dir '/path/to/input/predict_prop' --seq 'fasta sequence' --name 'system' --coil_dir '/path/to/input/coil_database_prop/' --basis_dir '/path/to/input/basis_dir/' --weight '0.01' `

Required parameters:

* --dir: the path to the predict probablity of each amino acid
* --seq: the fasta sequence
* --name: the system name
* --coil_dir: the path to coil database
* --basis_dir: the path to basis trajectory of reweighting
* --weight: the weight for eRMSD, default is 1e-2
