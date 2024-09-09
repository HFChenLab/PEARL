# PEARL
Personalized Energy Adaptation through Reweighting Learning (PEARL) Force Field for Intrinsically Disordered Proteins
# Usage
The DeepReweighting method is employed to optimize LJ parameters and RNA-specific CMAP parameters. For further details, please refer to this repository: https://github.com/Junjie-Zhu/DeepReweight. For optimizing CMAP parameters of proteins, please refer to the usage examples below:
`python3 DeepReweighting_protein_CMAP_system.py --dir '/path/to/input/predict_prop' --seq '/path/to/input/fasta' --name '/path/to/input/system' --coil_dir '/path/to/input/coil_database_prop/' --basis_dir '/path/to/input/basis_dir/' --weight '0.01' `
