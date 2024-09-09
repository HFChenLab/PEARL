import os
import time
import numpy as np
import re
import sys
import subprocess
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', type=str, default='./')
parser.add_argument('--seq', '-s', type=str, default='AAAAA')
parser.add_argument('--name', '-n', type=str, default='ACTR')
parser.add_argument('--coil_dir', '-w', type=str, default='./')
parser.add_argument('--target_dir', '-w', type=str, default='./')
parser.add_argument('--basis_dir', '-w', type=str, default='./')
parser.add_argument('--weight', '-w', type=str, default='0.01')
args, _ = parser.parse_known_args()
AminoAcid = {
    'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C',
    'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I',
    'LEU' : 'L', 'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P',
    'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V'
}
AminoAcidX = dict(zip(AminoAcid.values(), AminoAcid.keys()))

def main():
    weight = args.weight
    for file in os.listdir(args.dir):
        file_path = os.path.join(args.dir, file)
        if os.path.isfile(file_path):
            parts = file.split('_')
            first_char = file[-12]
            ss = file[0:-11]
            index = int(parts[0])
            coil_path = f"{args.coil_dir}/{first_char}_coil.txt" #coil database
            dihedral_path = f"{args.basis_dir}/{first_char}_ff19SB_opc_600.dat"  #basis
            output_path = f"./{args.name}/{args.name}_{weight}"
            cmap_path = f"./{args.name}/{args.name}_{weight}/{ss}_CMAP.dat"  #CMAP parameters
            reweight_path = f"./{args.name}/{ss}_reweight.dat"
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            subprocess.run(["python3", "DeepReweighting_protein_CMAP_single.py", "--input", dihedral_path, "--output", cmap_path, "--target", file_path,  "--weight", weight, "--reweight", reweight_path])
    for i,res in enumerate(args.seq[1:-1]):
        charmPara = open(f'./{args.name}/{args.name}_{weight}/'+str(i+1)+'_'+ res +'_CMAP.dat','r')
        para = []
        for line in charmPara.readlines():
            line = line.strip('\n')
            if not re.match('^[!a-zA-Z]',line):
                data = re.split(r'[,\s]\s*', line)
                data = list(filter(None, data))
                data = list(map(float, data))
                para.append(data)
        paras = str(para)
        paras = paras.replace('[', '')
        paras = paras.replace(']', '')
        paras = list(eval(paras))
        paras = np.array(paras)

        amberPara = open(f'./{args.name}/{args.name}_{weight}.para','a+')
        amberPara.write('%%FLAG %s_MAP\n' % AminoAcidX[res])
        amberPara.write('%FORMAT(8(F9.5))\n')
        for j,data in enumerate(paras):
            amberPara.write('%9.5f' % data)
            if j % 8 == 7:
                amberPara.write('\n')
        amberPara.close()

if __name__ == '__main__':
    main()
