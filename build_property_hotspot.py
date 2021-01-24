"""
Update: normalize the B factor by Z score (x-mui)/sd.
Description:
The script is for generating an input file for prediction of deamidation by machine learning. It collects structural properties including B-factors, local secondary structure, percentage solvent accessible area, psi and phi angles as well as sequence based deamidation half life from Robinson et al 2001 data.

The script read into a csv file contains 3 columns: PDB ID, residue number of key asparagines, and deamidation observation. The script also require a set of standard PDB files of the protein of interest and csv files contains amino acid residue properties which are output from Discovery Studio. deamidation_half_life.txt contains half life data of key asparagine deamidation which was obtained from Robinson et al.

The script output a csv file contains 11 columns: PDB ID, residue number of key asparagines, residue name after asparagines, deamidation half life, norm_B_factor_CA, norm_B_factor_CB, norm_B_factor_CG, secondary_structure, percentage solvent accessibility (PSA), percentage sidechain solvent accessibility (PSSA), Psi and Phi angles, and deamidation observation. The B_factors have been normalized through the whole protein.

Usage and example (it's in python 2 syntax):
python build_property_hotspot.py <input_file> <output_file>

------------------------------------------------------------------------
This is a personally developed freeware. There is no warrenty. Use it at your own risk. The script may be bug-rich. It can be improved by anyone like you.
---------------------------------------------------------------------------------------------------------------------------------------------------
                                                                            
"""

import os, re, sys, math
import numpy as np

# Dihedral angle calcualtion function.

def dihedral(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))

# Main function.

if (len(sys.argv) != 3):
    print "python build_property.py <input_file> <output_file>"

# Readin and format deamidation half life into dictionaries.
deamidation_halflife_file = file('deamidation_half_life.txt', 'r')
deamidation_halflife_file.readline()
deamidation_halflife_table = [np.array(map(float, line.split()[1:])) for line in deamidation_halflife_file]
deamidation_halflife_file.close()

aa1_index = {'GLY':0, 'SER':1, 'THR':2, 'CYS':3, 'MET':4, 'PHE':5, 'TYR':6, 'ASP':7, 'GLU':8, 'HIS':9, 'LYS':10, 'ARG':11, 'ALA':12, 'LEU':13, 'VAL':14, 'ILE':15, 'TRP':16, 'PRO':17}
aa2_index = {'GLY':0, 'HIS':1, 'SER':2, 'ALA':3, 'ASP':4, 'THR':5, 'CYS':6, 'LYS':7, 'MET':8, 'GLU':9, 'ARG':10, 'PHE':11, 'TYR':12, 'TRP':13, 'LEU':14, 'VAL':15, 'ILE':16}

# Readin input data files.
infile = file(sys.argv[1], 'r')
outfile = file(sys.argv[2], 'w')

for line in infile:
    extract_line = line.rstrip('\r\n').split(',')
    if (extract_line[0] == 'PDB'):
        outfile.write('PDB,Residue,aa2,attack_distance,Half_life,norm_B_factor_C,norm_B_factor_CA,norm_B_factor_CB,norm_B_factor_CG,secondary_structure,PSA,PSSA,Psi,Phi,Chi1,Chi2,Deamidation\n')
        continue
    pdb_file = 'cleaned/' + extract_line[0] + '.pdb'
    csv_file = 'cleaned/' + extract_line[0] + '.csv'
    residue_no = int(extract_line[1])
    deamidation = extract_line[2]

# Average B factors through pdb file.
    in_pdb_file = file(pdb_file, 'r')
    B_factor_sum = 0
    B_factor_count = 0
    for line_pdb in in_pdb_file:
        if (line_pdb[0:4] == 'ATOM'):
            B_factor_sum += float(line_pdb[60:66])
            B_factor_count += 1
    B_factor_ave = B_factor_sum / B_factor_count
    print B_factor_ave
    in_pdb_file.close()

# Calculate STD of B factors through pdb file
    in_pdb_file = file(pdb_file, 'r')
    B_factor_variance_sum = 0
    B_factor_count = 0
    for line_pdb in in_pdb_file:
        if (line_pdb[0:4] == 'ATOM'):
            B_factor_variance_sum += (float(line_pdb[60:66])-B_factor_ave)**2
            B_factor_count += 1
    B_factor_variance = B_factor_variance_sum/(B_factor_count - 1)
    B_factor_std = math.sqrt(B_factor_variance)
    print B_factor_std
    in_pdb_file.close()

# Obtain amino acid before and after asn information and obtain normalized B factors for ASN C, CA, CB, CG.
    aa1_absent = True
    aa2_absent = True
    N_absent = True
    B_factor_CA = '0'
    B_factor_CB = '0'
    B_factor_CG = '0'
    B_factor_C = '0'
    in_pdb_file = file(pdb_file, 'r')
    for line_pdb in in_pdb_file:
        if (line_pdb[0:4] == 'ATOM' and int(line_pdb[23:26]) == (residue_no-1)):
            aa1 = line_pdb[17:20]
            aa1_absent = False
        if (line_pdb[0:4] == 'ATOM' and int(line_pdb[23:26]) == (residue_no+1)):
            aa2 = line_pdb[17:20]
            aa2_absent = False
            if (line_pdb[13:15] == 'N '):
                Nx = float(line_pdb[30:38])
                Ny = float(line_pdb[38:46])
                Nz = float(line_pdb[46:54])
                N_absent = False
        if (line_pdb[0:4] == 'ATOM' and int(line_pdb[23:26]) == residue_no):
            if (line_pdb[17:20] != 'ASN'):
                B_factor_CA = 'NAN'
                B_factor_CB = 'NAN'
                B_factor_CG = 'NAN'
                B_factor_C = 'NAN'
            elif (line_pdb[13:15] == 'CA'):
                B_factor_CA = str((float(line_pdb[60:66])-B_factor_ave)/B_factor_std)
            elif (line_pdb[13:15] == 'C '):
                B_factor_C = str((float(line_pdb[60:66])-B_factor_ave)/B_factor_std)
            elif (line_pdb[13:15] == 'CB'):
                B_factor_CB = str((float(line_pdb[60:66])-B_factor_ave)/B_factor_std)
            elif (line_pdb[13:15] == 'CG'):
                B_factor_CG = str((float(line_pdb[60:66])-B_factor_ave)/B_factor_std)
                CGx = float(line_pdb[30:38])
                CGy = float(line_pdb[38:46])
                CGz = float(line_pdb[46:54])
# Calculate nucleophilic attack distance between Asn side chain amide C and next residue mainchain amino N.
    if (N_absent == False and B_factor_C != 'NAN'):
        attack_distance = str(math.sqrt((Nx-CGx)**2+(Ny-CGy)**2+(Nz-CGz)**2))
    else:
        attack_distance = 'NAN'
    in_pdb_file.close()

# Calculate dihedral angles of ASN side chain.
    switch = False
    in_pdb_file = file(pdb_file, 'r')
    for line_pdb in in_pdb_file:
        if (line_pdb[0:4] == 'ATOM' and int(line_pdb[23:26]) == residue_no):
            if (line_pdb[17:20] == 'ASN'):
                switch = True
                if(line_pdb[13:15] == 'C '):
                    Cx = float(line_pdb[30:38])
                    Cy = float(line_pdb[38:46])
                    Cz = float(line_pdb[46:54])
                if(line_pdb[13:15] == 'CA'):
                    CAx = float(line_pdb[30:38])
                    CAy = float(line_pdb[38:46])
                    CAz = float(line_pdb[46:54])
                if(line_pdb[13:15] == 'CB'):
                    CBx = float(line_pdb[30:38])
                    CBy = float(line_pdb[38:46])
                    CBz = float(line_pdb[46:54])
                if(line_pdb[13:16] == 'ND2'):
                    ND2x = float(line_pdb[30:38])
                    ND2y = float(line_pdb[38:46])
                    ND2z = float(line_pdb[46:54])
    if switch:
        p1 = np.array([
            [Cx,    Cy, Cz],
            [CAx,   CAy,    CAz],
            [CBx,   CBy,    CBz],
            [CGx,   CGy,    CGz]
            ])
        p2 = np.array([
            [CAx,   CAy,    CAz],
            [CBx,   CBy,    CBz],
            [CGx,   CGy,    CGz],
            [ND2x,  ND2y,   ND2z]
            ])
        chi1 = str(dihedral(p1))
        chi2 = str(dihedral(p2))
    else:
        chi1 = 'NAN'
        chi2 = 'NAN'
    in_pdb_file.close()

# Obtain deamidation half life for each ASN from the half life file.
    if (aa1_absent):
        aa1 = 'GLY'
    if (aa2_absent):
        aa2 = 'NAN'
    if (aa2_index.has_key(aa2)):
        if (not(aa1_index.has_key(aa1))):
            aa1 = 'GLY'
        deamidation_halflife = str(deamidation_halflife_table[aa1_index[aa1]][aa2_index[aa2]])
    elif (aa2 == 'PRO'):
        deamidation_halflife = '999'
    else:
        deamidation_halflife = 'NAN'

# Obtain secondary structure, PSA, PSSA, Psi and Phi from DS output csv files.
    in_csv_file = file(csv_file, 'r')
    for line_csv in in_csv_file:
        AAproperties = line_csv.rstrip('\r\n').split(',')
        if (AAproperties[0] == 'Name'):
            for i in range (34):
                if (AAproperties[i] == 'ID'):
                    ID_no = i
                if (AAproperties[i] == 'PDB Name'):
                    res_name = i
                if (AAproperties[i] == 'Secondary'):
                    sec_no = i
                if (AAproperties[i] == 'Percent Solvent Accessibility'):
                    psa_no = i
                if (AAproperties[i] == 'Percent Sidechain Solvent Accessibility'):
                    pssa_no = i
                if (AAproperties[i] == 'Psi'):
                    psi_no = i
                if (AAproperties[i] == 'Phi'):
                    phi_no = i
                secondary_structure = '0'
                PSA = '0'
                PSSA = '0'
                Psi = '0'
                Phi = '0'
            continue
        if (int(AAproperties[ID_no]) == residue_no and AAproperties[res_name] == 'ASN'):
            if (AAproperties[sec_no] == 'Helix'):
                secondary_structure = '1'
            elif (AAproperties[sec_no] == 'Sheet'):
                secondary_structure = '2'
            elif (AAproperties[sec_no] == 'Turn'):
                secondary_structure = '3'
            elif (AAproperties[sec_no] == 'Coil'):
                secondary_structure = '4'
            else:
                secondary_structure = '5'
            PSA = AAproperties[psa_no]
            PSSA = AAproperties[pssa_no]
            Psi = AAproperties[psi_no]
            Phi = AAproperties[phi_no]
        if (int(AAproperties[ID_no]) == residue_no and AAproperties[res_name] != 'ASN'):
            secondary_structure = AAproperties[res_name]
    in_csv_file.close()

# Output file.
    newline = extract_line[0] + ',' + extract_line[1] + ',' + aa2 + ',' + attack_distance + ',' + deamidation_halflife + ',' + B_factor_C + ',' + B_factor_CA + ',' + B_factor_CB + ',' + B_factor_CG + ',' + secondary_structure + ',' + PSA + ',' + PSSA + ',' + Psi + ',' + Phi + ',' + chi1 + ',' + chi2 + ',' + deamidation + '\n'
    print newline
    outfile.write(newline)
infile.close()
outfile.close()
