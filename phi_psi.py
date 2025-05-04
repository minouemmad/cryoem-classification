#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:56:31 2025

@author: polinagoldberg
"""

import os
import shutil
import time
import re
import pandas as pd
import subprocess
import tempfile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ramachandraw.utils import plot
from Bio.PDB import PDBList

def display_greeting():
    
    '''
    Displays how to use the program and any outputs given.
    
    '''
    
    print('Welcome to the φ/ψ comparison tool.')
    print('\n')
    time.sleep(1)  
    print('You will be prompted to input two PDB files (PDB id or filepath)')
    print('and a minimum angle threshold by which to compare them')
    print('\n')
    time.sleep(5)
    print('The program will output a list of any amino acid residues whose φ/ψ')
    print('angles exceed the minimum threshold angle')
    print('\n')
    time.sleep(5)
    print('Important notes:')
    time.sleep(1)
    print(' * the program is intended to be used on the same protein in two')
    print('   different conformational states or on two similar proteins')
    time.sleep(5)
    print(' * the program assumes the proteins use UNIPROT numbering (typical')
    print('   for PDB entries) or use some other standardized numbering system')
    print('   set by the user')

    
    

def fetch_protein():
    
    '''
    
    Asks user for the protein in either PDB entry ID or file path format. 
    
    Returns:
        
        PDB entry ID or file path (str)
        entry type (PDB entry ID or file path)
    
    '''
    
    pdb_entry_id_check = input('Do you have an entry ID from the Protein Data Bank (PDB) (y/n): ').lower()
    
    #check for invalid responses
    while pdb_entry_id_check not in ['y','n']:
        
        time.sleep(0.5)
        print("Invalid entry. Enter 'y' or 'n'")
        time.sleep(1.5)
        pdb_entry_id_check = input('Do you have an entry ID from the Protein Data Bank (PDB) (y/n): ').lower()
        
    
    #obtain PDB ID if available
    if pdb_entry_id_check =='y':
        
        protein = input('Please input the PDB entry ID: ')
        entry_type = 'id'
    
    #else obtain PDB filepath
    else:
        
        protein = input('Please input PDB file path: ')
        entry_type = 'file path'
        
        
    return protein, entry_type
    


def get_min_angle():
    
    '''
    
    obtains user input for minimum angle of comparison
    
    '''
    
    ang = int(input('Enter a minimum angle difference: '))
    
    #ensure angle is within range
    ang = ang % 360

    #get smallest circular angle
    if ang > 180:
        
        ang =  360 - ang
    
    return ang



def fetch_phi_psi_raw(protein, proteintype):
    
    '''
    
    Loads protein into PyMOL and obtains its phi psi values
    
    
    Args: 
        
        protein: PDB entry ID or file path (str)
        proteintype: type of entry (PDB ID or file path)
        
    Returns:
        
        df containing all residues with respective phi and psi angles.
    
    '''
    
    #make pymol command 'fetch' if input is a PDB id
    if proteintype == 'id':
        pml_cmd = f'fetch {protein}, async=0' #asyn makes sure structure fully loads 
        target = protein
    
    #else use 'load' to load in PDB file from filepath
    else:
        file_name = os.path.splitext(os.path.basename(protein))[0]
        pml_cmd = f'load {protein}'
        target = file_name
    
    #full pymol command to be given to script
    full_pml_commands = "\n".join([pml_cmd,
                                   "remove not alt ''+A",
                                   "alter all, alt=''",
                                   'sort',
                                   f'phi_psi {target}'])
    
    #creating tempfile with full script
    with tempfile.NamedTemporaryFile('w', suffix='.pml', delete=False) as f:
        f.write(full_pml_commands)
        script = f.name
    
    #run pymol script and return output
    out = subprocess.check_output(['pymol', '-cq', script])
    out = out.decode()
    
    #regular expression for each residue's phi psi value output
    pattern = r'(\w+)-(\d+):\s+\(\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\)'
    
    #find all instances of the phi_psi output pattern in out
    matches = re.findall(pattern, out)
    
    #stack all matches into a df
    df = pd.DataFrame(matches, columns=['ResName', 'Residue', 'Phi', 'Psi'])

    #convert following columns to int or float
    df['Residue'] = df['Residue'].astype(int)
    df['Phi'] = df['Phi'].astype(float)
    df['Psi'] = df['Psi'].astype(float)
    
    
    #adding chain ids for multimeric complexes
    chain_id = 1
    chain_id_series = []
    
    previous_residue = None
    
    for residue in df['Residue']:
        
        #if previous residue exists and is larger than next residue (chain reset)
        if previous_residue and previous_residue >= residue:
            chain_id +=1
            
        chain_id_series.append(chain_id)
        previous_residue = residue
    
    
    df['Chain'] = chain_id_series
    
    return df
    

def circular_diff(angle1: pd.Series, angle2: pd.Series):
    
    '''
    
    Calculates the phi psi values differences between the two proteins
    
    Args:
        angle1: pandas series containing dihedral angle information from protein 1
        angle2: pandas series containing dihedral angle information from protein 2
        
    Returns:
        minimum angle difference
    
    
    '''
    
    #raw absolute difference
    diff = (angle1-angle2).abs() % 360
    
    #wherever angle is greater than 180, get the smaller angle to take into account 360° space
    return diff.mask(diff > 180, 360 - diff)



def analysis(df1, df2, angle):
    
    '''
    
    Obtains dataframe of all protein residues whose phi or psi angle differences exceed a given value
    
    Args:
        
        df1: pandas dataframe contianing phi psi information from first protein
        df2: pandas dataframe containing phi psi information from second protein
        angle: minimum angle 
        
    Returns:
        
        dataframe of all protein residues whose phi or psi angle differences exceed a given value.
    
    '''
    
    #remove any residues/backbones that are not amino acids (e.g. ligand)
    df1_new = df1[df1['ResName'].notna()].reset_index()
    df2_new = df2[df2['ResName'].notna()].reset_index()
    
    both = pd.merge(df1_new, df2_new, on=['Chain','ResName','Residue'], how = 'inner', suffixes = ('_1','_2'))
    
    both['Phi_diff'] = circular_diff(both['Phi_1'], both['Phi_2'])
    both['Psi_diff'] = circular_diff(both['Psi_1'], both['Psi_2'])
    
    mask = (both['Phi_diff'] >= angle) | (both['Psi_diff'] >= angle)
       
    notable = both.loc[mask, ['Chain','ResName','Residue','Phi_diff','Psi_diff']].copy()
    notable = notable.reset_index(drop = True)
    
    return both, notable
    

def getpdb_fromid(pdb_id):
    
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format= 'pdb')
    
    return pdb_file

    

def run_analysis():
    
    
    '''
    Runs program
    
    '''
    
    old_items = set(os.listdir('.'))
    
    print('Protein 1: ')
    protein1, protein1_type = fetch_protein()
    
    print('\n')
    print('Protein 2: ')
    protein2, protein2_type = fetch_protein()
    
    print('\n')
    angle = get_min_angle()
    
    
    protein1_angles = fetch_phi_psi_raw(protein1, protein1_type)
    protein2_angles = fetch_phi_psi_raw(protein2, protein2_type)

    all_residues, notable_residues = analysis(protein1_angles, protein2_angles, angle)
    
    if notable_residues.empty:
        print('\n')
        print('No residues exceed that threshold.')
    
    else:
        print('\n')
        print(notable_residues.to_string(index=False))
        
    
    
    #plot ramachandran plots
    pdb_file1 = getpdb_fromid(protein1) if protein1_type == 'id' else protein1
    pdb_file2 = getpdb_fromid(protein2) if protein2_type == 'id' else protein2
    
    plot([pdb_file1, pdb_file2], cmap = 'viridis', alpha = 0.75, dpi = 100, show = False)
    
    ax = plt.gca()
    
    custom_handles = [Line2D([],[], marker = 'o', color = 'blue', linestyle = 'None', markersize = 4),
                      Line2D([],[], marker = 'o', color = 'orange', linestyle = 'None', markersize = 4)]
    short_legend_form = [os.path.basename(pdb_file1), os.path.basename(pdb_file2)]
    
    ax.legend(handles = custom_handles, labels = short_legend_form)
    ax.set_title('Ramachandran Plot Comparison')
    
    plt.show()
    
    
    
    #plot line graph (for main chain only)
    plt.figure(figsize=(8,4))
    
    chain1 = all_residues[all_residues['Chain']==1]
    
    plt.plot(chain1['Residue'], chain1['Phi_diff'], color = '#F95700FF', linestyle = '-', label = 'φ difference')
    plt.plot(chain1['Residue'], chain1['Psi_diff'], color = '#00B1D2FF', linestyle = '-', label = 'Ψ difference')
    
    plt.xlabel('Residue Number')
    plt.ylabel('Dihedral Angle Difference (°)')
    plt.title('φ/Ψ Difference by Residue for Main Chain')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
    
    #delete any newly downloaded files from the directory
    items_after = set(os.listdir('.'))
    new_items = items_after - old_items
    
    for item in new_items:
        
        path = os.path.join('.', item)
        
        if os.path.isfile(path):
            os.remove(path)
        
        if os.path.isdir(path):
            shutil.rmtree(path)



def phi_psi_analysis():
    
    display_greeting()
    
    #allow user to run program multuple times
    again = 'y'
    while again == 'y':
        
        run_analysis()
        print('\n')
        again = input('Would you like to run another analysis (y/n)? ')
        
        while again not in ['y','n']:
            
            time.sleep(0.5)
            print('\n')
            print("Invalid entry. Response must be 'y' or 'n'")
            time.sleep(1.5)
            print('\n')
            again = input('Would you like to run another analysis (y/n)? ').lower()
    
    print('\n')
    print('Session complete.')



if __name__ == '__main__':
    phi_psi_analysis()
    
    
    
    


