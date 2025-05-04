# COMS W2132 Intermediate Computing in Python, Final Project 
## Phi Psi Comparison

### Author:
- [Polina Goldberg](https://github.com/...) <pmg2147@columbia.edu>

## Project Description
In protein structure, phi and psi angles are backbone diherdal angles that define the geometry and conformation of the protein chain. Comparing these angles between proteins can help understanding folding patterns, conformational changes upon drug binding or genetic mutations, and much more. However, individually calculating and comparing every angle in a full-length protein is a tedious task involving loading relevant structures into PyMol, extracting the angles from the command line, parsing this information into a datable (typically wih excel), aligning the protein sequences, and then finally carrying out calculations. 

![Phi and Psi angles](PhiPsiDiagram.png)

This project automates this  protein structure comparison task through a Python script that essentially compiles these five steps into one program. Once the differences in angles are calculated, the program will notify the users of any areas of interest and provide some simple visualizations.

## Requirements 
The program will require the following python packages: os, shutil, time, re, pandas, subprocess, tempfile, matplotlib, biopython, and ramachandraw. The program also requires that the PyMOL application be installed on the computer system.

## Instructions
When running the program, you will be prompted to input two PDB files. This can be either in the from of a Protein Data Bank entry ID or a filepath within the program directory. You will also be prompted to enter a minimum angle threshold by which to compare the proteins in the PDB files.

Important Notes:
* The program is intended to be used on the same protein in two different conformational states or on two similar proteins. Comparing two nonrelated PDB files will lead to nonsensical results.
* The program assumes the proteins use UNIPROT residue numbering (typical for PDB entries) or some other standardized numbering system set by the user.

The program will output a dataframe of all residues which have a phi or psi angle difference above the specified threshold. These residues can be identified by the first three columns, namely chain number (Chain), residue name (ResName), and residue number (Residue). The last two columns, Phi_diff and Psi_diff, give the difference in phi and psi angles, respectively.

## Example Inputs 

If you don't know where to start, here are some example pairs of proteins you can input into the program! All of these are PDB entry id codes. The recommended minimum angle for comparison is 60°.

* 1AKE and 4AKE
* 4YU3 and 4YU4

## Milestones 
Milestone 1: Extract and process the phi psi angles from two given PDB files. List the protein residues whose phi or psi angle differences exceed a given value.

Milestone 2: Provide data visualizations corresponding to the computed phi psi differences (e.g. Ramachandran Plots, line graphs)
