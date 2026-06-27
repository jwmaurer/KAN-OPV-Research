"""
Jacob Maurer
6/13/2026
Purpose: Create the images of the molecules, dropping the no variance columns and the highly correlated columns.
"""

# credit to: https://github.com/CNUClasses/utils/blob/master/transforms.py for drop no variance and drop correlated
import csv

import numpy as np
from import_dataset import open_csv_dataset, open_hopv15_dataset
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageOps

def drop_no_variance_columns(df,verbose=True):
    '''
    drops all columns that only have 1 value
    df: a dataframe to inspect
    return: df columns that only have 1 value dropped
    '''
    vals=df.nunique()
    
    #get list of columns that only have 1 value
    todrop=[df.dtypes.index[i] for i,val in enumerate(df.nunique()) if val ==1]
    
    #bail if no columns to drop
    if not todrop:
        return df
    
    if(verbose==True):
         print(f'dropping columns {todrop} since each only has 1 value')
    
    #drop em
    df.drop(columns=todrop, inplace=True)
    
    return df

def get_correlated_columns(df,correlation_threshold =.95):
    '''
    df: a dataframe
    correlation_threshold: select all rows and columns that have a correlation >= to this value
    return: list of tuples of form [ (col,row),...]
    '''
    #make sure we do correlations on non-object columns only
    df = df.loc[:, df.dtypes != 'object']
    
    # generate the correlation matrix (abs converts to absolute value, this way we only look for 1 color range)
    corr = df.corr().abs()
    # Generate mask for the upper triangle (see https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
    # the matrix is symmetric, the diagonal (all 1's) and upper triangle are visual noise, use this to mask both out
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)    #k=-1 means get rid of the diagonal
    corr = corr.where(cond=mask)
    
    correlated=[]
    for col in corr.columns:
        for i,val in enumerate(corr.loc[col]):
            if( val>= correlation_threshold):
                correlated.append((col,corr.loc[col].index[i]))
    return correlated

def drop_correlated_columns(df,correlation_threshold = .95, verbose=True):
    '''
    Drops 1 of each 2 correlated columns
    CAREFUL WITH THIS ONE< YOU WANT TO DROP THE COLUMN WITH THE LEAST INFORMATION
    df: a dataframe
    return: df with 1 of each 2 correlated columns dropped
    '''
    correlated = get_correlated_columns(df, correlation_threshold)
    while correlated:
        if (verbose==True):
            print(f'dropping column {correlated[0][0]} which is correlated with {correlated[0][1]}')
            
        df.drop(columns=[correlated[0][0]], inplace=True)
        correlated = get_correlated_columns(df, correlation_threshold)
    return df

if __name__ == "__main__":
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    csv_data = open_csv_dataset("./OPV_Datasets/opv_db.csv")['smile']
    smiles = [molecule.smiles_molecule for molecule in data if not np.isnan(molecule.experimental_data.PCE)]
    smiles.extend(csv_data.to_list())
    smiles = np.unique(smiles)
    with open("./OPV_Datasets/molecule_images.csv", 'at') as file:
        writer = csv.writer(file, delimiter=',')
        for i, smile in enumerate(smiles):
            print(f"Item {i}, Smile: {smile}")
            molecule = Chem.AddHs(Chem.MolFromSmiles(smile))
            image = np.asarray(ImageOps.grayscale(Draw.MolToImage(molecule))).flatten()
            writer.writerow(image)
        