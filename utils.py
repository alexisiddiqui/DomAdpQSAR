### Utils - TODO - update to use params

# This python file contains utility functions for the domain adaptation project
# This includes generating fingerprints (from RDKIT/MOlBERT), calculating similarity scores (Tanimoto)
import numpy as np
from rdkit.Chem import AllChem
import pandas as pd
import json


# get either ECFP4 fingerprint or bitvector (optional) from SMILES or from MOLBERT (Future)




# define function that transforms SMILES strings into ECFPs
def get_fingerprint(smiles,
                    FP_type = 'ECFP4' or 'FCFP2' or 'MOLBERT',
                    R = 2, ### these should be moved to params
                    L = 2**10, ### these should be moved to params
                    use_features = False,
                    use_chirality = False):
    """
    Inputs:
    
    - smiles ... SMILES string of input compound
    - R ... maximum radius of circular substructures
    - L ... fingerprint-length
    - use_features ... if false then use standard DAYLIGHT atom features, if true then use pharmacophoric atom features
    - use_chirality ... if true then append tetrahedral chirality flags to atom features
    
    Outputs:
    - np.array(feature_list) ... ECFP with length L and maximum radius R
    """
    if FP_type is "FCFP2":
        use_features = True

    if FP_type is not "MOLBERT":
        molecule = AllChem.MolFromSmiles(smiles)
        feature_list = AllChem.GetMorganFingerprintAsBitVect(molecule,
                                                            radius = R,
                                                            nBits = L,
                                                            useFeatures = use_features,
                                                            useChirality = use_chirality)
        return np.array(feature_list)
    
    elif type is "MolBERT":
        raise NotImplementedError("MolBERT is not implemented yet")
    

import numpy as np
from numpy import dot
from numpy.linalg import norm

def calculate_tanimoto_similarity(fp1, fp2):
    # Convert fingerprints to numpy arrays
    try:
        # for pandas dataframes
        arr1 = fp1.to_numpy()[0]
        arr2 = fp2.to_numpy()[0]
    except:
        try:
            # for lists
            arr1 = np.asarray(fp1)
            arr2 = np.asarray(fp2)
        except:
            raise ValueError("Input fingerprints must be pandas slices or lists/arrays")

    # Calculate dot product and norm of each fingerprint
    dot_prod = dot(arr1, arr2)
    norm1 = norm(arr1)
    norm2 = norm(arr2)
    
    # Calculate Tanimoto similarity
    similarity = dot_prod / (norm1**2 + norm2**2 - dot_prod)
    return similarity


# function that calculates the average similarity score of a single compound to the Target set
def calculate_target_similarity(FP, target_set, simi_type='Tanimoto', mean="mean"):
    similarity_scores = []
    if simi_type == 'Tanimoto':
        for target_FP in target_set.FP:
            # print(FP, target_FP)
            similarity_score = calculate_tanimoto_similarity(FP, target_FP)
            similarity_scores.append(similarity_score)
    else:
        raise NotImplementedError("simi_type must be 'Tanimoto'")
    if mean is not None:
        if mean == 'mean':
            return np.mean(similarity_scores)
        elif mean == 'median':
            return np.median(similarity_scores)
        else:
            raise NotImplementedError("mean type must be None, 'mean', or 'median'")
    else:
        return similarity_scores

#function that calculates the similarity scores of a set of compounds to a target set returns a dataframe with the similarity scores added
def calculate_set_similarity(FP_set, target_set, simi_type='Tanimoto', mean="mean"):
    similarity_scores = []
    if simi_type == 'Tanimoto':
        for FP in FP_set.FP:
            similarity_score = calculate_target_similarity(FP, target_set, simi_type=simi_type, mean=mean)
            similarity_scores.append(similarity_score)
    else:
        raise NotImplementedError("simi_type must be 'Tanimoto'")
    FP_set_ranked = FP_set.copy()
    FP_set_ranked[simi_type] = similarity_scores

    return FP_set_ranked


### TODO change the format of this so we can use it for the lhasa parameters

# def save_parameters(parameters, file_name):
#     with open(parameter_folder + file_name + '.json', 'w') as fp:
#         json.dump(parameters, fp, indent=4)

# def load_parameters(file_name):
#     with open(parameter_folder + file_name + '.json') as json_file:
#         return json.load(json_file)
    