
import pandas as pd
import numpy as np


## Only consider general amino acid
## padding amino acid X is not considered.
letterDict = {"A": 0,
              "C": 1,
              "D": 2,
              "E": 3,
              "F": 4,
              "G": 5,
              "H": 6,
              "I": 7,
              "K": 8,
              "L": 9,
              "M": 10,
              "N": 11,
              "P": 12,
              "Q": 13,
              "R": 14,
              "S": 15,
              "T": 16,
              "V": 17,
              "W": 18,
              "Y": 19}

#letterDict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
#                  "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19, "U": 20, "B": 21}

def encodePeptideByInteger(peptide:str,max_length=50):
    AACategoryLen = len(letterDict)
    peptide_length = len(peptide)
    use_peptide = peptide
    if max_length is not None:
        if peptide_length < max_length:
            use_peptide = peptide + "X" * (max_length - peptide_length)

    vec = [letterDict[aa]+1 if aa in letterDict.keys() else 0 for aa in use_peptide]
    return vec

def encodePeptides(x,max_length=50,seq_encode_method="one_hot"):
    ## encoded data
    #n_aa_types = len(letterDict)
    #peptide_length = len(x['x'].iloc[0])
    #encoded_data = np.zeros((x.shape[0], peptide_length, n_aa_types))
    k = 0
    #for i, row in x.iterrows():
    #    peptide = row['x']
    #    encoded_data[k] = encodePeptideOneHot(peptide)
    #    k = k + 1
    if seq_encode_method == "one_hot":
        encoded_data = np.array([encodePeptideOneHot(peptide,max_length=max_length) for peptide in x['x']])
    else:
        ## integer representation
        encoded_data = np.array([encodePeptideByInteger(peptide, max_length=max_length) for peptide in x['x']])

    return encoded_data


def add_mod(mod=None):

    i = len(letterDict)
    mod = sorted(mod)
    for m in mod:
        if str(m) not in letterDict:
            letterDict[str(m)] = i
            print("New aa: %s -> %d" % (str(m),i))
            i = i + 1

def load_aa(file):
    print("Load aa coding data from file %s" % (file))
    dat = pd.read_table(file, sep="\t", header=0, low_memory=False)
    letterDict.clear()
    for i, row in dat.iterrows():
        letterDict[row['aa']] = row['i']

def save_aa(file):
    print("Save aa coding data to file %s" % (file))
    with open(file,"w") as f:
        f.write("aa\ti\n")
        for aa in letterDict.keys():
            f.write(aa+"\t"+str(letterDict[aa])+"\n")


def encodePeptideOneHot(peptide: str, max_length=None, add_reverse=False):  # changed add one column for '1'
    """
    Used by AutoRT
    :param peptide:
    :param max_length:
    :return:
    """

    AACategoryLen = len(letterDict)
    peptide_length = len(peptide)
    use_peptide = peptide
    if max_length is not None:
        if peptide_length < max_length:
            use_peptide = peptide + "X" * (max_length - peptide_length)


    if add_reverse is True:
        #max_length = max_length + max_length
        use_peptide = use_peptide + use_peptide[::-1]

    en_vector = np.zeros((len(use_peptide), AACategoryLen))

    i = 0
    for AA in use_peptide:
        if AA == "X":
            i = i + 1
            continue
        elif AA in letterDict.keys():
            en_vector[i][letterDict[AA]] = 1
            i = i + 1
        else:
            print("Error: invalid amino acid: %s" % (str(AA)))
            exit(1)
            #en_vector[i] = np.full(AACategoryLen,1/AACategoryLen)

    return en_vector



def encodePeptideOneHot_new(peptide: str, max_length=None):  # changed add one column for '1'
    """
    Experimental function
    :param peptide:
    :param max_length:
    :return:
    """

    AACategoryLen = len(letterDict)
    peptide_length = len(peptide)
    use_peptide = peptide
    #if max_length is not None:
    #    if peptide_length < max_length:
    #        use_peptide = peptide + "X" * (max_length - peptide_length)

    en_vector = np.zeros((max_length, AACategoryLen))

    i = 0
    for AA in use_peptide:
        if AA in letterDict.keys():
            try:
                en_vector[i][letterDict[AA]] = 1
            except:
                print("peptide: %s, i => aa: %d, %s, %d" % (use_peptide,i, AA, letterDict[AA]))
                exit(1)
        else:
            print("Error: invalid amino acid: %s" % (str(AA)))
            exit(1)
            #en_vector[i] = np.full(AACategoryLen,1/AACategoryLen)

        i = i + 1

    return en_vector

