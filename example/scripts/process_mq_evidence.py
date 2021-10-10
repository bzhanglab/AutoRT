import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import re
import argparse


def get_confident_mod_pos(ptm_name: str, mod_pep: str, ptm_res: str, site_prob_cutoff=0.75):
    # ptm_name: example: Phospho (STY)
    # mod_pep: AAAAAAAATM(0.002)ALAAPSSPTPESPTM(0.998)LTK
    # ptm_res: 
    # example: Phospho (STY)
    # example: Acetyl (Protein N-term),Phospho (STY)
    # example: 2 Oxidation (M),2 Phospho (STY)
    # print("ptm_name:%s, mod_pep:%s, ptm_res:%s" % (ptm_name,mod_pep,ptm_res))
    if isNaN(mod_pep):
        return None, None
    elif ptm_name in ptm_res:
        ptms = ptm_res.split(",")
        n_mod = 0
        for ptm in ptms:
            if ptm_name in ptm:
                ptm_m = ptm.replace(ptm_name, "")
                ptm_m = re.sub(pattern=r"\s+", repl="", string=ptm_m)
                if re.search(pattern=r"\d", string=ptm_m):
                    n_mod = int(ptm_m)
                else:
                    n_mod = 1

        probs = re.findall(r"\(([0-9.]*)\)", mod_pep)
        # probs: ['0.002', '0.998']
        probs = [float(v) for v in probs]
        # print(probs)
        mod_pep_dot = re.sub(pattern=r"\([0-9.]*\)", repl=".", string=mod_pep)
        # mod_pep_dot: AAAAAAAATM.ALAAPSSPTPESPTM.LTK
        # print(mod_pep_dot)
        # get position of ptms
        k = 0
        mod_pos = list()
        for i in range(0, len(mod_pep_dot)):
            if mod_pep_dot[i] == ".":
                k = k + 1
                mod_pos.append(i - k)
        # [9, 24]
        # print(mod_pos)

        # check
        if n_mod > len(probs) or n_mod > len(mod_pos) or len(probs) != len(mod_pos):
            print("Modification parse error: n_mod=%d, len(probs)=%d, len(mod_pos)=%d!" % (
                n_mod, len(probs), len(mod_pos)))
            sys.exit()

        # position -> prob
        pos2prob = dict()
        for i in range(len(mod_pos)):
            pos2prob[mod_pos[i]] = probs[i]
        pos2prob_list = sorted(pos2prob.items(), key=lambda x: x[1], reverse=True)

        valid_pos_i = list()
        valid_prob_i = list()
        for k in pos2prob_list:
            if len(valid_pos_i) >= n_mod:
                break
            else:
                if k[1] >= site_prob_cutoff:
                    valid_pos_i.append(k[0])
                    valid_prob_i.append(k[1])

        if len(valid_pos_i) == n_mod:
            # confident identification
            valid_pos_i = np.array(valid_pos_i)
            return valid_pos_i, valid_prob_i
        else:
            return None, None
    else:
        # no modification
        return None, None


def get_ptm_name(ptm_res: str):
    # ptm_res: 
    # example: Phospho (STY)
    # example: Acetyl (Protein N-term),Phospho (STY)
    # example: 2 Oxidation (M),2 Phospho (STY)    
    ptms = ptm_res.split(",")
    ptm_names = list()
    for ptm in ptms:
        ptm_name = re.sub(pattern="^[ 0-9]*", repl="", string=ptm)
        ptm_names.append(ptm_name)

    return ptm_names


def encode_mod_peptide(i_pos, aa_list, mod_aa_encode=None):
    if mod_aa_encode is None:
        mod_aa_encode = {"M": 1, "S": 2, "T": 3, "Y": 4}
    for i in i_pos:
        aa_list[i] = mod_aa_encode[aa_list[i]]
    return aa_list


def get_experiment_name(name):
    # 13CPTAC_LSCC_P_BI_20190806_BD_f05
    exp_name = re.sub(pattern=r'^(\d+).*$', repl=r'\1', string=name)
    return exp_name


def get_fraction_name(name):
    # 13CPTAC_LSCC_P_BI_20190806_BD_f05
    fraction_name = re.sub(pattern=r'^.*_(f[0-9A-Z]*)', repl=r'\1', string=name)
    return fraction_name


def format_mq_evidence(mq_evidence_file, out_file="evidence_format.tsv",
                       target_mod="Acetyl (K)",
                       remove_mods=None,
                       score_cutoff=0):
    if remove_mods is None:
        remove_mods = ['Acetyl (Protein N-term)', 'Dimethyl (KR)', 'Trimethyl (K)']

    psm_data = pd.read_csv(mq_evidence_file, low_memory=False, sep="\t")
    op = open(out_file, "w")
    # column x is the input for autort
    column_names = ["peptide", "x", "mq_modification", "row_mod_probs",
                    "protein", "rt", "cat_rt", "cat_rt_start", "cat_rt_end", "PEP", "score", "delta_score",
                    "raw_file", "experiment", "fraction",
                    "decoy", "contaminant", "is_ptm"]
    op.write("\t".join(column_names) + "\n")

    n_decoy_or_cont = 0
    n_low_confident_mod_pep = 0

    n_not_included_ptm = 0
    n_low_score = 0

    for i, row in tqdm(psm_data.iterrows(), total=psm_data.shape[0]):

        pep = row['Sequence']
        ptm_res = row['Modifications']
        pro = row['Proteins']
        rt = row['Retention time']
        cat_rt = row['Calibrated retention time']
        cat_rt_start = row['Calibrated retention time start']
        cat_rt_end = row['Calibrated retention time finish']
        score = row['Score']
        delta_score = row['Delta score']
        raw_file = row['Raw file']
        reverse = str(row['Reverse'])
        contaminant = str(row['Potential contaminant'])
        post_e_p = row['PEP']

        is_low_confident = False

        if score < score_cutoff:
            n_low_score = n_low_score + 1
            is_low_confident = True
            continue

        if any(ele in ptm_res for ele in remove_mods):
            n_not_included_ptm = n_not_included_ptm + 1
            continue

        if "+" in reverse or "+" in contaminant:
            n_decoy_or_cont = n_decoy_or_cont + 1
            continue

        is_ptm = 0

        ptm_names = get_ptm_name(ptm_res)
        is_confident_mod_pep = False
        om = None
        row_mod_probs = "-"
        # always consider this variable modification
        if "Oxidation (M)" in ptm_res:
            om, om_prob = get_confident_mod_pos(ptm_name="Oxidation (M)", mod_pep=row['Oxidation (M) Probabilities'],
                                                ptm_res=ptm_res, site_prob_cutoff=0.75)
            row_mod_probs = str(row['Oxidation (M) Probabilities'])

            if om is None or len(om) <= 0:
                n_low_confident_mod_pep = n_low_confident_mod_pep + 1
                is_low_confident = True
                continue

        ps = None
        if target_mod in ptm_res:
            ps, ps_prob = get_confident_mod_pos(ptm_name=target_mod, mod_pep=row[target_mod + ' Probabilities'],
                                                ptm_res=ptm_res, site_prob_cutoff=0.75)
            if row_mod_probs != "-":
                row_mod_probs = row_mod_probs + ";" + str(row[target_mod + ' Probabilities'])
            else:
                row_mod_probs = str(row[target_mod + ' Probabilities'])

            if ps is None or len(ps) <= 0:
                n_low_confident_mod_pep = n_low_confident_mod_pep + 1
                is_low_confident = True
                continue

        # format peptides
        aas = np.array(list(pep), dtype=str)
        if om is not None and len(om) >= 1:
            # There is Oxidation (M)
            is_confident_mod_pep = True
            aas = encode_mod_peptide(om, aas, mod_aa_encode={"M": 1})

        if ps is not None and len(ps) >= 1:
            # There is target mod
            is_confident_mod_pep = True
            is_ptm = 1
            if "Phospho" in target_mod:
                aas = encode_mod_peptide(ps, aas, mod_aa_encode={"M": 1, "S": 2, "T": 3, "Y": 4})
            else:
                aas = encode_mod_peptide(ps, aas, mod_aa_encode={"M": 1, "K": 2})

        if is_confident_mod_pep:
            encoded_mod_pep = "".join(aas)
        else:
            encoded_mod_pep = pep

        fraction_name = raw_file
        exp_name = raw_file

        # for output
        out_list = [pep, encoded_mod_pep, ptm_res, row_mod_probs, pro, rt, cat_rt, cat_rt_start, cat_rt_end, post_e_p,
                    score, delta_score, raw_file, exp_name, fraction_name, reverse, contaminant, is_ptm]
        op.write("\t".join(map(str, out_list)) + "\n")

    op.close()

    print("Total decoy or contaminant PSMs: %d" % n_decoy_or_cont)
    print("Low confident modified peptides: %d" % n_low_confident_mod_pep)
    print("Low score peptides (< %f): %d" % (score_cutoff, n_low_score))



def format_mq_msms(mq_msms_file, out_file="evidence_format.tsv", target_mod="Acetyl (K)", remove_mods=None,
                   max_pep_len=60):
    if remove_mods is None:
        remove_mods = ['Acetyl (Protein N-term)', 'Dimethyl (KR)', 'Trimethyl (K)']

    psm_data = pd.read_csv(mq_msms_file, low_memory=False, sep="\t")
    op = open(out_file, "w")
    # column x is the input for autort
    column_names = ["index", "raw_file", "scan_number", "scan_index", "peptide", "format_mod_pep", "row_mod_probs",
                    "n_missed_cleavages", "ptm_res", "protein", "charge", "mz", "mass", "site_prob", "mass_error_ppm",
                    "mass_error_da", "rt", "score_pep", "score", "delta_score", "is_target", "is_ptm"]
    op.write("\t".join(column_names) + "\n")

    n_decoy_or_cont = 0
    n_low_confident_mod_pep = 0

    n_not_included_ptm = 0

    rm_by_lens = 0
    for i, row in tqdm(psm_data.iterrows(), total=psm_data.shape[0]):
        raw_file = row['Raw file']
        scan_number = row['Scan number']
        scan_index = row['Scan index']
        index = raw_file+":"+str(scan_number) + ":"+str(scan_index)
        pep = row['Sequence']
        if len(pep) > max_pep_len:
            rm_by_lens += 1
            continue
        n_missed_cleavages = row['Missed cleavages']
        ptm_res = row['Modifications']
        pro = row['Proteins']
        charge = row['Charge']
        mz = row['m/z']
        mass = row['Mass']
        mass_error_ppm = row['Mass error [ppm]']
        mass_error_da = row['Mass error [Da]']
        rt = row['Retention time']
        score_pep = row['PEP']
        score = row['Score']
        delta_score = row['Delta score']
        reverse = str(row['Reverse'])
        is_target = 1

        if any(ele in ptm_res for ele in remove_mods):
            n_not_included_ptm = n_not_included_ptm + 1
            continue

        if "+" in reverse:
            n_decoy_or_cont = n_decoy_or_cont + 1
            is_target = -1
            # print(is_target)

        is_ptm = 0
        ptm_names = get_ptm_name(ptm_res)
        is_confident_mod_pep = False
        om = None
        om_prob = None
        row_mod_probs = "-"
        # always consider this variable modification
        if "Oxidation (M)" in ptm_res:
            # don't filter with site prob
            om, om_prob = get_confident_mod_pos(ptm_name="Oxidation (M)", mod_pep=row['Oxidation (M) Probabilities'],
                                                ptm_res=ptm_res, site_prob_cutoff=-1)
            row_mod_probs = str(row['Oxidation (M) Probabilities'])

            if om is None or len(om) <= 0:
                print("Parser error: Oxidation:")
                print(row)
                sys.exit()

        ps = None
        ps_prob = None
        if target_mod in ptm_res:
            ps, ps_prob = get_confident_mod_pos(ptm_name=target_mod, mod_pep=row[target_mod + ' Probabilities'],
                                                ptm_res=ptm_res,
                                                site_prob_cutoff=-1)
            if row_mod_probs != "-":
                row_mod_probs = row_mod_probs + ";" + str(row[target_mod + ' Probabilities'])
            else:
                row_mod_probs = str(row[target_mod + ' Probabilities'])

            if ps is None or len(ps) <= 0:
                print("Parser error:%s" % ptm_res)
                print(target_mod)
                print(row[target_mod + ' Probabilities'])
                print(ptm_res)
                print(ps)
                sys.exit()

        # format peptides
        aas = np.array(list(pep), dtype=str)
        if om is not None and len(om) >= 1:
            # There is Oxidation (M)
            is_confident_mod_pep = True
            aas = encode_mod_peptide(om, aas, mod_aa_encode={"M": 1})

        if ps is not None and len(ps) >= 1:
            # There is target mod
            is_confident_mod_pep = True
            is_ptm = 1
            if "Phospho" in target_mod:
                aas = encode_mod_peptide(ps, aas, mod_aa_encode={"M": 1, "S": 2, "T": 3, "Y": 4})
            else:
                aas = encode_mod_peptide(ps, aas, mod_aa_encode={"M": 1, "K": 2})

        if is_confident_mod_pep:
            encoded_mod_pep = "".join(aas)
        else:
            encoded_mod_pep = pep

        if is_target == -1:
            pro = "XXX_protein"

        prob_str = "1"
        prob_list = list()
        if om_prob is not None:
            prob_list = prob_list + om_prob
        if ps_prob is not None:
            prob_list = prob_list + ps_prob
        
        if len(prob_list) >= 1:
            prob_str = ",".join(map(str, prob_list))
        # for output
        out_list = [index,raw_file, scan_number, scan_index, pep, encoded_mod_pep, row_mod_probs, n_missed_cleavages,
                    ptm_res, pro, charge, mz, mass, prob_str,
                    mass_error_ppm, mass_error_da, rt, score_pep, score, delta_score, is_target, is_ptm]
        op.write("\t".join(map(str, out_list)) + "\n")

    op.close()

    print("Total decoy or contaminant PSMs: %d" % n_decoy_or_cont)
    print("Low confident modified peptides: %d" % n_low_confident_mod_pep)
    print("Rows removed due to unsupported modifications:%d" % n_not_included_ptm)
    print("Rows removed due to peptide length greater than %d:%d" % (max_pep_len, rm_by_lens))


def isNaN(num):
    return num != num


def main():
    parser = argparse.ArgumentParser(description='Preprocess MaxQuant evidence file')
    parser.add_argument('-i', '--input_file', default=None, type=str,  # required=True,
                        help="MaxQuant evidence file")

    parser.add_argument('-o', '--out_file', default="format_evidence.txt", type=str,
                        help="Output file")
    parser.add_argument('-m', '--modification', default=None, type=str,
                        help="modification. For example, 'Acetyl (K)'")
    parser.add_argument('-f', '--format', default=1, type=int,
                        help="1:evidence.txt or 2:msms.txt")

    args = parser.parse_args(sys.argv[1:])

    input_file = args.input_file
    out_file = args.out_file
    mods = args.modification
    n_format = args.format
    if n_format == 1:
        format_mq_evidence(input_file, out_file=out_file, target_mod=mods)
    else:
        format_mq_msms(input_file, out_file=out_file, target_mod=mods)


if __name__ == "__main__":
    main()
