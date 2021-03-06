from Bio import SeqIO
import pandas as pd
import numpy as np
import subprocess
import multiprocessing as mp
import sys
import os
import time

help = \
    """
    Score DNA/RNA sequences stored in FASTA format (fasta_path) according to any RBP/TF
    model (features_path) listed in the DeepBind web repository:
    http://tools.genes.toronto.edu/deepbind
    For scoring, the program uses deepbind executable (deepbind_path) that can be downloaded at:
    http://tools.genes.toronto.edu/deepbind/download.html
    Scores are written in a file (results_path). To speed up computation,
    the program supports parallelism on multiple CPU cores (num_of_cpu).


    Usage:
        python binding_score.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>,
        where <num_of_cpu> is optional (default is 4).

    Example:
        python binding_score.py promoter_sequences/promoter_sequences.fasta
        goal1/features.ids ./deepbind goal1/promoter_scores.csv 12,
        where deepbind executable is in same dir.
"""


def num_of_seq_and_leq(fasta_file, threshold):
    """ Prints number of sequences in fasta file and number of sequences less or equal of 1000 nucleotides."""

    num_seq = len(list(SeqIO.parse(fasta_file, "fasta")))
    print("Number of sequences:", num_seq)
    less = np.sum([1 for record in SeqIO.parse(fasta_file, "fasta") if len(record.seq) <= threshold])
    print("Number of sequences less or equal " + str(threshold) + ":", less)


def get_seq_and_id(fasta_file, promoter_seq, promoter_ids, threshold):
    """ Extracts raw sequence strings and ids to separate files."""

    sequences = []
    record_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq)[-threshold:])
        record_id = str(record.id)
        end = record_id.find('|')
        if end != -1:
            record_id = record_id[:end]
        record_ids.append(record_id)
    data_record_ids = pd.DataFrame({"record_id": record_ids})
    data_sequences = pd.DataFrame({"record_sequence": sequences})
    data_record_ids.to_csv(promoter_ids, index=False, header=False)
    data_sequences.to_csv(promoter_seq, index=False, header=False)


def deep_bind_exec(features_ids, promoter_seq, results_txt, deepbind_path):
    """ Executes the deepbind program: deepbind features_ids < promoter_seq > results_file """

    promoter_seq_file = open(promoter_seq, "r")
    results_file = open(results_txt, "w")
    subprocess.run([deepbind_path, "--all-scores", features_ids], stdin=promoter_seq_file, stdout=results_file)
    promoter_seq_file.close()
    results_file.close()


def deep_bind_exec_parallel(features_ids, promoter_seq, deepbind_path, p, promoter_ids, seq_len, final_results_file):
    """ Executes the deepbind program in parallel with starmap. Calls join_results."""

    result_paths = []
    feature_list = []
    promoter_seq_list = []
    deepbind_paths = []
    features = open(features_ids, "r")
    for i, feature in enumerate(features):
        result_paths.append('result' + str(i) + '.txt')
        feature_list.append(feature)
        promoter_seq_list.append(promoter_seq)
        deepbind_paths.append(deepbind_path)
    features.close()
    zipped = zip(feature_list, promoter_seq_list, result_paths, deepbind_paths)
    p.starmap(deep_bind_exec, zipped)
    write_results(result_paths, promoter_ids, seq_len, final_results_file)


def get_binding_score(fasta_file, features_ids, deepbind_path, p, seq_len, final_results_file):
    """ Prepares and calls deep_bind_exec_parallel. """

    data_dir, _ = os.path.split(fasta_file)
    data_dir2, _ = os.path.split(final_results_file)
    promoter_seq = data_dir + "/promoter_seq_with_exp.tab"
    promoter_ids = data_dir + "/promoter_id_with_exp.tab"
    deep_bind_exec_parallel(features_ids, promoter_seq, deepbind_path, p, promoter_ids, seq_len, final_results_file)


def write_results(result_paths, promoter_ids, seq_len, final_results_file):
    """ Concatenates id of sequence with scores."""

    def remove_last_and_float(x):
        x = x[:-1]
        return list(map(float, x))

    df1 = pd.read_csv(promoter_ids, header=None)
    data_dir, _ = os.path.split(final_results_file)
    for result_path in result_paths:
        col = pd.read_csv(result_path, squeeze=True)
        print(col)
        col_name = col.name
        print(col_name)
        col = col.str.split('\t')
        col = col.apply(remove_last_and_float)
        new_col = []
        for index, row in col.iteritems():
            rowl = len(row)
            print(rowl)
            missing = seq_len - 23 - rowl
            print(missing)
            nan_list = [np.nan] * missing
            row = nan_list + row
            new_col.append(row)
        col = pd.DataFrame(new_col, index=None)
        print(col)
        promoter_scores = pd.concat([df1, col], axis=1, ignore_index=True)
        promoter_scores = promoter_scores.round(decimals=2)
        promoter_scores = promoter_scores.fillna('?')
        new_col_names = ['ID'] + list(np.arange(len(list(promoter_scores.columns.values)) - 1).tolist())
        promoter_scores.to_csv(data_dir + "/" + col_name + ".csv", sep='\t', index=None, header=new_col_names)
        subprocess.run(['rm', result_path])


def write_ranked_scores(results_txt, promoter_ids, final_results_file):
    """ Computes and writes ranks of biggest scores for every feature (TF)."""

    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_txt, delimiter="\t")
    ranks = df1.as_matrix()[np.argsort(df2.as_matrix(), axis=0)[::-1]][:, :, 0]
    data_dir, data_file = os.path.split(final_results_file)
    data_path = data_dir + "/ranked_scores.csv"
    ranks_handle = open(data_path, "w")
    df = pd.DataFrame(data=ranks)
    df.to_csv(data_path, sep='\t', index=None, header=list(df2))
    print("Program has successfully written rank lists at " + data_path + ".")
    ranks_handle.close()


def write_scores_modified(promoter_ids, results_txt, final_results_file, features_ids):
    """ Writes bindings scores and sequence ids transposed to "final_results_file"."""

    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_txt, delimiter="\t")
    df3 = pd.read_csv(features_ids, header=None, delimiter=" ")
    df2 = df2.transpose()
    df2.reset_index(level=0, inplace=True)
    df2['index'] = df3[1]
    names = df1.as_matrix().T.tolist()[0]
    names.insert(0, 'ID')
    df2.to_csv(final_results_file, sep='\t', index=None, header=names)


def remove_temp_files(results_txt):
    """ Removes all temp files."""

    subprocess.run(['rm', results_txt])


def main():
    start = time.time()
    arguments = sys.argv[1:]
    num_cpu = 4  # default value
    max_seq_len = 900

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python binding_score.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>, "
              "where <num_of_cpu> is optional (default is 4).")
        return

    fasta_file = arguments[0]
    features_ids = arguments[1]
    deepbind_path = arguments[2]
    final_results_file = arguments[3]
    if len(arguments) > 4:
        num_cpu = int(arguments[4])
    print("Program running on " + str(num_cpu) + " CPU cores.")

    p = mp.Pool(num_cpu)
    get_binding_score(fasta_file, features_ids, deepbind_path, p, max_seq_len, final_results_file)

    end = time.time() - start
    print("Program has successfully written scores at " + final_results_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
