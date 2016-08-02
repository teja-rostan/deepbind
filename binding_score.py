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
    the program supports parallelization on multiple CPU cores (num_of_cpu).


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


def get_seq_and_id(fasta_file, sequence_path, id_path, threshold):
    """ Extracts raw sequence strings and ids to separate files."""

    sequences = []
    record_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        if len(str(record.seq)) <= threshold:
            sequences.append(str(record.seq))
            record_ids.append(str(record.id))
    data_record_ids = pd.DataFrame({"record_id": record_ids})
    data_sequences = pd.DataFrame({"record_sequence": sequences})
    data_record_ids.to_csv(id_path, index=False, header=False)
    data_sequences.to_csv(sequence_path, index=False, header=False)


def deep_bind_exec(features_ids, promoter_seq, results_txt, deepbind_path):
    """ Executes the deepbind program: deepbind features_ids < promoter_seq > results_file """

    promoter_seq_file = open(promoter_seq, "r")
    results_file = open(results_txt, "w")
    subprocess.run([deepbind_path, features_ids], stdin=promoter_seq_file, stdout=results_file)


def deep_bind_exec_parallel(features_ids, promoter_seq, results_txt, deepbind_path, p):
    """ Executes the deepbind program in parallel with starmap. """

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
    zipped = zip(feature_list, promoter_seq_list, result_paths, deepbind_paths)
    p.starmap(deep_bind_exec, zipped)
    join_results(result_paths, results_txt)


def get_binding_score(fasta_file, features_ids, deepbind_path, p, max_seq_len):
    """ Prepares and calls deep_bind_exec_parallel. """

    promoter_seq = "promoter.seq"
    promoter_ids = "promoter.ids"
    results_txt = "results.txt"

    get_seq_and_id(fasta_file, promoter_seq, promoter_ids, max_seq_len)
    deep_bind_exec_parallel(features_ids, promoter_seq, results_txt, deepbind_path, p)
    return promoter_seq, promoter_ids, results_txt


def join_results(result_paths, results_txt):
    """ Concatenates id of sequence with scores."""

    frames = []
    for result_path in result_paths:
        frames.append(pd.read_csv(result_path))
        subprocess.run(['rm', result_path])
    results = pd.concat(frames, axis=1)
    results.to_csv(results_txt, sep='\t', index=None)


def write_scores(promoter_ids, results_txt, final_results_file):
    """ Writes joined results to "final_results_file"."""

    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_txt, delimiter="\t")
    promoter_scores = pd.concat([df1, df2], axis=1)
    promoter_scores = promoter_scores.rename(columns={0: 'ID'})
    promoter_scores.to_csv(final_results_file, sep='\t')


def write_ranked_scores(results_txt, promoter_ids, final_results_file):
    """ Computes ranks of biggest scores for every feature (TF)."""

    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_txt, delimiter="\t")
    ranks = df1.as_matrix()[np.argsort(df2.as_matrix(), axis=0)[::-1]][:, :, 0]
    data_dir, data_file = os.path.split(final_results_file)
    data_path = data_dir + "/ranked_list.csv"
    ranks_handle = open(data_path, "w")
    print(ranks.shape)
    df = pd.DataFrame(data=ranks)
    df.to_csv(data_path, sep='\t', index=None, header=list(df2))

    print("Program has successfully written rank lists at " + data_path + ".")
    ranks_handle.close()


def remove_temp_files(promoter_seq, promoter_ids, results_txt):
    """ Removes all temp files."""

    subprocess.run(['rm', promoter_seq])
    subprocess.run(['rm', promoter_ids])
    subprocess.run(['rm', results_txt])


def main():
    start = time.time()
    arguments = sys.argv[1:]
    num_cpu = 4  # default value
    max_seq_len = 1000

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
    promoter_seq, promoter_ids, results_txt = get_binding_score(fasta_file, features_ids, deepbind_path, p, max_seq_len)
    write_scores(promoter_ids, results_txt, final_results_file)
    write_ranked_scores(results_txt, promoter_ids, final_results_file)
    remove_temp_files(promoter_seq, promoter_ids, results_txt)

    end = time.time() - start
    print("Program has successfully written scores at " + final_results_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
