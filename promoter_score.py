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
        python promoter_score.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>,
        where <num_of_cpu> is optional (default is 4).

    Example:
        python promoter_score.py promoter_sequences/promoter_sequences.fasta
        goal1/features.ids deepbind goal1/promoter_scores.csv 12,
        where deepbind executable is in same dir.
"""

from Bio import SeqIO
import pandas as pd
import numpy as np
import subprocess
import multiprocessing as mp
import sys
import time


def get_seq_and_id(dataname, sequence_path, id_path, threshold):
    sequences = []
    record_ids = []
    for record in SeqIO.parse(dataname, "fasta"):
        if len(str(record.seq)) <= threshold:
            sequences.append(str(record.seq))
            record_ids.append(str(record.id))
    data_record_ids = pd.DataFrame({"record_id": record_ids})
    data_sequences = pd.DataFrame({"record_sequence": sequences})
    data_record_ids.to_csv(id_path, index=False, header=False)
    data_sequences.to_csv(sequence_path, index=False, header=False)


def num_of_seq(dataname):
    num_seq = len(list(SeqIO.parse(dataname, "fasta")))
    print("Number of sequences:", num_seq)


def get_leq(dataname, threshold):
    less = np.sum([1 for record in SeqIO.parse(dataname, "fasta") if len(record.seq) <= threshold])
    print("Number of sequences less or equal " + str(threshold) + ":", less)


def create_ranked_records(promoter_ids, results_txt, final_results_file):
    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_txt, delimiter="\t")
    promoter_scores = pd.concat([df1, df2], axis=1)
    promoter_scores = promoter_scores.rename(columns={0: 'promoterID'})
    promoter_scores.to_csv(final_results_file, sep='\t')


def deep_bind_exec(features_ids, promoter_seq, results_txt, deepbind_path):
    # % deepbind features_ids < promoter_seq > results_file
    promoter_seq_file = open(promoter_seq, "r")
    results_file = open(results_txt, "w")
    # print("Starting deepbind subprocess...")
    subprocess.run(["./"+deepbind_path, features_ids], stdin=promoter_seq_file, stdout=results_file)
    # print("...deepbind subprocess ended.")


def join_results(result_paths, results_txt):
    frames = []
    for result_path in result_paths:
        frames.append(pd.read_csv(result_path))
        subprocess.run(['rm', result_path])
    results = pd.concat(frames, axis=1)
    results.to_csv(results_txt, sep='\t', index=None)


def deep_bind_exec_parallel(features_ids, promoter_seq, results_txt, deepbind_path, num_cpu):
    p = mp.Pool(num_cpu)
    result_paths = []
    feature_list =[]
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


def main():  # fasta.file features.ids results.csv num_of_cpu_cores
    start = time.time()
    arguments = sys.argv[1:]
    num_cpu = 4
    max_seq_len = 1000

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage example: \n"
              "python promoter_score.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>, "
              "where <num_of_cpu> is optional (default is 4).")
        return

    dataname = arguments[0]             # "promoter_sequences/promoter_sequences.fasta"
    features_ids = arguments[1]         # "goal1/features.ids"
    deepbind_path = arguments[2]        # "deepbind"
    final_results_file = arguments[3]   # "goal1/promoter_scores.csv"

    if len(arguments) > 4:
        num_cpu = int(arguments[4])  # 12

    print("Program running on " + str(num_cpu) + " CPU cores.")
    promoter_seq = "promoter.seq"
    promoter_ids = "promoter.ids"
    results_txt = "results.txt"

    num_of_seq(dataname)
    get_leq(dataname, max_seq_len)
    get_seq_and_id(dataname, promoter_seq, promoter_ids, max_seq_len)

    # deep_bind_exec(features_ids, promoter_seq, results_txt)
    deep_bind_exec_parallel(features_ids, promoter_seq, results_txt, deepbind_path, num_cpu)

    create_ranked_records(promoter_ids, results_txt, final_results_file)

    subprocess.run(['rm', promoter_seq])
    subprocess.run(['rm', promoter_ids])
    subprocess.run(['rm', results_txt])

    end = time.time()-start
    print("Program has successfully written scores at " + final_results_file + ".")
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
