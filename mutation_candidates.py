import binding_score as ss
from Bio import SeqIO
import pandas as pd
import numpy as np
import subprocess
import time
import sys
import os


def split_fasta_seqs(fasta_file):
    fasta_dir = "/all_sequences"  # in directory where is fasta file, we create fastadir with all fastas.
    data_dir, data_file = os.path.split(fasta_file)
    fasta_dir = data_dir+fasta_dir

    subprocess.run(['mkdir', fasta_dir])
    sequence_paths = []
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        sequence_path = fasta_dir + "/sequence" + str(i) + ".fasta"
        sequence_paths.append(sequence_path)
        output_handle = open(sequence_path, "w")
        SeqIO.write(record, output_handle, "fasta")
        output_handle.close()
    return sequence_paths, fasta_dir


def remove_sequences(sequence_paths, fasta_dir, mutations_file):
    for sequence_path in sequence_paths:
        subprocess.run(['rm', sequence_path])
    subprocess.run(['rm', mutations_file])
    subprocess.run(['rmdir', fasta_dir])


def make_mutated_seqs(fasta_file, mutations_file):
    original_seq = str(SeqIO.read(fasta_file, "fasta").seq)
    nucleotids = ['A', 'T', 'G', 'C']
    mutations = []
    for i in range(len(original_seq)):
        for nucleotide in nucleotids:
            mutated_seq = original_seq
            mutated_seq = mutated_seq[:i] + nucleotide + mutated_seq[i+1:]
            mutations.append(mutated_seq)
    mutations_df = pd.DataFrame({"mutations": mutations})
    mutations_df.to_csv(mutations_file, index=False, header=False)


def get_original_score(fasta_file, features_ids, deepbind_path, num_cpu, max_seq_len):
    promoter_seq, promoter_ids, results_txt = ss.get_binding_score(fasta_file, features_ids, deepbind_path,
                                                                   num_cpu, max_seq_len)
    score_df = pd.read_csv(results_txt, delimiter="\t")
    id_df = pd.read_csv(promoter_ids, delimiter="\t")
    return score_df.iat[0, 0], list(id_df)[0], promoter_seq, promoter_ids


def get_score_changes(mutations_file, features_ids, deepbind_path, num_cpu, original_score, sequence_id,
                      final_results_file, results_txt):
    output_handle = open(final_results_file, "a")
    output_handle.write('\n\n' + sequence_id + '\n')
    output_handle.close()

    ss.deep_bind_exec_parallel(features_ids, mutations_file, results_txt, deepbind_path, num_cpu)

    mutated_scores_df = pd.read_csv(results_txt, delimiter="\t")
    mutated_scores = mutated_scores_df.as_matrix()
    mutated_scores = mutated_scores.reshape((4, int(mutated_scores.shape[0]/4)), order='F')

    score_changes = np.multiply((mutated_scores - original_score), np.maximum(0, original_score, mutated_scores))
    score_changes_df = pd.DataFrame(data=score_changes, index=np.arange(4)).round(5)
    score_changes_df.to_csv(final_results_file, mode='a')
    print("Program has successfully written score changes at " + final_results_file + ".")


def main():
    start = time.time()
    arguments = sys.argv[1:]
    num_cpu = 4  # default value
    max_seq_len = 1000

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python mutation_candidates.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>, "
              "where <num_of_cpu> is optional (default is 4).")
        return

    fasta_file = arguments[0]
    features_ids = arguments[1]
    deepbind_path = arguments[2]
    final_results_file = arguments[3]
    if len(arguments) > 4:
        num_cpu = int(arguments[4])
    print("Program running on " + str(num_cpu) + " CPU cores.")
    results_txt = "results.txt"
    mutations_file = 'mutations.seq'
    output_handle = open(final_results_file, "w")
    output_handle.write('')
    output_handle.close()

    sequence_paths, fasta_dir = split_fasta_seqs(fasta_file)
    for sequence_path in sequence_paths:
        original_score, sequence_id, promoter_seq, promoter_ids = get_original_score(sequence_path, features_ids,
                                                                                     deepbind_path, num_cpu,
                                                                                     max_seq_len)
        print("Original score of " + sequence_id + ":", original_score)
        make_mutated_seqs(sequence_path, mutations_file)
        get_score_changes(mutations_file, features_ids, deepbind_path, num_cpu, original_score, sequence_id,
                          final_results_file, results_txt)

    ss.remove_temp_files(promoter_seq, promoter_ids, results_txt)
    remove_sequences(sequence_paths, fasta_dir, mutations_file)
    end = time.time()-start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
