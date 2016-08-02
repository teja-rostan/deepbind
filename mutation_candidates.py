import binding_score as ss
from Bio import SeqIO
import pandas as pd
import numpy as np
import subprocess
import multiprocessing as mp
import time
import sys
import os


def split_fasta_seqs(fasta_file):
    """ In the same directory where is found a fasta file, it creates a directory "all_sequences" with every fasta
    sequence in separate file."""

    fasta_dir = "/all_sequences"
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


def remove_sequences(sequence_paths, mutations_file, subseqs_file):
    """ Removes all temp files of fasta sequences."""

    for sequence_path in sequence_paths:
        subprocess.run(['rm', sequence_path])
    subprocess.run(['rm', mutations_file])
    subprocess.run(['rm', subseqs_file])


def make_mutated_seqs(fasta_file, mutations_file):
    """ Creates file "mutations_file of all possible point mutations (SNP)."""

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
    return mutations


def get_original_score(fasta_file, features_ids, deepbind_path, p, max_seq_len):
    """ Extracts and returns binding score of original sequence."""

    promoter_seq, promoter_ids, results_txt = ss.get_binding_score(fasta_file, features_ids, deepbind_path,
                                                                   p, max_seq_len)
    score_df = pd.read_csv(results_txt, delimiter="\t")
    id_df = pd.read_csv(promoter_ids, delimiter="\t")
    return score_df.iat[0, 0], list(id_df)[0], promoter_seq, promoter_ids


def write_score_changes(score_changes, final_results_file, sequence_id):
    """ Writes score changes into a file "final_results_file"."""

    output_handle = open(final_results_file, "a")
    output_handle.write('\n\n' + sequence_id + '\n')
    output_handle.close()

    score_changes_df = pd.DataFrame(data=score_changes, index=np.arange(4)).round(4)
    score_changes_df.to_csv(final_results_file, mode='a')
    print("Program has successfully written score changes at " + final_results_file + ".")


def get_score_changes(mutations_file, features_ids, deepbind_path, p, original_score, results_txt):
    """ Computes scores of mutated sequences and the final sensitivity (Creates sensitivity map)"""

    ss.deep_bind_exec_parallel(features_ids, mutations_file, results_txt, deepbind_path, p)

    mutated_scores_df = pd.read_csv(results_txt, delimiter="\t")
    mutated_scores = mutated_scores_df.as_matrix()
    mutated_scores = mutated_scores.reshape((4, int(mutated_scores.shape[0]/4)), order='F')

    score_changes = np.multiply((mutated_scores - original_score), np.maximum(0, original_score, mutated_scores))
    return score_changes


def get_score_changes_advanced(mutations, features_ids, deepbind_path, p, original_score, results_txt, subseqs_file,
                               motif_len):
    """ Get sensitivity map for long sequences and with several binding sites that each need to be
    recognized independently."""

    mutations_np = np.array([list(mutation) for mutation in mutations])

    final_changed_scores = np.zeros((4, mutations_np.shape[1] - 1))
    repetitions = np.zeros((4, mutations_np.shape[1] - 1))
    mask = np.ones((4, motif_len))

    for i in range(mutations_np.shape[1] - motif_len):
        subseqs = mutations_np[:, i:i+motif_len][:4 * motif_len]
        subseqs_list = []
        for j in range(len(subseqs)):
            subseqs_list.append(''.join(subseqs[j]))
        df = pd.DataFrame(data=subseqs_list)
        df.to_csv(subseqs_file, sep='\t', index=None, header=None)

        score_changes = get_score_changes(subseqs_file, features_ids, deepbind_path, p, original_score, results_txt)
        repetitions[:, i:i+motif_len] += mask
        final_changed_scores[:, i:i+motif_len] += score_changes
    final_changed_scores /= repetitions
    return final_changed_scores


def get_ranked_scores(ranked_scores, score_changes):
    """ Computes ranks of biggest changes."""

    ranks = []
    for score_change in score_changes:
        rank = np.argsort(abs(score_change))[::-1]
        ranks.append(rank[:np.count_nonzero(score_change)])
    ranked_scores.append(ranks)
    return ranked_scores


def write_ranked_list(ranked_scores, final_results_file, sequence_ids):
    """ Gets a list of ranks from "ranked_scores" with "sequence_ids" and writes them in to a file
    "final_results_file."""

    data_dir, data_file = os.path.split(final_results_file)
    data_path = data_dir + "/ranked_list.csv"
    ranks_handle = open(data_path, "w")
    for i in range(len(ranked_scores)):
        ranks_handle.write(sequence_ids[i] + "\n" +
                           '\n'.join(str('\t'.join(str(rank) for rank in ranks)) for ranks in ranked_scores[i]) +
                           "\n\n")
    print("Program has successfully written rank lists at " + data_path + ".")
    ranks_handle.close()


def main():
    """ Creates sensitivity (mutation) map for visualization of DeepBind models.
    First it computes the score of original sequence. Creates all possible point mutations (SNP). Computes score of
    every mutated sequence. Computes the final sensitivity from original score and all scores of mutated sequences.
    The program can handle long sequences and/or multiple binding sites."""
    start = time.time()
    arguments = sys.argv[1:]
    num_cpu = 4  # default value
    motif_len = 10  # default value
    max_seq_len = 1000

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python mutation_candidates.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu=4>"
              "<motif_len=10>, where <num_of_cpu=4> and <motif_len=10> are optional.")
        return

    fasta_file = arguments[0]
    features_ids = arguments[1]
    deepbind_path = arguments[2]
    final_results_file = arguments[3]
    if len(arguments) > 4:
        num_cpu = int(arguments[4])
    if len(arguments) > 5:
        motif_len = int(arguments[5])
    output_handle = open(final_results_file, "w")
    output_handle.write('')
    output_handle.close()
    results_txt = "results.txt"
    mutations_file = 'mutations.seq'
    subseqs_file = 'subsequences.seq'
    ranked_scores = []
    sequence_ids = []
    print("Program running on " + str(num_cpu) + " CPU cores.")

    p = mp.Pool(num_cpu)
    sequence_paths, fasta_dir = split_fasta_seqs(fasta_file)
    for sequence_path in sequence_paths:
        original_score, sequence_id, promoter_seq, promoter_ids = get_original_score(sequence_path, features_ids,
                                                                                     deepbind_path, p, max_seq_len)
        print("Original score of " + sequence_id + ":", original_score)
        sequence_ids.append(sequence_id)

        mutations = make_mutated_seqs(sequence_path, mutations_file)

        score_changes = get_score_changes_advanced(mutations, features_ids, deepbind_path, p, original_score,
                                                   results_txt, subseqs_file, motif_len)

        ranked_scores = get_ranked_scores(ranked_scores, score_changes)
        write_score_changes(score_changes, final_results_file, sequence_id)

    remove_sequences(sequence_paths, mutations_file, subseqs_file)
    write_ranked_list(ranked_scores, final_results_file, sequence_ids)
    ss.remove_temp_files(promoter_seq, promoter_ids, results_txt)
    subprocess.run(['rmdir', fasta_dir])
    end = time.time()-start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
