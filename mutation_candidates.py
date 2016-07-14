import binding_score as ss
from Bio import SeqIO
import pandas as pd
import subprocess
import time
import sys
import os


def split_fasta_seqs(fasta_file):
    fasta_dir = "/all_sequences" # in directory where is fasta file, we create fastadir with all fastas.
    data_dir, data_file = os.path.split(fasta_file)
    fasta_dir = data_dir+fasta_dir

    subprocess.run(['mkdir', fasta_dir])
    sequence_paths = []
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        sequence_path = fasta_dir+"/sequence"+str(i)+".fasta"
        sequence_paths.append(sequence_path)
        output_handle = open(sequence_path, "w")
        SeqIO.write(record, output_handle, "fasta")
        output_handle.close()
    return sequence_paths, fasta_dir


def remove_sequences(sequence_paths, fasta_dir):
    for sequence_path in sequence_paths:
        subprocess.run(['rm', sequence_path])
    subprocess.run(['rmdir', fasta_dir])


def make_mutation_seqs(fasta_file):
    original_seq = SeqIO.read(fasta_file, "fasta")
    #  create fasta full with mutations


def get_original_score(fasta_file, features_ids, deepbind_path, final_results_file, num_cpu, max_seq_len):
    promoter_seq, promoter_ids, results_txt = ss.get_binding_score(fasta_file, features_ids, deepbind_path,
                                                                   final_results_file, num_cpu, max_seq_len)
    score_df = pd.read_csv(results_txt, delimiter="\t")
    ss.remove_temp_files(promoter_seq, promoter_ids, results_txt)
    return score_df.iat[0, 0]


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

    sequence_paths, fasta_dir = split_fasta_seqs(fasta_file)
    original_score = get_original_score(sequence_paths[0], features_ids, deepbind_path, final_results_file, num_cpu,
                                        max_seq_len)

    print("Original score:", original_score)
    # mutations = make_mutation_seqs(sequence_paths[0])
    remove_sequences(sequence_paths, fasta_dir)

    end = time.time()-start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()