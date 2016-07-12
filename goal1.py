from Bio import SeqIO, Entrez
import pandas as pd
import numpy as np
import subprocess
import multiprocessing as mp


def ncbi_read(id):
    Entrez.email = "teja.rostan@gmail.com"
    handle = Entrez.efetch(db="nucleotide", id=id, rettype="gb")
    return SeqIO.read(handle, "genbank")


def seqio_read(filename):
    return SeqIO.read(open(filename, "r"), "fasta")


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


def create_ranked_records(promoter_ids, results_txt):
    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_txt, delimiter="\t")
    promoter_scores = pd.concat([df1, df2], axis=1)
    promoter_scores = promoter_scores.rename(columns={0: 'promoterID'})
    promoter_scores.to_csv("goal1/promoter_scores.csv", sep='\t')


def deep_bind_exec(features_ids, promoter_seq, results_txt):
    # % deepbind features_ids < promoter_seq > results_file
    promoter_seq_file = open(promoter_seq, "r")
    results_file = open(results_txt, "w")
    # print("Starting deepbind subprocess...")
    subprocess.run(["./deepbind", features_ids], stdin=promoter_seq_file, stdout=results_file)
    # print("...deepbind subprocess ended.")


def join_results(result_paths):
    frames = []
    for result_path in result_paths:
        frames.append(pd.read_csv(result_path))
        subprocess.run(['rm', result_path])
    results = pd.concat(frames, axis=1)
    results.to_csv("goal1/results.txt", sep='\t', index=None)


def deep_bind_exec_parallel(features_ids, promoter_seq):
    N = 4  # our titan computer has 4 cores
    p = mp.Pool(N)
    result_paths = []
    feature_list =[]
    promoter_seq_list = []
    features = open(features_ids, "r")
    for i, feature in enumerate(features):
        result_paths.append('goal1/result' + str(i) + '.txt')
        feature_list.append(feature)
        promoter_seq_list.append(promoter_seq)
    zipped = zip(feature_list, promoter_seq_list, result_paths)
    p.starmap(deep_bind_exec, zipped)
    join_results(result_paths)


def main():
    dataname = "promoter_sequences/promoter_sequences.fasta"
    num_of_seq(dataname)
    get_leq(dataname, 50)

    promoter_seq = "goal1/promoter.seq"
    promoter_ids = "goal1/promoter.ids"
    get_seq_and_id(dataname, promoter_seq, promoter_ids, 50)

    features_ids = "goal1/features.ids"
    results_txt = "goal1/results.txt"
    # deep_bind_exec(features_ids, promoter_seq, results_txt)
    deep_bind_exec_parallel(features_ids, promoter_seq)

    create_ranked_records(promoter_ids, results_txt)


if __name__ == '__main__':
    main()
