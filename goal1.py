from Bio import SeqIO, Entrez
import pandas as pd
import numpy as np
import subprocess


def ncbi_read(id):
    Entrez.email = "teja.rostan@gmail.com"
    handle = Entrez.efetch(db="nucleotide", id=id, rettype="gb")
    return SeqIO.read(handle, "genbank")


def seqio_read(filename):
    return SeqIO.read(open(filename, "r"), "fasta")


def get_seq_and_ID(dataname, sequence_path, id_path, threshold):
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


def get_leq(dataname, threshold):  # get the number of sequences that has threshold value and less nucleotides
    less = np.sum([1 for record in SeqIO.parse(dataname, "fasta") if len(record.seq) <= threshold])
    print("Number of sequences less or equal " + str(threshold) + ":", less)


def create_ranked_records(promoter_ids, results_file):
    df1 = pd.read_csv(promoter_ids, header=None)
    df2 = pd.read_csv(results_file, delimiter="\t")
    frames = [df1, df2]
    promoter_scores = pd.concat(frames, axis=1)
    promoter_scores = promoter_scores.rename(columns={0: 'promoterID'})
    promoter_scores.to_csv("goal1/promoter_scores.csv", sep='\t')


def deep_bind_exec(features_ids, promoter_seq, results_file):
    # subprocess.run(args)  # % deepbind features_ids < promoter_seq > results_file
    input = open(promoter_seq, "r")
    output = open(results_file, "w")
    print('subprocess deepbind starting...')
    subprocess.run(["./deepbind", features_ids], stdin=input, stdout=output)
    print('...subprocess deepbind finnished')


def main():
    dataname = "promoter_sequences/promoter_sequences.fasta"
    num_of_seq(dataname)
    get_leq(dataname, 50)

    promoter_seq = "goal1/promoter.seq"
    promoter_ids = "goal1/promoter.ids"
    get_seq_and_ID(dataname, promoter_seq, promoter_ids, 50)

    features_ids = "goal1/features.ids"
    results_file = "goal1/results.txt"
    deep_bind_exec(features_ids, promoter_seq, results_file)

    create_ranked_records(promoter_ids, results_file)


if __name__ == '__main__':
    main()
