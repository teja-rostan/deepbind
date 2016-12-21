import time
import sys
from Bio import SeqIO
import pandas as pd
from NNET import get_data_target


def get_seq_and_id(fasta_file, promoter_seq, promoter_ids, threshold, scores_file, delimiter):
    """ Extracts raw sequence strings and ids to separate files."""

    map_txt = "/Users/tejarostan/PycharmProjects/deepBind/DEEPBIND/DDB_DDB_G/DDB-GeneID-UniProt.txt"
    df = pd.read_csv(map_txt, sep="\t")
    ddb_id = list(df['DDBDDB ID'].as_matrix())
    ddb_g_id = list(df['DDB_G ID'].as_matrix())

    all_valid_records = get_data_target.get_ids(scores_file, delimiter, 'ID')
    print(all_valid_records)
    sequences = []
    record_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        record_id = str(record.id)
        end = record_id.find('|')
        record_id_short = record_id
        if end != -1:
            record_id_short = record_id[:end]
        print(record_id_short)
        try:
            ddbg_record_id_short = ddb_g_id[ddb_id.index(record_id_short)]
        except ValueError:
            ddbg_record_id_short = record_id_short
        if ddbg_record_id_short in all_valid_records:
            print(record_id_short + "  NOOOOOOOOT")
            record_ids.append(ddbg_record_id_short)
            seq = str(record.seq)[-threshold:]
            sequences.append(seq)
    data_record_ids = pd.DataFrame({"record_id": record_ids})
    data_sequences = pd.DataFrame({"record_sequence": sequences})
    data_record_ids.to_csv(promoter_ids, index=False, header=False)
    data_sequences.to_csv(promoter_seq, index=False, header=False)


def main():
    start = time.time()
    arguments = sys.argv[1:]
    max_seq_len = 900

    if len(arguments) < 5:
        print("Not enough arguments stated! Usage: \n"
              "python get_seq_with_exp.py <input_fasta> <output_seq> <output_id> <reduced_table_with_id> <delimiter>.")
        return

    fasta_file = arguments[0]
    promoter_seq = arguments[1]
    promoter_ids = arguments[2]
    scores_file = arguments[3]
    delimiter = arguments[4]

    get_seq_and_id(fasta_file, promoter_seq, promoter_ids, max_seq_len, scores_file, delimiter)
    end = time.time() - start
    print("Program has successfully written scores at " + promoter_seq + " and " + promoter_ids + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
