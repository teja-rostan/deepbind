import pandas as pd
import numpy as np
import time
import sys
import os


help = \
    """
    Gene ID converter. Files use different ID groups for the same sequence. Possible groups are DDB ID and DDB_G ID.
    The program supports conversion from DDB to DDB_G (conv_type=0) or from DDB_G to DDB (conv_type=1).

    Usage:
        python map_columns.py <original_file> <column_name> <conversion_type> <converted_file> <delimiter>.

    Example:
        python map_columns.py original_file.csv ID 0 converted_file.csv
"""


def get_column_to_change(old_file, column, delimiter):
    """ Extracts and returns the column of original_file by column_name."""

    df = pd.read_csv(old_file, sep=delimiter)
    return df[column].as_matrix()


def convert_ids(old_ids, map_txt, conv_type):
    """ Performs conversion of IDs with the help of DDB-GeneID-UniProt.txt file."""

    df = pd.read_csv(map_txt, sep="\t")
    ddb_id = df['DDBDDB ID'].as_matrix()
    ddb_g_id = df['DDB_G ID'].as_matrix()
    map_names = df['Name'].as_matrix()
    new_ids = old_ids.copy()
    names = np.empty(old_ids.shape, dtype=object)
    if conv_type == '0':
        for i, old_id in enumerate(old_ids):
            occurrence = list(np.where(ddb_id == old_id)[0])
            if len(occurrence) > 0:
                new_ids[i] = ddb_g_id[occurrence[0]]
                if map_names[occurrence[0]][:12] == ddb_g_id[occurrence[0]]:
                    names[i] = ""
                else:
                    names[i] = str(map_names[occurrence[0]])
    else:
        for i, old_id in enumerate(old_ids):
            occurrence = list(np.where(ddb_g_id == old_id)[0])
            if len(occurrence) > 0:
                new_ids[i] = ddb_id[occurrence[0]]
    return new_ids, names


def write_changed_file(new_ids, old_file, column, delimiter, new_file, multiple, names):
    """ Writes converted_file with converted column."""

    names_tf = pd.read_csv("transcriptionFactors/features.ids", header=None, sep=" ")[1].as_matrix()

    if multiple:
        data_dir, data_file = os.path.split(old_file)
        first, second = data_file.split(".")
        new_file = data_dir + "/mutants_G/" + first + "_G." + second
        print(new_file)
        df = pd.read_csv(old_file, sep=delimiter)
        df[column] = new_ids
        df.to_csv(new_file, sep=delimiter, index=None)
    else:
        df = pd.read_csv(old_file, sep=delimiter)
        header = list(df)
        for i in range(1, len(header)):
            header[i] = names_tf[i-1] + "_" + header[i]
        df.columns = header
        df.insert(1, "Name", names)
        df[column] = new_ids
        df = df.round(decimals=2)
        df.to_csv(new_file, sep=delimiter, index=None)
        df2 = pd.read_csv(new_file, sep=delimiter, header=None)
        df2.T.to_csv(new_file+'.T', sep=delimiter, index=None, header=None)


def main():
    start = time.time()
    map_txt = "code/DDB_DDB_G/DDB-GeneID-UniProt.txt"
    arguments = sys.argv[1:]
    multiple = False
    if len(arguments) < 5:
        print("Not enough arguments stated! Usage: \n"
              "python map_columns.py <original_file> <column_name> <conversion_type> <converted_file> <delimiter>")
        return

    old_file = arguments[0]
    column = arguments[1]
    conv_type = arguments[2]
    new_file = arguments[3]
    delimiter = arguments[4]

    if len(arguments) > 5:
        multiple = True

    if multiple:
        file = open(old_file, "r")
        data_dir, _ = os.path.split(old_file)
        for row in file:
            new_old_file = data_dir + "/" + row[:-1]
            print(new_old_file)
            old_ids = get_column_to_change(new_old_file, column, delimiter)
            new_ids, names = convert_ids(old_ids, map_txt, conv_type)
            write_changed_file(new_ids, new_old_file, column, delimiter, new_file, multiple, names)
    else:
        old_ids = get_column_to_change(old_file, column, delimiter)
        new_ids, names = convert_ids(old_ids, map_txt, conv_type)
        write_changed_file(new_ids, old_file, column, delimiter, new_file, multiple, names)

    end = time.time() - start
    print("Program has successfully written mapped file at " + new_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
