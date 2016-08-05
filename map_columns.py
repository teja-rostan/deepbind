import pandas as pd
import numpy as np
import time
import sys


help = \
    """
    Gene ID converter. Files use different ID groups for the same sequence. Possible groups are DDB ID and DDB_G ID.
    The program supports conversion from DDB to DDB_G (conv_type=0) or from DDB_G to DDB (conv_type=1).

    Usage:
        "python map_columns.py <original_file> <column_name> <conversion_type> <converted_file> <delimiter>, "
              "where <delimiter> is optional (default is '\\t')."

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
    new_ids = old_ids.copy()
    if conv_type == '0':
        for i, id in enumerate(old_ids):
            occurrence = list(np.where(ddb_id == id)[0])
            if len(occurrence) > 0:
                new_ids[i] = ddb_g_id[occurrence[0]]
    else:
        for i, id in enumerate(old_ids):
            occurrence = list(np.where(ddb_g_id == id)[0])
            if len(occurrence) > 0:
                new_ids[i] = ddb_id[occurrence[0]]
    return new_ids


def write_changed_file(new_ids, old_file, column, delimiter, new_file):
    """ Writes converted_file with converted column."""

    df = pd.read_csv(old_file, sep=delimiter)
    df[column] = new_ids
    print(delimiter)
    df.to_csv(new_file, sep=delimiter)


def main():
    start = time.time()
    map_txt = "DDB_DDB_G/DDB-GeneID-UniProt.txt"
    delimiter = '\t'
    arguments = sys.argv[1:]

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python map_columns.py <original_file> <column_name> <conversion_type> <converted_file> <delimiter>, "
              "where <delimiter> is optional (default is '\\t').")
        return

    old_file = arguments[0]  # path to file we want to change
    column = arguments[1]  # column name we want to change
    conv_type = arguments[2]  # 0:ID->ID_G, 1:ID_G->ID
    new_file = arguments[3]  # path to new/changed file
    if len(arguments) > 4:
        delimiter = arguments[4]  # delimiter of old_file, default = '\t

    old_ids = get_column_to_change(old_file, column, delimiter)
    new_ids = convert_ids(old_ids, map_txt, conv_type)
    write_changed_file(new_ids, old_file, column, delimiter, new_file)

    end = time.time() - start
    print("Program has successfully written mapped file at " + new_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
