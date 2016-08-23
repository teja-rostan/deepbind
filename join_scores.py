import time
import sys
import pandas as pd
import numpy as np
import os


help = \
    """
    The program joins scores of two files: binding_score_file (binding scores of sequences (with ID name) on features,
    the output of binding_score.py) and expression_score_file (expression scores of sequences (with ID name). For all
    pairs of sequences (sequence of same ID that exist in both files) the program saves binding scores along with
    expression score.

    Usage:
        python join_scores.py <binding_score_file> <expression_score_file> <column_ID_name> <column_score_name>
        <new_joined_file> <delimiter>.

    Example:
        python join_scores.py binding_score_file.csv expression_score_file.txt gene_ID exp_score new_joined_file.csv
"""


def get_expression_scores(expression_file, column_id_name, column_scores_name, delimiter):
    """ From expression file returns two columns: column with IDs and a column with expressions (column_scores_name)."""

    df = pd.read_csv(expression_file, sep=delimiter)
    return df[column_scores_name].as_matrix(), df[column_id_name].as_matrix()


def get_all_expression_scores(expression_file, column_id_name, column_scores_name, delimiter):
    """ From expression file returns a matrix of expressions and a column with IDs."""

    file = open(expression_file, "r")
    data_dir, _ = os.path.split(expression_file)
    all_exp_scores = []
    col_names = []
    exp_ids = []
    for row in file:
        new_exp_file = data_dir + "/" + row[:-1]
        col_names.append(row[:-4])
        print(new_exp_file)
        exp_scores, exp_ids = get_expression_scores(new_exp_file, column_id_name, column_scores_name, delimiter)
        all_exp_scores.append(exp_scores)
    return np.asarray(all_exp_scores).T, exp_ids, col_names


def create_new_exp_score_col(binding_file, exp_scores, exp_ids):
    """ Creates a new column with expressions in same order as all matching pairs of IDs from expression file and
    binding file."""

    df = pd.read_csv(binding_file, sep='\t')
    bind_ids = df['ID'].as_matrix()
    bad = []
    new_exp_scores = []
    for i, bind_id in enumerate(bind_ids):
        occurrence = list(np.where(exp_ids == bind_id)[0])
        if len(occurrence) > 0:
            new_exp_scores.append(exp_scores[occurrence[0]])
        else:
            bad.append(i)
            print(bind_id)
    return new_exp_scores, len(exp_ids), bad


def write_joined_file_multiple(binding_file, new_exp_scores, new_file, bad, col_names):
    """ Same as function write_joined_file, only it handles lists of new_exp_scores and bad. (also names columns of
    expression differently (same as file_name with expression)."""

    df = pd.read_csv(binding_file, sep='\t')
    df = df.drop(df.index[bad])
    df = df.reset_index()
    df2 = pd.DataFrame(data=new_exp_scores, columns=col_names)
    results = pd.concat([df, df2], axis=1)
    results.to_csv(new_file, sep="\t", index=None)


def write_joined_file(binding_file, new_exp_scores, new_file, bad):
    """ Joins matching IDs with expression score values and writes in to a new_file."""

    df = pd.read_csv(binding_file, sep='\t')
    df = df.drop(df.index[bad])
    df = df.reset_index()
    df['expression_score'] = pd.Series(new_exp_scores)
    df.to_csv(new_file, sep="\t", index=None)


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 6:
        print("Not enough arguments stated! Usage: \n"
              "python join_scores.py <binding_score_file> <expression_score_file> <column_ID_name> <column_score_name> "
              "<new_joined_file> <delimiter>")
        return

    binding_file = arguments[0]
    expression_file = arguments[1]
    column_id_name = arguments[2]
    column_scores_name = arguments[3]
    new_file = arguments[4]
    delimiter = arguments[5]
    multiple = False
    if len(arguments) > 6:
        multiple = True

    if multiple:
        exp_scores, exp_ids, col_names = get_all_expression_scores(expression_file, column_id_name, column_scores_name, delimiter)
        new_exp_scores, all_scores, bad = create_new_exp_score_col(binding_file, exp_scores, exp_ids)
        if len(bad) == all_scores:
            print("No instances were found!")
            return
        write_joined_file_multiple(binding_file, new_exp_scores, new_file, bad, col_names)

    else:
        exp_scores, exp_ids = get_expression_scores(expression_file, column_id_name, column_scores_name, delimiter)
        new_exp_scores, all_scores, bad = create_new_exp_score_col(binding_file, exp_scores, exp_ids)
        if len(bad) == all_scores:
            print("No instances were found!")
            return
        write_joined_file(binding_file, new_exp_scores, new_file, bad)

    end = time.time() - start
    print("Program has successfully written joined file at " + new_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
