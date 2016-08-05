import time
import sys
import pandas as pd
import numpy as np


help = \
    """
    The program joins scores of two files: binding_score_file (binding scores of sequences (with ID name) on features,
    the output of binding_score.py) and expression_score_file (expression scores of sequences (with ID name). For all
    pairs of sequences (sequence of same ID taht exist in both files) the program saves binding scores along with
    expression score.

    Usage:
        python join_scores.py <binding_score_file> <expression_score_file> <column_ID_name> <column_score_name>
        <new_joined_file> <delimiter>, where <delimiter> is optional (default is '\\t').

    Example:
        python join_scores.py binding_score_file.csv expression_score_file.txt gene_ID exp_score new_joined_file.csv
"""


def get_expression_scores(expression_file, column_id_name, column_scores_name, delimiter):
    df = pd.read_csv(expression_file, sep=eval(delimiter))
    return df[column_scores_name].as_matrix(), df[column_id_name].as_matrix()


def create_new_exp_score_col(binding_file, exp_scores, exp_ids):
    df = pd.read_csv(binding_file, sep='\t')
    bind_ids = df['ID'].as_matrix()
    new_exp_scores = []
    copied = []
    for i, bind_id in enumerate(bind_ids):
        occurrence = list(np.where(exp_ids == bind_id)[0])
        if len(occurrence) > 0:
            new_exp_scores.append(exp_scores[occurrence[0]])
            copied.append(i)
    return new_exp_scores, copied


def write_joined_file(binding_file, new_exp_scores, new_file, copied):
    df = pd.read_csv(binding_file, sep='\t')
    df = df.loc[copied, :]
    df['expression_score'] = pd.Series(new_exp_scores)
    df.to_csv(new_file, sep="\t", index=None)


def main():
    start = time.time()
    delimiter = '\t'
    arguments = sys.argv[1:]

    if len(arguments) < 5:
        print("Not enough arguments stated! Usage: \n"
              "python join_scores.py <binding_score_file> <expression_score_file> <column_ID_name> <column_score_name> "
              "<new_joined_file> <delimiter>, where <delimiter> is optional (default is '\\t').")
        return

    binding_file = arguments[0]
    expression_file = arguments[1]
    column_id_name = arguments[2]
    column_scores_name = arguments[3]
    new_file = arguments[4]
    if len(arguments) > 5:
        delimiter = arguments[5]

    exp_scores, exp_ids = get_expression_scores(expression_file, column_id_name, column_scores_name, delimiter)
    new_exp_scores, copied = create_new_exp_score_col(binding_file, exp_scores, exp_ids)
    if len(copied) == 0:
        print("No instances were found!")
        return
    write_joined_file(binding_file, new_exp_scores, new_file, copied)

    end = time.time() - start
    print("Program has successfully written joined file at " + new_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
