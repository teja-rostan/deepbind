import time
import sys
import os
import subprocess
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

help = \
    """
    Creates four directories:
     - new_10: for every TF matrix file (sequences * binding scores) creates a new matrix file of 10 pca components
               (sequences * 10)
     - new_50: same as new_10 but with 50 pca components.
     - org_10: for every TF matrix file (sequences * binding scores) creates a new matrix file which is an inverse
               transformation back to original space with 10 pca components.
     - org_50: same as org_10 but with 50 pca components.

    Usage:
        python preprocess_bind.py <bind_tf_dir> <output_dir> <delimiter>.
"""


def get_tf_scores(data_dir, tf, delimiter):
    """ Gets a TF matrix where nan values are filled with most frequent value in a row
    (if any nan value remains is filed with median in a row). """

    tf_scores = pd.read_csv(data_dir + "/" + tf, delimiter=delimiter, na_values='?')
    tf_scores = tf_scores.T.fillna(tf_scores.mode(axis=1, numeric_only=True)[0]).T
    tf_scores = tf_scores.T.fillna(tf_scores.median(axis=1, numeric_only=True)).T
    # tf_scores = tf_scores.fillna(0)
    # tf_scores = tf_scores.T.dropna(axis=1, how='any').T
    # tf_scores = tf_scores.replace('?', 0)
    return tf_scores.drop('ID', axis=1).as_matrix(), tf_scores['ID'].as_matrix()


def get_pca(X, n_components, tf_name, seq_ids, output_dir):
    """ gets pca with n_components (10 or 50). The matrix is saved to a file.
     The function returns explained varience and explained varience ratio. """

    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)
    exp_var = pca.explained_variance_
    exp_var_ratio = pca.explained_variance_ratio_
    X_org = pca.inverse_transform(X_new)

    X_new_df = pd.DataFrame(X_new).set_index(seq_ids).round(decimals=2)
    X_org_df = pd.DataFrame(X_org).set_index(seq_ids).round(decimals=2)
    X_new_df.index.name = 'ID'
    X_org_df.index.name = 'ID'

    print(tf_name[:-4])
    X_new_df.to_csv(output_dir + "new_" + str(n_components) + "/" + tf_name[:-4] + "_" + str(n_components) + "X_new.csv", sep='\t')
    X_org_df.to_csv(output_dir + "org_" + str(n_components) + "/" + tf_name[:-4] + "_" + str(n_components) + "X_org.csv", sep='\t')

    return exp_var, exp_var_ratio


def pca_processing(tfs, data_dir, output_dir, delimiter, n_components):
    """ Runs all get_pca functions and saves a file with data of expleined varience and expleined varience ratio. """

    all_exp_vars = []
    header_names = []
    seq_ids = 0

    for tf in tfs:
        X, seq_ids = get_tf_scores(data_dir, tf, delimiter)

        exp_var, exp_var_ratio = get_pca(X, n_components, tf, seq_ids, output_dir)
        all_exp_vars.append(exp_var)
        all_exp_vars.append(exp_var_ratio)

        header_names.append(tf[:-4] + '_exp_var')
        header_names.append(tf[:-4] + '_exp_var_ratio')

    all_exp_vars = np.array(all_exp_vars)
    df_exp_vars = pd.DataFrame(all_exp_vars, index=header_names).round(decimals=2)
    df_exp_vars.index.name = 'ID'
    df_exp_vars.to_csv(output_dir + "/explained_variance_" + str(n_components) + "comp.csv", sep='\t')
    return seq_ids, header_names


def concatenate_tables(output_dir, delimiter, n_components, seq_ids):
    """ Horizontically stacks all pca components (10 or 500) matrices into a single matrix of shape:
    [4112 * (n_components * num_of_TFs)]. """

    pca_dir = output_dir + "new_" + str(n_components) + "/"
    tfs_pca = os.listdir(pca_dir)
    all_pcas = []
    for tf_pca in tfs_pca:
        print(pca_dir+tf_pca)
        all_pcas.append(pd.read_csv(pca_dir + tf_pca, delimiter=delimiter).drop('ID', axis=1).as_matrix())
    df = pd.DataFrame(np.hstack(all_pcas), index=seq_ids)
    df.index.name = 'ID'
    df.to_csv(output_dir + "new_" + str(n_components) + "/concat.csv", sep='\t')


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 3:
        print("Not enough arguments stated! Usage: \n"
              "python preprocess_bind.py <bind_tf_dir> <output_dir> <delimiter>.")
        return

    bind_tf_dir = arguments[0]
    output_dir = arguments[1]
    delimiter = arguments[2]

    data_dir, _ = os.path.split(bind_tf_dir)
    tfs = os.listdir(bind_tf_dir)

    subprocess.run(['mkdir', output_dir + "new_10"])
    subprocess.run(['mkdir', output_dir + "new_50"])
    subprocess.run(['mkdir', output_dir + "org_10"])
    subprocess.run(['mkdir', output_dir + "org_50"])

    seq_ids, header_names = pca_processing(tfs, data_dir, output_dir, delimiter, 10)
    concatenate_tables(output_dir, delimiter, 10, seq_ids)

    seq_ids, header_names = pca_processing(tfs, data_dir, output_dir, delimiter, 50)
    concatenate_tables(output_dir, delimiter, 50, seq_ids)

    end = time.time() - start
    print("Program has successfully written scores at " + output_dir + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
