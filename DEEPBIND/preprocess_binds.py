import time
import sys
import os
import subprocess
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


def get_tf_scores(data_dir, tf, delimiter):
    # One by one, otherwise, too much memory consumption
    tf_scores = pd.read_csv(data_dir + "/" + tf, delimiter=delimiter)
    tf_scores = tf_scores.replace('?', 0)
    print(tf_scores)
    return tf_scores.drop('ID', axis=1).as_matrix(), tf_scores['ID'].as_matrix()


def get_pca(X, n_components, tf_name, seq_ids, output_dir):
    # pca with 10, 50 components. Save in table, along with variance covered
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)
    exp_var = pca.explained_variance_
    exp_var_ratio = pca.explained_variance_ratio_
    X_org = pca.inverse_transform(X_new)

    X_new_df = pd.DataFrame(X_new).set_index(seq_ids)
    X_org_df = pd.DataFrame(X_org).set_index(seq_ids)

    print(seq_ids)

    X_new_df.to_csv(output_dir + "/new_" + str(n_components) + "/" + tf_name[:-3] + "_" + str(n_components) + "X_new.csv", sep='\t')
    X_org_df.to_csv(output_dir + "/org_" + str(n_components) + "/" + tf_name[:-3] + "_" + str(n_components) + "X_org.csv", sep='\t')

    return exp_var, exp_var_ratio


def pca_processing(tfs, data_dir, output_dir, delimiter, n_components):
    all_exp_vars = []
    header_names = []
    seq_ids = 0
    for tf in tfs:
        X, seq_ids = get_tf_scores(data_dir, tf, delimiter)

        exp_var_10, exp_var_ratio_10 = get_pca(X, n_components, tf, seq_ids, output_dir)
        all_exp_vars.append(exp_var_10)
        all_exp_vars.append(exp_var_ratio_10)

        header_names.append(tf[:-3] + '_exp_var')
        header_names.append(tf[:-3] + '_exp_var_ratio')

    df_exp_vars = pd.DataFrame(all_exp_vars).set_index(seq_ids)
    df_exp_vars.to_csv(output_dir + "/explained_variance_" + str(n_components) + "comp.csv", header=header_names, sep='\t')
    return seq_ids, header_names


def concatenate_tables(output_dir, delimiter, n_components, seq_ids):
    list_file = open(output_dir + "/new_" + str(n_components) + "/", "r")
    tfs_pca = list_file.readlines()
    list_file.close()
    all_pcas = []
    for tf_pca in tfs_pca:
        all_pcas.append(pd.read_csv(output_dir + "/new_" + str(n_components) + "/" + tf_pca, delimiter=delimiter))
    df = pd.DataFrame(np.hstack(all_pcas)).set_index(seq_ids)
    df.to_csv(output_dir + "/new_" + str(n_components) + "/concat.csv", sep='\t')


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

    subprocess.run(['mkdir', output_dir + "/new_10"])
    subprocess.run(['mkdir', output_dir + "/new_50"])
    subprocess.run(['mkdir', output_dir + "/org_10"])
    subprocess.run(['mkdir', output_dir + "/org_50"])

    seq_ids, header_names = pca_processing(tfs, data_dir, output_dir, delimiter, 10)
    seq_ids, header_names = pca_processing(tfs, data_dir, output_dir, delimiter, 50)

    concatenate_tables(data_dir, delimiter, 10, seq_ids)
    concatenate_tables(data_dir, delimiter, 50, seq_ids)

    end = time.time() - start
    print("Program has successfully written scores at " + output_dir + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
