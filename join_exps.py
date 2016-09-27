import pandas as pd


join_list = open("results/joined_scores/join_mul_all_26_8_2016/join_mul_26_8_2016.txt", 'r')

dfs = []
for row in join_list:
    dfs.append(pd.read_csv("results/joined_scores/join_mul_all_26_8_2016/" + row[:-1], sep='\t'))
id_gene = []
name_gene = []
mutant = []
times = []
exps = []


one_len = pd.read_csv("results/joined_scores/join_mul_all_26_8_2016/join_mul0h_26_8_2016.csv", sep='\t')
one_len = one_len['ID'].shape[0]
for i in range(one_len):
    print(i)
    for j in range(14):
        idx = 0
        for k, df in enumerate(dfs):
            # print(file)
            id_gene.append(df.iat[i, 0])
            name_gene.append(df.iat[i, 1])
            mutant.append(list(df)[-14+j][:-3])
            times.append(str(idx) + 'h')
            idx += 2
            exps.append(df.iat[i, -14+j])
df = pd.DataFrame()
print(df.shape)
df.insert(0, "expression_score", exps)
print(df.shape)
df.insert(0, "time_interval", times)
print(df.shape)
df.insert(0, "mutant", mutant)
print(df.shape)
df.insert(0, "Name", name_gene)
print(df.shape)
df.insert(0, "ID", id_gene)
print(df.shape)
df.to_csv("nana.csv", sep='\t', index=None)
