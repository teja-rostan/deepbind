# deepbind

### binding_score.py

Score DNA/RNA sequences stored in FASTA format (fasta_path) according to any RBP/TF
model (features_path) listed in the DeepBind web repository:
[http://tools.genes.toronto.edu/deepbind](http://tools.genes.toronto.edu/deepbind)
For scoring, the program uses deepbind executable (deepbind_path) that can be downloaded at:
[http://tools.genes.toronto.edu/deepbind/download.html](http://tools.genes.toronto.edu/deepbind/download.html)
Scores are written in a file (results_path). To speed up computation,
the program supports parallelization on multiple CPU cores (num_of_cpu).


###### Usage:
python promoter_score.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>,
where <num_of_cpu> is optional (default is 4).

###### Example:
python promoter_score.py promoter_sequences/promoter_sequences.fasta
goal1/features.ids deepbind goal1/promoter_scores.csv 12,
where deepbind executable is in same dir.

### map_columns.py

Gene ID converter. Files use different ID groups for the same sequence. Possible groups are DDB ID and DDB_G ID.
The program supports conversion from DDB to DDB_G (conv_type=0) or from DDB_G to DDB (conv_type=1). 
The program uses a file DDB-GeneID-UniPort.txt found in folder DDB_DDB_G for a reference. The file can be found at
[dictybase.org](http://dictybase.org/db/cgi-bin/dictyBase/download/download.pl?area=general&ID=DDB-GeneID-UniProt.txt).

###### Usage:
python map_columns.py <original_file> <column_name> <conversion_type> <converted_file> <delimiter>,
where <delimiter> is optional (default is '\\t').

###### Example:
python map_columns.py original_file.csv ID 0 converted_file.csv

### join_scores.py

The program joins scores of two files: binding_score_file (binding scores of sequences (with ID name) on features,
the output of binding_score.py) and expression_score_file (expression scores of sequences (with ID name). For all
pairs of sequences (sequence of same ID taht exist in both files) the program saves binding scores along with
expression score.

###### Usage:
python join_scores.py <binding_score_file> <expression_score_file> <column_ID_name> <column_score_name>
<new_joined_file> <delimiter>, where <delimiter> is optional (default is '\\t').

###### Example:
python join_scores.py binding_score_file.csv expression_score_file.txt gene_ID exp_score new_joined_file.csv

### mutation_candidates.py

Creates a mutation (sensitivity) map of sequences presented in fasta_path file. It uses binding_score.py for scoring
original sequence and its mutations. Mutation map is presented in
Alipanahi et al. 2015. The program creates another file of ranked mutations.

###### Usage:
python mutation_candidates.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu=4>
<motif_len=10>, where <num_of_cpu=4> and <motif_len=10> are optional.

###### Example:
python mutation_candidates.py sequences.fasta feature.id deepbind/deepbind scores.csv

## Requirements    
Requrements for this repository are found in requirements.txt.
