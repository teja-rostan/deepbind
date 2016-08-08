# deepbind

### binding_score.py

Score DNA/RNA sequences stored in FASTA format (_fasta_path_) according to any RBP/TF
model (_features_path_) listed in the DeepBind web repository:
[http://tools.genes.toronto.edu/deepbind](http://tools.genes.toronto.edu/deepbind)
For scoring, the program uses deepbind executable (_deepbind_path_) that can be downloaded at:
[http://tools.genes.toronto.edu/deepbind/download.html](http://tools.genes.toronto.edu/deepbind/download.html)
Scores are written in a file (_results_path_). To speed up computation,
the program supports parallelization on multiple CPU cores (_num_of_cpu_).


###### Usage:
_python promoter_score.py fasta_path features_path deepbind_path results_path num_of_cpu_,
where _num_of_cpu_ is optional (default is 4).

###### Example:
_python promoter_score.py promoter_sequences/promoter_sequences.fasta
goal1/features.ids deepbind goal1/promoter_scores.csv 12_,
where _deepbind_ executable is in same dir.

### map_columns.py

Gene ID converter. Files use different ID groups for the same sequence. Possible groups are DDB ID and DDB_G ID.
The program supports conversion from DDB to DDB_G (_conv_type_=0) or from DDB_G to DDB (_conv_type_=1). 
The program uses a file DDB-GeneID-UniPort.txt found in folder DDB_DDB_G for a reference. The file can be found at
[dictybase.org](http://dictybase.org/db/cgi-bin/dictyBase/download/download.pl?area=general&ID=DDB-GeneID-UniProt.txt).

###### Usage:
_python map_columns.py original_file column_name conversion_type converted_file delimiter_,
where _delimiter_ is optional (default is '\\t').

###### Example:
_python map_columns.py original_file.csv ID 0 converted_file.csv_
### join_scores.py

The program joins scores of two files: _binding_score_file_ (binding scores of sequences (with ID name) on features,
the output of _binding_score.py_) and _expression_score_file_ (expression scores of sequences (with ID name). For all
pairs of sequences (sequence of same ID that exist in both files) the program saves binding scores along with
expression score.

###### Usage:
_python join_scores.py binding_score_file expression_score_file column_ID_name column_score_name
new_joined_file delimiter_, where _delimiter_ is optional (default is '\\t').

###### Example:
_python join_scores.py binding_score_file.csv expression_score_file.txt gene_ID exp_score new_joined_file.csv_

### mutation_candidates.py

Creates a mutation (sensitivity) map of sequences presented in _fasta_path_ file. It uses _binding_score.py_ for scoring
original sequence and its mutations. Mutation map is presented in
Alipanahi et al. 2015. The program creates another file of ranked mutations.

###### Usage:
_python mutation_candidates.py fasta_path features_path deepbind_path results_path num_of_cpu=4
motif_len=10_, where _num_of_cpu=4_ and _motif_len=10_ are optional.

###### Example:
_python mutation_candidates.py sequences.fasta feature.id deepbind/deepbind scores.csv_

## Requirements    
Requrements for this repository are found in _requirements.txt_.
