# deepbind

    Score DNA/RNA sequences stored in FASTA format (fasta_path) according to any RBP/TF
    model (features_path) listed in the DeepBind web repository:
    [http://tools.genes.toronto.edu/deepbind](http://tools.genes.toronto.edu/deepbind)
    For scoring, the program uses deepbind executable (deepbind_path) that can be downloaded at:
    [http://tools.genes.toronto.edu/deepbind/download.html](http://tools.genes.toronto.edu/deepbind/download.html)
    Scores are written in a file (results_path). To speed up computation,
    the program supports parallelization on multiple CPU cores (num_of_cpu).


    Usage:
        python promoter_score.py <fasta_path> <features_path> <deepbind_path> <results_path> <num_of_cpu>,
        where <num_of_cpu> is optional (default is 4).

    Example:
        python promoter_score.py promoter_sequences/promoter_sequences.fasta
        goal1/features.ids deepbind goal1/promoter_scores.csv 12,
        where deepbind executable is in same dir.