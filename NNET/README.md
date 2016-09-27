# Neural network learning and scoring

### nnet.py

### nnet_score.py

The program uses a Neural Network learning model to predict expressions (target variables) from binding scores of
    transcription factors (evaluated by deepbind)... MORE LATER...

###### Usage:
python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>,
where learning_type=class or learning_type=reg.

###### Example:
python nnet_score.py input_files.txt results.csv reg $'\t' 14

### nnet_class.py


### nnet_reg.py

