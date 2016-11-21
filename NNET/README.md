# Neural network learning and scoring

### nnet_score.py

The main program of a Neural Network learning model to predict and score
expressions (target variables) from binding scores of
transcription factors (evaluated by deepbind).

###### Usage:
python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>,
where learning_type={class, class_one, class_one_time, reg, reg_one, reg_one_time, ord}.

###### Example:
python nnet_score.py path/to/data_and_target_concatenated.csv path/to/correlation_results.csv reg $'\t' 14


### get_data_target.py

A code with functions for all required tasks that handle the main input 
data. Makes input data appropriate for classification, regression or 
ordinal problem. 

### nnet.py

A code with functions that are building blocks of neural network 
architecture.

### NnetClassLearner.py

Classification: Neural Network Learner (handles two or three hidden layers)

### NnetRegLearner.py

Regression: Neural Network Learner (handles two or three hidden layers)

### nnet_class_one.py

Program that uses classification neural network learner. If the problem
is multi-target, the program handles every target separately. The
program can (manually) balance data.

### nnet_reg_one.py

Program that uses regression neural network learner. If the problem
is multi-target, the program handles every target separately. The
program can (manually) balance data.

### nnet_class_one_time.py

Program that uses classification neural network learner. If the problem
is multi-target, the program handles every target separately. The
program can (manually) balance data. It takes expressions as attributes
form every time interval available in one learning model. 

### nnet_reg_one_time.py

Program that uses regression neural network learner. If the problem
is multi-target, the program handles every target separately. The
program can (manually) balance data.It takes expressions as attributes
form every time interval available in one learning model. 

### nnet_class.py

Program that uses classification neural network learner. If the problem
is multi-target, the program uses a multi-target neural network.
Needed of update...

### nnet_reg.py

Program that uses regression neural network learner. If the problem
is multi-target, the program uses a multi-target neural network.
Needed of update...

### nnet_ord.py
Program that uses ordinal neural network learner [1]. If the problem
is multi-target, the program uses a multi-target neural network.
Needed of update...


### Source
[1] Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. 
"A neural network approach to ordinal regression." 
2008 IEEE International Joint Conference on Neural Networks 
(IEEE World Congress on Computational Intelligence). IEEE, 2008.