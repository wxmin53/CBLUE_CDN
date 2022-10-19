import os
from os.path import dirname, abspath

project_dir = os.path.dirname(dirname(dirname(abspath(__file__))))
data_dir = os.path.join(project_dir, "cblue/CBLUE_DATA")
model_dir = os.path.join(project_dir, "MODEL_TYPE")
model_type = "bert"
model_name = "chinese-macbert-large"
task_name = "cdn"
output_dir = os.path.join(project_dir, "myGit/CBLUE_CDN/data/output")
do_train = True
do_predict = False
result_output_dir = os.path.join(project_dir, "myGit/CBLUE_CDN/data/result_output")

# For CDN task
recall_k = 200  # 召回数量
num_neg = 5
do_aug = 6

# models param
max_length = 40
train_batch_size = 32
eval_batch_size = 256
learning_rate = 2e-5
weight_decay = 0.01
adam_epsilon = 1e-8
max_grad_norm = 0.0
epochs = 20
warmup_proportion = 0.1
earlystop_patience = 100

logging_steps = 250
save_steps = 250
seed = 2021

