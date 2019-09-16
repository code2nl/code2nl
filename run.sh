WORK_DIR=$1
DATANAME=$2
GPU_ID=$3
DATA_DIR=$WORK_DIR/data_model/$DATANAME
python preprocess.py -root_dir "$WORK_DIR/data_model/" -dataset django -src_words_min_frequency 3 -tgt_words_min_frequency 5
CUDA_VISIBLE_DEVICES=$GPU_ID CUDA_LAUNCH_BLOCKING=1 python train.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -rnn_size 300 -word_vec_size 250 -decoder_input_size 200 -layers 1 -start_checkpoint_at 5 -learning_rate 0.002 -global_attention "dot" -attn_hidden 0 -dropout 0.3 -dropout_i 0.3 -lock_dropout -copy_prb hidden
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split dev -model_path "$DATA_DIR/run.1/m_*.pt"
MODEL_PATH=$(head -n1 $DATA_DIR/dev_best.txt)
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$MODEL_PATH"