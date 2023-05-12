batch_size=10
gpu="0,1"
epoch=150
base_lr=1e-3
data_root=''
output="./del_test/seg0"
out=''
#nohup python -u train_equ.py --gpu $gpu -b $batch_size -o $output -e $epoch --base_lr $base_lr > $out 2>&1 &
python -u train_equ.py --data_root $data_root --gpu $gpu -b $batch_size -o $output -e $epoch --base_lr $base_lr
