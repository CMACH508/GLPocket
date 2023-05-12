batch_size=20
output="./del_test/seg0"
gpu="0,1,2,3"
epoch=150
base_lr=1e-3
txt='./del_test.out'
data_root=''
#nohup python -u train.py --data_root $data_root --gpu $gpu -b $batch_size -o $output -e $epoch --base_lr $base_lr > $txt 2>&1 &
python -u train.py --data_root $data_root --gpu $gpu -b $batch_size -o $output -e $epoch --base_lr $base_lr
