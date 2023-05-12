batch_size=8
output="./del_test/seg0"
gpu="0,1,2,3"
epoch=20
base_lr=1e-4
txt='./del_test.out'
data_root=''
nohup python -u train.py --data_root $data_root --gpu $gpu -b $batch_size -o $output -e $epoch --base_lr $base_lr > $txt 2>&1 &