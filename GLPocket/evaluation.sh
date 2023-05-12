gpu='0'
test_set='coach420'
#test_set='holo4k'
#test_set='sc6k'
#test_set='pdbbind'
is_dca=0
top_n=0
data_root=''
ckpt_path=''
# nohup python -u camera_test.py --data_root $data_root --gpu $gpu --test_set $test_set --ckpt_path $ckpt_path > $out_txt 2>&1 &
python -u camera_test.py --data_root $data_root --gpu $gpu --test_set $test_set --ckpt_path $ckpt_path --is_dca $is_dca --top_n $top_n
