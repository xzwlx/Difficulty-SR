# EDSR (x3)
# python main.py --data_test Set5 --model EDSR --scale 3 --patch_size 144 --n_colors 1 --n_GPUs 2 --n_threads 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# EDSR (x3) - switch
# python main.py --data_test Set5 --model EDSR_switch --scale 3 --patch_size 144 --n_colors 1 --n_GPUs 2 --n_threads 4 --save edsr_switch_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --switch --chop --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# FSRCNN baseline
# python main.py --data_test Set5 --data_train General100 --model FSRCNN --scale 3 --data_range='1-100' --test_every 100 --lr 1e-3 --epochs 100 --lr_decay 50 --patch_size 72 --batch_size 128 --loss='1*MSE' --n_colors 1 --n_GPUs 2 --n_threads 4 --save fsrcnn_base --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# FSRCNN baseline DIV2K
# python main.py --data_test Set5 --model FSRCNN --scale 3 --test_every 100 --lr 1e-3 --gamma 0.1 --epochs 200 --lr_decay 100 --patch_size 144 --batch_size 128 --loss='1*MSE' --n_colors 1 --n_GPUs 2 --n_threads 4 --save fsrcnn_base_div2k --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# FSRCNN
# python main.py --data_test Set5 --model FSRCNN --scale 3 --test_every 100 --lr 1e-3 --epochs 100 --lr_decay 20 --patch_size 144 --batch_size 128 --loss='1*MSE' --n_colors 1 --n_GPUs 2 --n_threads 4 --save fsrcnn2 --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/fsrcnn/model/model_latest.pt'

# FSRCNN - switch_self
# python main.py --model FSRCNN_switch --scale 3 --epochs 100 --lr_decay 20 --patch_size 144 --batch_size 16 --loss='1*MSE' --n_colors 1 --n_GPUs 2 --n_threads 4 --save fsrcnn_switch_new --reset --switch --chop --test_patch_size 48 --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/fsrcnn_switch/model/model_latest.pt'

# FSRCNN - switch_r
# python main.py --model FSRCNN_switch_r --scale 3 --epochs 100 --lr_decay 20 --patch_size 144 --batch_size 16 --loss='1*MSE' --n_colors 1 --n_GPUs 2 --n_threads 4 --save fsrcnn_switch_r_new --reset --switch --chop --test_patch_size 48 --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/fsrcnn_switch_r/model/model_latest.pt'

# FSRCNN - switch_new
# python main.py --data_test Set5 --model FSRCNN_switch_new --switch --chop --scale 3 --test_every 100 --lr 1e-3 --gamma 0.1 --epochs 200 --lr_decay 100 --patch_size 144 --batch_size 128 --loss='1*MSE' --n_colors 1 --n_GPUs 2 --n_threads 4 --save fsrcnn_switch_new_div2k --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# IDN
# python main.py --data_test Set5 --test_every 200 --model IDN --scale 3 --epochs 100 --lr_decay 100 --patch_size 144 --batch_size 64 --loss='1*L1' --n_colors 1 --n_GPUs 2 --n_threads 4 --save idn_l1_3 --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/idn_l1_2/model/model_best.pt'

# IDN - switch
# python main.py --data_test Set5 --test_every 200 --model IDN_switch --switch --chop --scale 3 --epochs 500 --lr_decay 250 --patch_size 144 --batch_size 64 --loss='1*L1' --n_colors 1 --n_GPUs 2 --n_threads 4 --save idn_switch_3 --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# IDN - switch finetune
# python main.py --data_test DIV2K --test_every 200 --model IDN_switch --switch --chop --shave 0 --scale 3 --level 4 --epochs 200 --lr_decay 100 --patch_size 144 --batch_size 64 --loss='1*L1' --n_colors 1 --n_GPUs 2 --n_threads 4 --save idn_switch_3 --reset --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/idn_switch_2/model/model_best.pt'

# lapsrn
# python main.py --model lapsrn --scale 3 --patch_size 144 --batch_size 16 --loss='1*L1' --n_colors 1 --n_GPUs 2 --n_threads 4 --save lapsrn --reset --test_patch --dir_data='/mnt/lustre/luhannan/ziwei/datasets'

# test
python main.py --data_test Set5+Set14+B100+Urban100 --model IDN_switch --switch --chop --shave 36 --level 4 --scale 3 --n_colors 1 --test_patch_size 48 --data_range='801-900' --n_resblocks 32 --n_feats 256 --res_scale 0.1 --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/idn_switch_3/model/model_best.pt' --test_only --save_results

# test branch
# python main.py --data_test DIV2K --model IDN_switch --test_branch 2 --switch --level 4 --scale 3 --n_colors 1 --test_patch_size 48 --data_range='801-900' --n_resblocks 32 --n_feats 256 --res_scale 0.1 --dir_data='/mnt/lustre/luhannan/ziwei/datasets' --pre_train='/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/idn_switch_2/model/model_best.pt' --test_only



