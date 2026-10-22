set -x
cd /workspace/environment/starter
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/f_3stage.npy --tag F_3stage_w160_D \
  --blocks 3,4,9 --width 160 --dsample d
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/g_br2.npy --tag G_w160_D_br2 \
  --blocks 3,4,6,2 --width 160 --dsample d --branches 2
