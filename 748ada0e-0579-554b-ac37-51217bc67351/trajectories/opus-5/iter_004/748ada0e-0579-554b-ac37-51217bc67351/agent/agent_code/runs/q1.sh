set -x
cd /workspace/environment/starter
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/b_deepD.npy --tag B_r52w128_D \
  --arch r52 --blocks 4,6,10,4 --width 128 --dsample d
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/c_baseD.npy --tag C_r34w160_D \
  --arch r34 --width 160 --dsample d
