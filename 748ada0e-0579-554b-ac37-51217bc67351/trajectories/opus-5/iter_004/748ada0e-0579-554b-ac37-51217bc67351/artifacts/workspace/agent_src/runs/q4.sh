while pgrep -f "q3.sh" > /dev/null; do sleep 3; done
set -x
cd /workspace/environment/starter
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/h_d2.npy --tag H_w160_d2 --dsample d2
python3 train.py --instance i20k --seed 1 --graded ../data/graded_images.npz \
  --out /workspace/runs/i_c_seed1.npy --tag I_w160_D_seed1 --dsample d
