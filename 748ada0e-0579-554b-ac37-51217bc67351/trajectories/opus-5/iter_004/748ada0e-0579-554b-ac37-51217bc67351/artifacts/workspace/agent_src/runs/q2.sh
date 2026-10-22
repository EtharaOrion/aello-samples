while pgrep -f "q1.sh" > /dev/null; do sleep 5; done
set -x
cd /workspace/environment/starter
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/d_r68.npy --tag D_r68w112_D \
  --arch r68 --blocks 5,8,14,5 --width 112 --dsample d
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
  --out /workspace/runs/e_br3.npy --tag E_r52w128_D_br3 \
  --arch r52 --blocks 4,6,10,2 --width 128 --dsample d --branches 3
