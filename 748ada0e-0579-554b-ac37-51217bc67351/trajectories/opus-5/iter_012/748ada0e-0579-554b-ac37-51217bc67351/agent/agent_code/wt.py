import numpy as np, sys
sys.path.insert(0, '/workspace/environment'); sys.path.insert(0, '/workspace/environment/starter')
import aello_data as data
from train import bal_acc
from postproc import VIEWSETS
c = np.load('/workspace/pv_A.npz', allow_pickle=True)['c'].item()
_, yv = data.load_val()
def views(split, nm, vs):
    d = c[(split, nm)]
    return np.mean([d[(o, f)] for o in VIEWSETS[vs] for f in (False, True)], 0)
for vs in ('flip5', 'flip9'):
    R, E = views('val', 'raw', vs), views('val', 'ema0.999', vs)
    print(vs, 'log-space (geometric) blend:')
    for w in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
        print('   w_raw=%.1f  %.4f' % (w, bal_acc((w * R + (1 - w) * E).argmax(1), yv)), end='')
    print()
    Rp, Ep = np.exp(R), np.exp(E)
    print(vs, 'prob-space (arithmetic) blend:')
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        print('   w_raw=%.1f  %.4f' % (w, bal_acc((w * Rp + (1 - w) * Ep).argmax(1), yv)), end='')
    print()
    T = views('val', 'ema0.9995', vs)
    print('   3-way equal log %.4f' % bal_acc(((R + E + T) / 3).argmax(1), yv))
