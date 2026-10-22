# Memory index

- [CIFAR-100 budget: TrivialAugment is the big win](cifar100-budget-trivialaugment-win.md) — +1.3% balanced acc over crop/flip/cutout; batched GPU implementation notes.
- [CIFAR-100 budget: H100 timings](cifar100-budget-timing-h100.md) — width/wall-clock curve, and why parallel runs buy nothing.
- [CIFAR-100 budget: eval noise](cifar100-budget-eval-noise.md) — val is ~1% noisy; post-hoc prior correction tuned on val backfired badly.
- [CIFAR-100 budget: best recipe](cifar100-budget-recipe.md) — the winning config, plus width/2-op-TA/no-cutout negatives.
