import numpy as np
from bounds import Traced

from ezmeans.config import SIFT_LEARN
from ezmeans.dataset import read_fvecs
from ezmeans.reference import EPOCH_SEED_MODULUS, EPOCH_SEED_MULTIPLIER, epoch_rows

POOL_ROWS = 100_000
CLUSTER_COUNTS = (256, 1024)
EPOCHS = 3
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3)


def main() -> None:
    pool = read_fvecs(SIFT_LEARN, POOL_ROWS)
    n = len(pool)
    print(f"SIFT {pool.shape}, e-means T={EPOCHS}: skip a revisit when its last relative margin is at least tau\n")
    print("| k | tau | revisits skipped | skipped whose label changed | all revisits whose label changed |")
    print("|---|---|---|---|---|")
    for k in CLUSTER_COUNTS:
        model = Traced(k, EPOCHS, n)
        model.train(pool)
        margin = np.full(n, -1.0)
        label = np.zeros(n, dtype=np.int64)
        previous_margin, changed = [], []
        step = 0
        for epoch, length in enumerate(epoch_rows(n, EPOCHS)):
            seed = (model.recipe.seed * EPOCH_SEED_MULTIPLIER + epoch) % EPOCH_SEED_MODULUS
            order = np.random.RandomState(seed).permutation(n)
            for begin in range(0, length, model.recipe.batch):
                batch = order[begin : min(begin + model.recipe.batch, length)]
                now = model.labels[step]
                seen = margin[batch] >= 0
                previous_margin.append(margin[batch][seen])
                changed.append(now[seen] != label[batch][seen])
                best = np.sqrt(np.maximum(model.first[step], 0.0))
                second = np.sqrt(np.maximum(model.second[step], 0.0))
                margin[batch] = (second - best) / np.maximum(best, 1e-12)
                label[batch] = now
                step += 1
        margins, moved = np.concatenate(previous_margin), np.concatenate(changed)
        for tau in THRESHOLDS:
            skip = margins >= tau
            wrong = moved[skip].mean() if skip.any() else 0.0
            print(f"| {k} | {tau} | {skip.mean():.3f} | {wrong:.4f} | {moved.mean():.3f} |", flush=True)


if __name__ == "__main__":
    main()
