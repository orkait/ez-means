import faiss
import numpy as np

from ezmeans.config import SIFT_LEARN
from ezmeans.dataset import Matrix, read_fvecs
from ezmeans.reference import (
    EPOCH_SEED_MODULUS,
    EPOCH_SEED_MULTIPLIER,
    Emeans,
    Metric,
    Revival,
    Wide,
    epoch_rows,
    law_recipe,
)

POOL_ROWS = 100_000
CLUSTER_COUNTS = (256, 1024)
EPOCHS = (2, 3)
SEED = 0


class Traced(Emeans):
    def __init__(self, k: int, epochs: int, n: int) -> None:
        super().__init__(k, law_recipe(n, k, epochs, SEED), Revival.MASS, metric=Metric.EUCLIDEAN)
        self.snapshots: list[Matrix] = []
        self.first: list[Wide] = []
        self.second: list[Wide] = []
        self.labels: list[np.ndarray] = []

    def assign(self, rows: Matrix) -> tuple[Wide, np.ndarray, Wide]:
        self.snapshots.append(self.centroids.copy())
        distances, labels = faiss.knn(rows, self.centroids, 2)
        best, second = distances[:, 0].astype(np.float64), distances[:, 1].astype(np.float64)
        self.first.append(best)
        self.second.append(second)
        self.labels.append(labels[:, 0].astype(np.int64))
        self.distances += len(rows) * self.k
        return best, labels[:, 0].astype(np.int64), second


def distances_to(rows: Matrix, centroids: Matrix) -> Wide:
    wide_rows, wide_centroids = rows.astype(np.float64), centroids.astype(np.float64)
    squared = (
        (wide_rows**2).sum(axis=1)[:, None]
        + (wide_centroids**2).sum(axis=1)[None, :]
        - 2.0 * (wide_rows @ wide_centroids.T)
    )
    return np.asarray(np.sqrt(np.maximum(squared, 0.0)), dtype=np.float64)


def elkan(model: Traced, x: Matrix, epochs: int, net: bool) -> tuple[list[float], int]:
    n, recipe = len(x), model.recipe
    moves = np.stack(
        [np.linalg.norm(b - a, axis=1) for a, b in zip(model.snapshots, model.snapshots[1:], strict=False)]
    )
    drift = np.vstack([np.zeros(model.k), np.cumsum(moves, axis=0)])
    table = np.zeros((n, model.k), dtype=np.float32)

    def displacement(now: int, earlier: np.ndarray) -> Wide:
        if not net:
            return np.asarray(drift[now][None, :] - drift[earlier], dtype=np.float64)
        rows = {
            int(then): np.linalg.norm(model.snapshots[now] - model.snapshots[then], axis=1)
            for then in np.unique(earlier)
        }
        return np.stack([rows[int(then)] for then in earlier]).astype(np.float64)

    seen_step = np.full(n, -1)
    seen_label = np.zeros(n, dtype=np.int64)
    shares: list[float] = []
    violations = 0
    step = 0
    for epoch, length in enumerate(epoch_rows(n, epochs)):
        order = np.random.RandomState((recipe.seed * EPOCH_SEED_MULTIPLIER + epoch) % EPOCH_SEED_MODULUS).permutation(n)
        hits = visits = 0
        for begin in range(0, length, recipe.batch):
            batch = order[begin : min(begin + recipe.batch, length)]
            back = seen_step[batch]
            known = back >= 0
            if known.any():
                earlier, points = back[known], batch[known]
                moved = displacement(step, earlier)
                lower = table[points].astype(np.float64) - moved
                owner = seen_label[points]
                upper = lower[np.arange(len(points)), owner] + 2.0 * moved[np.arange(len(points)), owner]
                lower[np.arange(len(points)), owner] = np.inf
                proved = upper <= lower.min(axis=1)
                hits += int(proved.sum())
                visits += int(known.sum())
                violations += int((proved & (model.labels[step][known] != owner)).sum())
            table[batch] = distances_to(x[batch], model.snapshots[step])
            seen_step[batch] = step
            seen_label[batch] = model.labels[step]
            step += 1
        if epoch > 0:
            shares.append(hits / max(1, visits))
    return shares, violations


def skipped(model: Traced, n: int, epochs: int) -> tuple[list[float], int]:
    recipe = model.recipe
    moves = np.stack(
        [np.linalg.norm(b - a, axis=1) for a, b in zip(model.snapshots, model.snapshots[1:], strict=False)]
    )
    drift = np.vstack([np.zeros(model.k), np.cumsum(moves, axis=0)])
    seen_step = np.full(n, -1)
    seen_label = np.zeros(n, dtype=np.int64)
    seen_upper = np.zeros(n)
    seen_lower = np.zeros(n)
    shares = []
    violations = 0
    step = 0
    for epoch, length in enumerate(epoch_rows(n, epochs)):
        order = np.random.RandomState((recipe.seed * EPOCH_SEED_MULTIPLIER + epoch) % EPOCH_SEED_MODULUS).permutation(n)
        hits = visits = 0
        for begin in range(0, length, recipe.batch):
            batch = order[begin : min(begin + recipe.batch, length)]
            back = seen_step[batch]
            known = back >= 0
            if known.any():
                earlier, points = back[known], batch[known]
                widest = np.zeros(len(drift))
                for then in np.unique(earlier):
                    widest[then] = (drift[step] - drift[then]).max()
                own = drift[step][seen_label[points]] - drift[earlier, seen_label[points]]
                upper = seen_upper[points] + own
                lower = seen_lower[points] - widest[earlier]
                proved = upper <= lower
                hits += int(proved.sum())
                visits += int(known.sum())
                violations += int((proved & (model.labels[step][known] != seen_label[points])).sum())
            seen_step[batch] = step
            seen_label[batch] = model.labels[step]
            seen_upper[batch] = np.sqrt(np.maximum(model.first[step], 0.0))
            seen_lower[batch] = np.sqrt(np.maximum(model.second[step], 0.0))
            step += 1
        if epoch > 0:
            shares.append(hits / max(1, visits))
    return shares, violations


def main() -> None:
    pool = read_fvecs(SIFT_LEARN, POOL_ROWS)
    print(f"SIFT {pool.shape}: share of revisits whose label a drift bound proves unchanged\n")
    print("| k | epochs | bound | skipped in epoch 2 | skipped in epoch 3 | proved but label changed |")
    print("|---|---|---|---|---|---|")
    for k in CLUSTER_COUNTS:
        for epochs in EPOCHS:
            model = Traced(k, epochs, len(pool))
            model.train(pool)
            for bound, (shares, violations) in (
                ("hamerly", skipped(model, len(pool), epochs)),
                ("elkan, path drift", elkan(model, pool, epochs, net=False)),
                ("elkan, net displacement", elkan(model, pool, epochs, net=True)),
            ):
                cells = " | ".join(f"{share:.3f}" for share in shares) + (" | -" if len(shares) == 1 else "")
                print(f"| {k} | {epochs} | {bound} | {cells} | {violations} |", flush=True)


if __name__ == "__main__":
    main()
