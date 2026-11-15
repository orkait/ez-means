import statistics

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from ezmeans.config import CLUSTERS, DIM, RECOVERY_PASS, RUN_SEEDS
from ezmeans.dataset import class_means, generate
from ezmeans.metrics import score
from ezmeans.reference import Emeans, Revival, law_recipe

EPOCHS = (5, 10, 25)


def main() -> None:
    data = generate()
    means = class_means(data)
    n = len(data.points)
    identical = 0
    for seed in RUN_SEEDS:
        reference, _ = NumpyEmeans(DIM, CLUSTERS, nepochs=EPOCHS[-1], spherical=True, seed=seed).train(data.points)
        ours = Emeans(CLUSTERS, law_recipe(n, CLUSTERS, EPOCHS[-1], seed), Revival.MASS).train(data.points)
        identical += int(np.array_equal(np.asarray(reference, dtype=np.float32), ours))
    print(f"ours (mass revival) bit-identical to faiss NumpyEmeans: {identical} of {len(RUN_SEEDS)} seeds\n")
    print(
        f"| epochs | revival | median mean cosine | recovered min / median | seeds >= {RECOVERY_PASS} | swaps, median |"
    )
    print("|---|---|---|---|---|---|")
    for epochs in EPOCHS:
        for revival in Revival:
            scores, swaps = [], []
            for seed in RUN_SEEDS:
                model = Emeans(CLUSTERS, law_recipe(n, CLUSTERS, epochs, seed), revival)
                scores.append(score(data.points, means, model.train(data.points)))
                swaps.append(model.swaps.done)
            recovered = [s.recovered for s in scores]
            print(
                f"| {epochs} | {revival.value} | {statistics.median(s.mean_cosine for s in scores):.5f} | "
                f"{min(recovered)} / {statistics.median(recovered):g} | "
                f"{sum(r >= RECOVERY_PASS for r in recovered)} of {len(scores)} | {statistics.median(swaps):g} |"
            )


if __name__ == "__main__":
    main()
