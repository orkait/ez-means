import statistics
from dataclasses import dataclass

from ezmeans.dataset import Spec, class_means, sample
from ezmeans.metrics import score
from ezmeans.reference import SWAP_RATIO, Emeans, Revival, law_recipe

SEEDS = range(10)
HEADER = (
    "| case | revival | ratio | train cosine, median | test cosine, median | recovered min / median of classes "
    "| swaps, median |\n|---|---|---|---|---|---|---|"
)


@dataclass(frozen=True)
class Case:
    name: str
    spec: Spec
    clusters: int
    epochs: int


CASES = (
    Case("A kepler-like", Spec(2176, 40, 0.025, 0.0), 40, 10),
    Case("B overlapping", Spec(2176, 40, 0.15, 0.0), 40, 10),
    Case("C unstructured", Spec(2176, 40, 10.0, 0.0), 40, 10),
    Case("D skewed sizes", Spec(2176, 40, 0.025, 1.0), 40, 10),
    Case("E large", Spec(65536, 256, 0.025, 0.0), 256, 5),
)


def row(case: Case, revival: Revival, ratio: float) -> str:
    train_cos, test_cos, recovered, swaps = [], [], [], []
    classes = 0
    for seed in SEEDS:
        train, test = sample(case.spec, seed)
        means = class_means(train)
        classes = len(means)
        recipe = law_recipe(len(train.points), case.clusters, case.epochs, seed)
        model = Emeans(case.clusters, recipe, revival, ratio)
        centroids = model.train(train.points)
        fit = score(train.points, means, centroids)
        train_cos.append(fit.mean_cosine)
        test_cos.append(score(test.points, means, centroids).mean_cosine)
        recovered.append(fit.recovered)
        swaps.append(model.swaps.done)
    return (
        f"| {case.name} | {revival.value} | {ratio:g} | {statistics.median(train_cos):.5f} | "
        f"{statistics.median(test_cos):.5f} | {min(recovered)} / {statistics.median(recovered):g} of {classes} | "
        f"{statistics.median(swaps):g} |"
    )


def main() -> None:
    print(HEADER)
    for case in CASES:
        for revival in Revival:
            print(row(case, revival, SWAP_RATIO))


if __name__ == "__main__":
    main()
