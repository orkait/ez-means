from generality import CASES, HEADER, row

from ezmeans.reference import SWAP_RATIO, Revival

RATIOS = (1.0, 4.0)


def main() -> None:
    print(HEADER)
    for case in CASES:
        print(row(case, Revival.MASS, SWAP_RATIO))
        for ratio in RATIOS:
            print(row(case, Revival.VERIFY, ratio))


if __name__ == "__main__":
    main()
