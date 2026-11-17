import pytest

from ezmeans import BatchLaw, Layout, Regroup, default_layout, law_recipe, paper_layout, work_layout
from ezmeans.layout import MAX_GROUPS, law_probes

EPOCHS = 2
SEED = 0
ROWS_PER_CLUSTER = 16
LAYOUT_SWEEP = 70_000


@pytest.mark.parametrize(
    ("groups", "probes"), [(1, 1), (4, 4), (16, 6), (64, 6), (128, 8), (256, 10), (512, 15), (1024, 20)]
)
def test_law_probes_grow_with_the_root_of_the_groups(groups: int, probes: int) -> None:
    assert law_probes(groups) == probes


@pytest.mark.parametrize("law", list(BatchLaw))
def test_work_layout_is_a_power_of_two_layout_under_the_cap(law: BatchLaw) -> None:
    for k in range(1, LAYOUT_SWEEP):
        layout = work_layout(ROWS_PER_CLUSTER * k, k, law_recipe(ROWS_PER_CLUSTER * k, k, EPOCHS, SEED, law))
        assert layout.groups & (layout.groups - 1) == 0
        assert 1 <= layout.probes == law_probes(layout.groups) <= layout.groups <= min(k, MAX_GROUPS)
        assert layout.regroup is Regroup.WARM


@pytest.mark.parametrize(
    ("k", "layout"),
    [
        (16, Layout(4, 4, Regroup.COLD)),
        (4096, Layout(64, 4, Regroup.COLD)),
        (16384, Layout(128, 8, Regroup.COLD)),
        (65536, Layout(256, 16, Regroup.COLD)),
    ],
)
def test_paper_layout_is_root_k_groups_with_a_sixteenth_probed(k: int, layout: Layout) -> None:
    assert paper_layout(k) == layout


@pytest.mark.parametrize(
    ("k", "layout"),
    [
        (1, Layout(1, 1, Regroup.COLD)),
        (16, Layout(4, 4, Regroup.COLD)),
        (4096, Layout(64, 6, Regroup.COLD)),
        (16384, Layout(128, 8, Regroup.COLD)),
        (65536, Layout(256, 16, Regroup.COLD)),
    ],
)
def test_default_layout_probes_at_least_six_groups(k: int, layout: Layout) -> None:
    assert default_layout(k) == layout
