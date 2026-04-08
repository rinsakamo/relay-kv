from relaykv import TierManager, build_blocks


def test_tier_split():
    tm = TierManager(hot_window=1024)
    split = tm.split_range(4096)

    assert split.cold_range == (0, 3072)
    assert split.hot_range == (3072, 4096)


def test_build_blocks():
    blocks = build_blocks((0, 3072), block_size=128)

    assert len(blocks) == 24
    assert blocks[0].start == 0
    assert blocks[0].end == 128
    assert blocks[-1].start == 2944
    assert blocks[-1].end == 3072