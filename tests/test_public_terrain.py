import pytest

from msr.geospatial.public_terrain import choose_terrain_tiles, terrain_tiles_for_bounds


def test_small_scene_uses_a_bounded_public_terrain_tile_set() -> None:
    tiles = terrain_tiles_for_bounds((9.0476, 41.9280, 9.0552, 41.9336), zoom=12)
    assert tiles == [(12, 2150, 1521), (12, 2151, 1521)]


def test_large_scene_is_rejected_before_network_access() -> None:
    with pytest.raises(ValueError, match="crop the image or attach a DEM"):
        terrain_tiles_for_bounds((-20.0, 20.0, 20.0, 60.0), zoom=12)


def test_larger_hilly_import_uses_a_coarser_bounded_source() -> None:
    tiles = choose_terrain_tiles((79.0, 30.0, 80.0, 31.0))
    assert 1 <= len(tiles) <= 16
    assert tiles[0][0] < 12
    assert all(tile[0] == tiles[0][0] for tile in tiles)


def test_small_import_preserves_preferred_detail() -> None:
    bounds = (9.0476, 41.9280, 9.0552, 41.9336)
    assert choose_terrain_tiles(bounds) == terrain_tiles_for_bounds(bounds)


def test_automatic_source_still_rejects_invalid_or_continental_extents() -> None:
    with pytest.raises(ValueError, match="Invalid WGS84"):
        choose_terrain_tiles((80, 30, 79, 31))
    with pytest.raises(ValueError, match="crop the image"):
        choose_terrain_tiles((-20.0, 20.0, 20.0, 60.0))
