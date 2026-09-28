import numpy as np

from msr.inference.viewer_mesh import (
    prepare_building_mesh,
    prepare_presentation_mesh,
    prepare_semantic_scene,
    resolve_semantic_masks,
    vegetation_mask_from_rgb,
)


def test_prepare_building_mesh_removes_ground_and_flattens_roof() -> None:
    height = np.full((30, 30), 2.0, dtype=np.float32)
    height[8:22, 7:23] = np.linspace(7, 11, 16, dtype=np.float32)[None, :]
    probability = np.zeros((30, 30), dtype=np.float32)
    probability[8:22, 7:23] = 0.9
    probability[2, 2] = 1.0

    mesh, components = prepare_building_mesh(
        height,
        probability,
        minimum_component_pixels=20,
        roof_flatten=1.0,
    )

    assert components == 1
    assert np.all(mesh[:7] == 0)
    roof = mesh[8:22, 7:23]
    assert float(roof.max() - roof.min()) == 0.0
    assert float(roof.mean()) > 7.0


def test_presentation_mesh_adds_terrain_and_vegetation_without_changing_source() -> None:
    height = np.zeros((40, 40), dtype=np.float32)
    height[12:28, 12:28] = 8.0
    height[4:10, 3:11] = 3.0
    original = height.copy()
    probability = np.zeros_like(height)
    probability[12:28, 12:28] = 0.95
    rgb = np.zeros((3, 40, 40), dtype=np.float32)
    rgb[:] = 80
    rgb[0, 4:10, 3:11] = 35
    rgb[1, 4:10, 3:11] = 145
    rgb[2, 4:10, 3:11] = 45
    prior = np.tile(np.linspace(0, 1, 40, dtype=np.float32), (40, 1))

    mesh, counts = prepare_presentation_mesh(
        height,
        probability,
        rgb=rgb,
        terrain_prior=prior,
        minimum_component_pixels=20,
    )

    assert counts["building_components"] == 1
    assert counts["vegetation_pixels"] > 0
    assert float(mesh[6, 6]) > float(mesh[35, 1])
    assert float(np.ptp(mesh[0, :])) > 0
    np.testing.assert_array_equal(height, original)


def test_vegetation_mask_rejects_neutral_ground() -> None:
    rgb = np.full((3, 20, 20), 90, dtype=np.float32)
    rgb[0, 5:15, 5:15] = 30
    rgb[1, 5:15, 5:15] = 150
    rgb[2, 5:15, 5:15] = 40
    mask = vegetation_mask_from_rgb(rgb)
    assert mask[10, 10]
    assert not mask[1, 1]


def test_presentation_mesh_uses_rgb_only_with_model_support() -> None:
    height = np.full((30, 30), 6.0, dtype=np.float32)
    buildings = np.zeros_like(height)
    vegetation = np.zeros_like(height)
    vegetation[8:22, 8:22] = 0.15
    rgb = np.full((3, 30, 30), 90, dtype=np.float32)
    rgb[0, 8:22, 8:22] = 30
    rgb[1, 8:22, 8:22] = 150
    rgb[2, 8:22, 8:22] = 40

    mesh, counts = prepare_presentation_mesh(
        height,
        buildings,
        rgb=rgb,
        vegetation_probability=vegetation,
        minimum_component_pixels=20,
    )

    assert counts["vegetation_pixels"] > 0
    assert float(mesh[15, 15]) > 0
    assert float(mesh[2, 2]) == 0


def test_presentation_mesh_recovers_visible_green_when_router_is_conservative() -> None:
    height = np.full((40, 40), 0.1, dtype=np.float32)
    buildings = np.zeros_like(height)
    vegetation = np.full_like(height, 0.02)
    rgb = np.full((3, 40, 40), 90, dtype=np.float32)
    rgb[0, 8:32, 8:32] = 30
    rgb[1, 8:32, 8:32] = 150
    rgb[2, 8:32, 8:32] = 40

    mesh, counts = prepare_presentation_mesh(
        height,
        buildings,
        rgb=rgb,
        vegetation_probability=vegetation,
        minimum_component_pixels=20,
    )

    assert counts["vegetation_pixels"] > 400
    assert 1.0 < float(mesh[20, 20]) < 4.0
    np.testing.assert_array_equal(height, np.full((40, 40), 0.1, dtype=np.float32))


def test_semantic_scene_separates_ground_and_building_solid() -> None:
    height = np.zeros((60, 80), dtype=np.float32)
    height[15:40, 20:55] = 11.0
    probability = np.zeros_like(height)
    probability[15:40, 20:55] = 0.9
    prior = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (60, 1))

    ground, solids, counts = prepare_semantic_scene(
        height,
        probability,
        terrain_prior=prior,
        minimum_component_pixels=50,
    )

    assert counts["building_solids"] == 1
    assert len(solids) == 1
    assert abs(float(solids[0]["height_m"]) - 11.0) < 0.01
    assert 0.3 < float(solids[0]["center_x"]) < 0.6
    assert float(np.nanmax(ground[15:40, 20:55])) < 2.0


def test_semantic_masks_are_exclusive_and_protect_buildings_in_mixed_scene() -> None:
    height = np.full((48, 64), 0.5, dtype=np.float32)
    height[10:34, 8:30] = 12.0
    height[12:38, 28:54] = 5.0
    buildings = np.zeros_like(height)
    buildings[10:34, 8:30] = 0.85
    vegetation = np.zeros_like(height)
    vegetation[12:38, 28:54] = 0.9
    # Simulate an uncertain router overlap at the roof/canopy boundary.
    vegetation[12:30, 24:34] = 0.95
    rgb = np.full((3, 48, 64), 90, dtype=np.float32)
    rgb[0, 12:38, 28:54] = 30
    rgb[1, 12:38, 28:54] = 150
    rgb[2, 12:38, 28:54] = 40

    building_mask, vegetation_mask, semantic = resolve_semantic_masks(
        height,
        buildings,
        rgb=rgb,
        vegetation_probability=vegetation,
        minimum_building_pixels=20,
    )

    assert not np.any(building_mask & vegetation_mask)
    assert np.all(semantic[building_mask] == 1)
    assert np.all(semantic[vegetation_mask] == 2)
    assert semantic[20, 20] == 1
    assert semantic[24, 44] == 2
    assert semantic[45, 60] == 0
