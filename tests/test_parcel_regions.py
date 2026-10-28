"""Coarse anatomical membership shared by the surface and depth maps."""
import pytest

# Importing pyBrainAnalyzIR loads its Cedalion-backed data classes.
pytest.importorskip("cedalion")

from pyBrainAnalyzIR.vis.parcel_regions import (  # noqa: E402
    REGIONS, larger_region, parcel_information, region_color, region_members,
)

pytestmark = pytest.mark.requires_cedalion


@pytest.mark.parametrize("family,region", [
    (family, region) for region, families, _ in REGIONS for family in families
])
def test_family_mapping_is_hemisphere_specific(family, region):
    family = (
        family if family in ("SomMotA", "SomMotB_Cent", "TempPar")
        else f"{family}_ExStr" if family in ("VisCent", "VisPeri")
        else f"Network_{family}"
    )
    assert larger_region(f"{family}_1_LH") == f"Left {region}"
    assert larger_region(f"{family}_23_RH") == f"Right {region}"


def test_membership_information_preserves_unknown_labels_and_escapes_html():
    labels = ["SomMotA_1_LH", "SomMotB_Cent_2_LH", "SomMotA_1_RH", "unknown<parcel>"]
    groups = region_members(labels + ["Background_LH"])
    assert groups["Left Sensorimotor cortex"] == ("SomMotA_1_LH", "SomMotB_Cent_2_LH")
    assert groups["unknown<parcel>"] == ("unknown<parcel>",)
    coarse = parcel_information("Left Sensorimotor cortex", labels, False)
    assert "Sub-region parcellations (2)" in coarse
    assert "SomMotB_Cent_2_LH" in coarse
    assert "central sulcus" in coarse
    fine = parcel_information("SomMotA_1_LH", labels, True)
    assert "Larger region:" in fine and "Left Sensorimotor cortex" in fine
    unknown = parcel_information("unknown<parcel>", labels, True)
    assert "&lt;parcel&gt;" in unknown and "No larger-region mapping" in unknown


def test_region_colors_are_shared_by_hemispheres_and_distinct_between_regions():

    colors = [region_color(f"Left {name}") for name, _, _ in REGIONS]
    assert colors == [region_color(f"Right {name}") for name, _, _ in REGIONS]
    assert len(set(colors)) == len(REGIONS)
    assert all(0 <= value <= 1 for color in colors for value in color)
    assert region_color("Custom_parcel") is None
