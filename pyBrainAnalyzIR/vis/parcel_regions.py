"""Shared, hemisphere-specific coarse ROIs for the Schaefer atlas parcels.

Membership follows anatomical name families, not the functional-network prefix
or atlas-specific lobe assignments. These broad ROIs are not anatomical masks:
in particular, sensorimotor parcels are not split into primary motor and sensory
cortex. Unknown parcel names remain available as individual regions.
"""
from __future__ import annotations

from html import escape
import re
from typing import Iterable


# (region, member-family suffixes, description)
REGIONS = (
    ("Sensorimotor cortex", ("SomMotA", "SomMotB_Cent"),
     "Precentral and postcentral gyri on either side of the central sulcus, containing primary motor "
     "cortex (BA 4) and primary somatosensory cortex (BA 3, 1, 2). Somatotopically organized; executes "
     "voluntary movement and processes touch and proprioception."),
    ("Postcentral / somatosensory association cortex", ("PostC",),
     "Postcentral gyrus and adjoining anterior parietal cortex. Processes tactile and proprioceptive "
     "information and relays it to posterior parietal areas for higher-order somatosensory functions "
     "such as shape and texture discrimination."),
    ("Lateral premotor / frontal eye-field region", ("FEF", "PrCv", "PrC"),
     "Caudal lateral frontal cortex (BA 6 and 8) anterior to primary motor cortex. Premotor cortex "
     "contributes to selecting, preparing and guiding movements, particularly in response to external "
     "cues; the frontal eye fields control voluntary saccades and shifts of visuospatial attention."),
    ("Medial frontal / SMA vicinity", ("FrMed",),
     "Superior frontal gyrus near the midline, including the supplementary and pre-supplementary motor "
     "areas (medial BA 6). Involved in internally generated and sequential movements, bimanual "
     "coordination and movement preparation."),
    ("Secondary somatosensory / parietal opercular region", ("S2", "ParOper"),
     "Parietal operculum on the upper bank of the lateral (Sylvian) fissure, containing secondary "
     "somatosensory cortex (SII). Contributes to higher-order tactile processing, integration of input "
     "from both sides of the body, and pain processing."),
    ("Dorsal / lateral prefrontal cortex", ("PFCd", "PFCl", "PFCld"),
     "Superior and middle frontal gyri (approximately BA 9 and 46). Associated with executive functions "
     "such as working memory, planning, maintaining task rules, and flexible goal-directed control of "
     "attention and behavior."),
    ("Ventral / inferior prefrontal cortex", ("PFClv", "PFCv"),
     "Inferior frontal gyrus and adjacent ventrolateral prefrontal cortex (approximately BA 44, 45 and "
     "47). Associated with response selection and inhibition and with controlled retrieval of "
     "information from memory; in the left hemisphere it includes regions important for language."),
    ("Frontal opercular cortex", ("FrOper",),
     "Frontal operculum, the part of the inferior frontal gyrus overlying the anterior insula at the "
     "lateral fissure. Contributes to speech and orofacial motor control and, with the adjacent anterior "
     "insula, to taste and cognitive control."),
    ("Medial prefrontal cortex", ("PFCm", "PFCmp"),
     "Medial surface of the frontal lobe anterior to the motor areas (approximately medial BA 8, 9, 10 "
     "and 32). Associated with self-referential and social cognition, emotion regulation and value-based "
     "decision making; a core node of the default mode network."),
    ("Orbitofrontal cortex", ("OFC",),
     "Ventral surface of the frontal lobe above the orbits (approximately BA 11, 13 and 47). Represents "
     "the expected value of rewards and outcomes and supports emotionally guided decision making and "
     "learning when contingencies change."),
    ("Superior parietal / intraparietal cortex", ("SPL", "IPS"),
     "Dorsal posterior parietal cortex: the superior parietal lobule (BA 5 and 7) and the intraparietal "
     "sulcus. Integrates visual and somatosensory information for spatial attention, visually guided "
     "reaching and eye movements, and working memory; the intraparietal sulcus is also associated with "
     "numerical processing."),
    ("Inferior parietal cortex", ("IPL",),
     "Supramarginal and angular gyri (BA 40 and 39). Integrates information across sensory modalities "
     "and is associated with spatial attention, planning of skilled actions, phonological and semantic "
     "language processing, and number processing, with differences between the hemispheres."),
    ("Temporoparietal / posterior temporal region", ("TempPar",),
     "Junction of the posterior superior temporal sulcus with the inferior parietal lobule. Associated "
     "with reorienting attention toward unexpected stimuli and with social cognition, including "
     "perspective taking; on the left, posterior temporal cortex also supports language comprehension."),
    ("Lateral temporal association cortex", ("Temp", "AntTemp"),
     "Middle and inferior temporal gyri and the superior temporal sulcus. Associated with semantic "
     "memory and language comprehension, recognition of objects and faces along its inferior extent, and "
     "perception of voices and biological motion along the superior temporal sulcus."),
    ("Auditory / superior temporal region", ("Aud",),
     "Superior temporal gyrus, including primary auditory cortex on Heschl's gyrus within the lateral "
     "fissure and the surrounding auditory association cortex. Processes sound, including pitch, speech "
     "and music; the left posterior superior temporal gyrus is part of the classic receptive language "
     "areas."),
    ("Temporal pole", ("TempPole",),
     "Anterior tip of the temporal lobe (approximately BA 38). Associated with semantic memory, "
     "particularly for people and other unique entities, and with social and emotional processing "
     "through its connections with the orbitofrontal cortex and amygdala."),
    ("Visual / occipital cortex", ("VisCent", "VisPeri"),
     "Occipital lobe, including primary visual cortex along the calcarine sulcus (V1, BA 17) and early "
     "visual areas (BA 18 and 19). Retinotopically organized; processes basic visual features such as "
     "contrast, orientation, color and motion."),
    ("Lateral occipitotemporal / parieto-occipital cortex", ("TempOcc", "ParOcc"),
     "Higher-order visual cortex where the occipital lobe meets the temporal and parietal lobes. Lateral "
     "occipitotemporal portions support object, body and motion perception (including MT/V5); parieto- "
     "occipital portions support spatial vision and visually guided action."),
    ("Medial parietal / precuneus region", ("pCun", "ParMed"),
     "Medial surface of the parietal lobe (medial BA 7). Associated with visuospatial imagery, episodic "
     "memory retrieval and self-referential processing; a major hub of the default mode network."),
    ("Posterior cingulate / retrosplenial region", ("pCunPCC", "Cingp", "Rsp"),
     "Posterior cingulate gyrus and retrosplenial cortex behind the splenium of the corpus callosum (BA "
     "23, 31, 29 and 30). A central hub of the default mode network, associated with internally directed "
     "thought and autobiographical memory; retrosplenial cortex contributes to spatial navigation and "
     "memory for context."),
    ("Anterior / midcingulate region", ("Cingm", "Cinga"),
     "Anterior and middle cingulate gyrus on the medial surface above the corpus callosum (BA 24, 32 and "
     "33). Associated with emotion and autonomic regulation, pain, monitoring of conflict and errors, "
     "and adjusting behavior in response."),
    ("Insular cortex", ("Ins",),
     "Cortex folded within the lateral fissure beneath the frontal, parietal and temporal opercula. "
     "Associated with interoception (awareness of internal bodily states), taste, pain, emotional "
     "awareness and autonomic regulation; the anterior insula is a key node of the salience network."),
    ("Parahippocampal cortex", ("PHC",),
     "Parahippocampal gyrus of the medial temporal lobe, adjacent to the hippocampus. Supports memory "
     "encoding and retrieval, particularly for scenes, places and spatial context, and provides cortical "
     "input to the hippocampal formation."),
)
_BY_FAMILY = {
    family: (name, description)
    for name, families, description in REGIONS for family in families
}
_LABEL = re.compile(r"^(?P<family>.+)_\d+_(?P<side>LH|RH)$")


def _region(label: str) -> tuple[str, str] | None:
    match = _LABEL.fullmatch(label)
    if match is None:
        return None
    family = match["family"]
    key = family if family in ("SomMotA", "SomMotB_Cent", "TempPar") else family.rsplit("_", 1)[-1]
    if family.startswith(("VisCent_", "VisPeri_")):
        key = family.split("_", 1)[0]
    region = _BY_FAMILY.get(key)
    if region is None:
        return None
    side = "Left" if match["side"] == "LH" else "Right"
    return f"{side} {region[0]}", region[1]


def larger_region(label: str) -> str:
    region = _region(str(label))
    return region[0] if region is not None else str(label)


# One distinct color per REGIONS entry, shared by the left and right
# hemispheres so homologous regions are easy to compare.
_REGION_COLORS = (
    "#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#ffe119", "#808080", "#469990",
    "#dcbeff", "#a9a9a9",
)


def region_color(name: str) -> tuple[float, float, float] | None:
    """Hemisphere-independent RGB color of a larger region, or None if unknown."""
    base = re.sub(r"^(Left|Right) ", "", str(name))
    for index, (region, _, _) in enumerate(REGIONS):
        if region == base:
            value = _REGION_COLORS[index % len(_REGION_COLORS)]
            return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return None


def region_members(labels: Iterable[str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, set[str]] = {}
    for label in labels:
        label = str(label)
        if label and not label.startswith("Background"):
            groups.setdefault(larger_region(label), set()).add(label)
    return {name: tuple(sorted(members)) for name, members in sorted(groups.items())}


def parcel_information(label: str, labels: Iterable[str], subregions: bool) -> str:
    """Escaped HTML for the scrollable brain hover panel."""
    groups = region_members(labels)
    parent = larger_region(label) if subregions else label
    members = groups.get(parent, (label,))
    definition = _region(members[0])
    description = (
        definition[1] if definition is not None
        else "No larger-region mapping is defined for this parcel; its original label is retained."
    )
    relationship = (
        f"<p><b>Larger region:</b> {escape(parent)}</p>"
        if subregions else
        f"<p><b>Sub-region parcellations ({len(members)}):</b><br>"
        + "<br>".join(escape(member) for member in members) + "</p>"
    )
    return (
        f"<b>{escape(label)}</b><p><b>Description:</b> {escape(description)}</p>"
        + relationship
    )


def add_parcellation_actions(menu, subregions: bool, callback):
    """Add the same exclusive parcellation-level choices to either GUI."""
    from PySide6.QtGui import QActionGroup

    group = QActionGroup(menu)
    group.setExclusive(True)
    for text, value in (
        ("Show Parcellations", False), ("Show Sub-Region Parcellations", True),
    ):
        action = menu.addAction(text)
        action.setCheckable(True)
        action.setChecked(value == subregions)
        group.addAction(action)
        action.triggered.connect(lambda checked=False, level=value: callback(level))
    return group
