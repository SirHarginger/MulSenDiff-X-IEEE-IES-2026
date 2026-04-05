from __future__ import annotations

ARCHETYPE_A_REPLACE = ("screw", "zipper")
ARCHETYPE_A_CONTROL = ("nut", "spring_pad")
ARCHETYPE_B_REWEIGHT = (
    "capsule",
    "solar_panel",
    "light",
    "button_cell",
    "screen",
)
ARCHETYPE_B_ACTIVE_GATE_MODES = {
    "capsule": "cross_inconsistency",
    "solar_panel": "thermal_hotspot",
}
ARCHETYPE_C_BASELINE = (
    "cotton",
    "flat_pad",
    "piggy",
    "cube",
    "toothbrush",
    "plastic_cylinder",
)

ALL_CATEGORIES = (
    "button_cell",
    "capsule",
    "cotton",
    "cube",
    "flat_pad",
    "light",
    "nut",
    "piggy",
    "plastic_cylinder",
    "screen",
    "screw",
    "solar_panel",
    "spring_pad",
    "toothbrush",
    "zipper",
)

LOCALIZATION_MIN_REGION_AREA_FRACTION_OVERRIDES = {
    "screw": 0.0002,
    "zipper": 0.0002,
}


def normalize_category_name(category_name: str) -> str:
    return str(category_name).strip().lower()


def get_descriptor_policy(category_name: str) -> str:
    name = normalize_category_name(category_name)
    if name in ARCHETYPE_A_REPLACE:
        return "replacement"
    if name in ARCHETYPE_B_REWEIGHT:
        return "reweighting"
    return "baseline"


def is_archetype_a_replace(category_name: str) -> bool:
    return normalize_category_name(category_name) in ARCHETYPE_A_REPLACE


def is_archetype_a_control(category_name: str) -> bool:
    return normalize_category_name(category_name) in ARCHETYPE_A_CONTROL


def is_archetype_b(category_name: str) -> bool:
    return normalize_category_name(category_name) in ARCHETYPE_B_REWEIGHT


def resolve_archetype_b_gate_mode(category_name: str) -> str:
    name = normalize_category_name(category_name)
    if name not in ARCHETYPE_B_REWEIGHT:
        return "inactive"
    return ARCHETYPE_B_ACTIVE_GATE_MODES.get(name, "monitor_only")


def is_archetype_b_active_rescue(category_name: str) -> bool:
    return resolve_archetype_b_gate_mode(category_name) in {"cross_inconsistency", "thermal_hotspot"}


def resolve_localization_min_region_area_fraction(
    category_name: str,
    default_value: float,
) -> float:
    name = normalize_category_name(category_name)
    override = LOCALIZATION_MIN_REGION_AREA_FRACTION_OVERRIDES.get(name)
    if override is None:
        return float(default_value)
    return min(float(default_value), float(override))
