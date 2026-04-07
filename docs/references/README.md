# Trusted References

Place approved manuals, process references, and other trusted explanation documents here.

Each source file must have a sidecar metadata file with the same stem and suffix `.meta.json`.

Example:

- `capsule_internal_defects.md`
- `capsule_internal_defects.meta.json`

Required metadata fields:

```json
{
  "doc_id": "capsule_internal_defects_v1",
  "title": "Capsule Internal Defect Guidance",
  "source_type": "manual",
  "trust_level": "approved",
  "category_tags": ["capsule"],
  "sensor_tags": ["cross_inconsistency", "thermal_hotspot"],
  "defect_tags": ["internal_defect_profile", "cross_modal_inconsistency"],
  "process_tags": ["seal_check", "assembly_review"]
}
```
