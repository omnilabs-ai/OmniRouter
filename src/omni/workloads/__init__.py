"""Workloads: fingerprinting, segmentation, stats and merges (design doc section 2.2)."""

from omni.workloads.fingerprint import Fingerprint, auto_name, fingerprint, mask_system_prompt
from omni.workloads.service import WorkloadResolver, merge, refresh, segment

__all__ = [
    "Fingerprint",
    "WorkloadResolver",
    "auto_name",
    "fingerprint",
    "mask_system_prompt",
    "merge",
    "refresh",
    "segment",
]
