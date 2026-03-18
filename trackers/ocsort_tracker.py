# trackers/ocsort_tracker.py
from .base_tracker import BaseBoxmotTracker

class OcSortTracker(BaseBoxmotTracker):
    _tracker_type = "ocsort"
    _needs_reid   = False
