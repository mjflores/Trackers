# trackers/strongsort_tracker.py
from .base_tracker import BaseBoxmotTracker

class StrongSortTracker(BaseBoxmotTracker):
    _tracker_type = "strongsort"
    _needs_reid   = True
