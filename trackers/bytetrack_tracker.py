# trackers/bytetrack_tracker.py
from .base_tracker import BaseBoxmotTracker

class ByteTrackTracker(BaseBoxmotTracker):
    _tracker_type = "bytetrack"
    _needs_reid   = False
