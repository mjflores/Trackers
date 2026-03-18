# strongsort_tracker.py
from .base_tracker import BaseBoxmotTracker

class StrongSortTracker(BaseBoxmotTracker):
    _tracker_type = "strongsort"
    _needs_reid   = True


# bytetrack_tracker.py
from .base_tracker import BaseBoxmotTracker

class ByteTrackTracker(BaseBoxmotTracker):
    _tracker_type = "bytetrack"
    _needs_reid   = False


# botsort_tracker.py
from .base_tracker import BaseBoxmotTracker

class BotSortTracker(BaseBoxmotTracker):
    _tracker_type = "botsort"
    _needs_reid   = True


# ocsort_tracker.py
from .base_tracker import BaseBoxmotTracker

class OcSortTracker(BaseBoxmotTracker):
    _tracker_type = "ocsort"
    _needs_reid   = False
