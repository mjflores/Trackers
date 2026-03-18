# trackers/botsort_tracker.py
from .base_tracker import BaseBoxmotTracker

class BotSortTracker(BaseBoxmotTracker):
    _tracker_type = "botsort"
    _needs_reid   = True
