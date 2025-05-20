import datetime

class CrowdAnalyzer:
    def __init__(self):
        self.crowd_counts = []

    def count_crowd(self, tracked_crowd, frame_number):
        count = len(tracked_crowd)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.crowd_counts.append({'frame': frame_number, 'time': timestamp, 'count': count})
        return count

    def get_summary(self):
        return self.crowd_counts
