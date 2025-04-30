import datetime

class TrafficAnalyzer:
    def __init__(self):
        self.vehicle_counts = []

    def count_vehicles(self, tracked_vehicles, frame_number):
        count = len(tracked_vehicles)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.vehicle_counts.append({'frame': frame_number, 'time': timestamp, 'count': count})
        return count

    def get_summary(self):
        return self.vehicle_counts
