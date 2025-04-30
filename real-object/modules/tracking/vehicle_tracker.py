from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None
        )

    def update_tracks(self, detections, frame):
        # Convert to format: [[x1, y1, x2, y2, confidence], ...]
        track_inputs = [det for det in detections]

        tracks = self.tracker.update_tracks(track_inputs, frame=frame)

        tracked_vehicles = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            tracked_vehicles.append({
                'id': track_id,
                'bbox': ltrb  # [x1, y1, x2, y2]
            })
        return tracked_vehicles
