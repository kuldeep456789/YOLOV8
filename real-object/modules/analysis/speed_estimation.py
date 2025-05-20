import math

class SpeedEstimator:
    def __init__(self):
        self.previous_positions = {}

    def estimate_speed(self, tracked_crowd):
        speeds = {}
        for person in tracked_crowd:
            pid = person['id']
            x1, y1, x2, y2 = person['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if pid in self.previous_positions:
                prev_center = self.previous_positions[pid]
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                pixel_distance = math.sqrt(dx**2 + dy**2)
                # Simulated scaling: assume 1 pixel = 0.05 meters, 30 FPS
                speed = (pixel_distance * 0.05) * 30 * 3.6  # km/h
                speeds[pid] = round(speed, 2)

            self.previous_positions[pid] = center
        return speeds
