import json
import numpy as np


class MotionPlan:
    def __init__(self, waypoints: np.ndarray, segment_times: np.ndarray):
        self.waypoints = np.array(waypoints, dtype=float)
        if self.waypoints.ndim == 1:
            self.waypoints = self.waypoints.reshape((1, -1))

        self.segment_times = np.array(segment_times, dtype=float)
        if self.segment_times.size == 0:
            self.segment_cum_times = np.array([0.0])
            self.traj_time = 0.0
        else:
            self.segment_cum_times = np.concatenate(([0.0], np.cumsum(self.segment_times)))
            self.traj_time = float(self.segment_cum_times[-1])

    @classmethod
    def from_json(cls, joint_dim=9, json_path=None):
        with open(json_path, "r") as f:
            plan = json.load(f)
        w = plan.get("waypoints")
        w = np.array(w, dtype=float)
        if w.shape[1] != joint_dim:
            raise ValueError(f"Each waypoint must have {joint_dim} values")
        waypoints = w
        st = np.array(plan.get("segment_times"), dtype=float)
        if st.size != max(0, len(waypoints) - 1):
            raise ValueError("segment_times must have length num_waypoints-1")
        segment_times = st
        return cls(waypoints, segment_times)


    def state_at(self, t: float):
        """Return joint positions and velocities at time t.

        Uses full-joint linear interpolation between the current segment waypoints.
        """
        q = np.array(self.waypoints[0], dtype=float)
        qd = np.zeros_like(q)

        if self.traj_time <= 0.0:
            return q, qd

        tt = min(max(t, 0.0), self.traj_time)
        # find segment index
        i = np.searchsorted(self.segment_cum_times, tt, side="right") - 1
        if i < 0:
            i = 0

        num_segments = len(self.segment_times)
        if i >= num_segments:
            # at or after final time
            q = np.array(self.waypoints[-1], dtype=float)
            return q, qd

        t0 = self.segment_cum_times[i]
        seg_dt = self.segment_times[i]
        local_t = tt - t0

        if seg_dt <= 0.0:
            alpha = 1.0
        else:
            alpha = min(max(local_t / seg_dt, 0.0), 1.0)

        start_q = np.array(self.waypoints[i], dtype=float)
        goal_q = np.array(self.waypoints[i + 1], dtype=float)

        q = start_q + alpha * (goal_q - start_q)

        if seg_dt > 0.0:
            qd = (goal_q - start_q) / seg_dt
        else:
            qd = np.zeros_like(q)

        return q, qd


__all__ = ["MotionPlan"]