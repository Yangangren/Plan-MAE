from typing import Any, Dict

import numpy as np

from .nuplan_feature_builder import NuplanFeatureBuilder


class SSLNuplanFeatureBuilder(NuplanFeatureBuilder):
    """Builds the same planTF features and augments them with SSL masks/targets."""

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Use a dedicated cache key so SSL features don't collide with supervised ones."""
        return "feature_ssl"

    def __init__(
        self,
        radius: float = 100,
        history_horizon: float = 2,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        max_agents: int = 64,
        agent_history_mask_ratio: float = 0.3,
        map_mask_ratio: float = 0.3,
        route_mask_ratio: float = 0.3,
        mask_ego_history: bool = False,
        min_agent_history: int = 4,
        min_map_points: int = 4,
    ) -> None:
        super().__init__(
            radius=radius,
            history_horizon=history_horizon,
            future_horizon=future_horizon,
            sample_interval=sample_interval,
            max_agents=max_agents,
        )
        self.agent_history_mask_ratio = agent_history_mask_ratio
        self.map_mask_ratio = map_mask_ratio
        self.route_mask_ratio = route_mask_ratio
        self.mask_ego_history = mask_ego_history
        self.min_agent_history = min_agent_history
        self.min_map_points = min_map_points

    def _build_feature(self, *args, **kwargs):
        feature = super()._build_feature(*args, **kwargs)
        feature.data["ssl"] = self._apply_ssl_masking(feature.data)
        return feature

    def _apply_ssl_masking(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        ssl_data: Dict[str, np.ndarray] = {}
        hist_steps = self.history_samples + 1

        # Always preserve a compact copy of the raw input so training can
        # visualize original vs masked scenes from cached features.
        ssl_data.update(self._copy_debug_data(data, hist_steps))

        ssl_data.update(self._mask_agent_history(data["agent"], hist_steps))
        ssl_data.update(self._mask_map_points(data["map"]))
        ssl_data.update(self._mask_route_flags(data["map"]))

        return ssl_data

    def _mask_agent_history(
        self, agent_data: Dict[str, np.ndarray], hist_steps: int
    ) -> Dict[str, np.ndarray]:
        valid_history = agent_data["valid_mask"][:, :hist_steps]
        position_history = agent_data["position"][:, :hist_steps]
        heading_history = agent_data["heading"][:, :hist_steps]
        velocity_history = agent_data["velocity"][:, :hist_steps]
        shape_history = agent_data["shape"][:, :hist_steps]

        valid_mask = valid_history.copy()
        candidate_mask = valid_mask.copy()
        if not self.mask_ego_history and candidate_mask.shape[0] > 0:
            candidate_mask[0] = False
        if candidate_mask.shape[1] > 0:
            # Keep the current timestep visible for agent token anchoring.
            candidate_mask[:, -1] = False

        history_mask = self._sample_segment_masks(
            candidate_mask,
            ratio=self.agent_history_mask_ratio,
            min_valid_points=self.min_agent_history,
        )

        position_gt = np.zeros_like(position_history)
        heading_gt = np.zeros_like(heading_history)
        velocity_gt = np.zeros_like(velocity_history)
        shape_gt = np.zeros_like(shape_history)

        position_gt[history_mask] = position_history[history_mask]
        heading_gt[history_mask] = heading_history[history_mask]
        velocity_gt[history_mask] = velocity_history[history_mask]
        shape_gt[history_mask] = shape_history[history_mask]

        position_history[history_mask] = 0.0
        heading_history[history_mask] = 0.0
        velocity_history[history_mask] = 0.0
        shape_history[history_mask] = 0.0
        valid_history[history_mask] = False

        return {
            "agent_hist_mask": history_mask,
            "agent_position_gt": position_gt,
            "agent_heading_gt": heading_gt,
            "agent_velocity_gt": velocity_gt,
            "agent_shape_gt": shape_gt,
        }

    def _mask_map_points(
        self, map_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        point_position = map_data["point_position"][:, 0]
        point_vector = map_data["point_vector"][:, 0]
        point_orientation = map_data["point_orientation"][:, 0]
        point_valid_mask = map_data["valid_mask"]

        point_mask = self._sample_segment_masks(
            point_valid_mask.copy(),
            ratio=self.map_mask_ratio,
            min_valid_points=self.min_map_points,
        )

        point_position_gt = np.zeros_like(point_position)
        point_vector_gt = np.zeros_like(point_vector)
        point_orientation_gt = np.zeros_like(point_orientation)

        point_position_gt[point_mask] = point_position[point_mask]
        point_vector_gt[point_mask] = point_vector[point_mask]
        point_orientation_gt[point_mask] = point_orientation[point_mask]

        point_position[point_mask] = 0.0
        point_vector[point_mask] = 0.0
        point_orientation[point_mask] = 0.0
        point_valid_mask[point_mask] = False

        return {
            "map_point_mask": point_mask,
            "map_point_position_gt": point_position_gt,
            "map_point_vector_gt": point_vector_gt,
            "map_point_orientation_gt": point_orientation_gt,
        }

    def _mask_route_flags(
        self, map_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        valid_polygons = map_data["valid_mask"].any(axis=-1)
        route_mask = np.zeros_like(valid_polygons, dtype=bool)

        candidate_indices = np.flatnonzero(valid_polygons)
        if self.route_mask_ratio > 0 and len(candidate_indices) > 0:
            num_mask = int(np.floor(len(candidate_indices) * self.route_mask_ratio))
            num_mask = min(len(candidate_indices), max(1, num_mask))
            selected = np.random.choice(candidate_indices, size=num_mask, replace=False)
            route_mask[selected] = True

        route_gt = map_data["polygon_on_route"].copy()
        map_data["polygon_on_route"][route_mask] = False

        return {
            "route_mask": route_mask,
            "route_gt": route_gt,
        }

    @staticmethod
    def _copy_debug_data(data: Dict[str, Any], hist_steps: int) -> Dict[str, Any]:
        return {
            "debug_raw_agent_position": data["agent"]["position"][:, :hist_steps].copy(),
            "debug_raw_agent_valid_mask": data["agent"]["valid_mask"][
                :, :hist_steps
            ].copy(),
            "debug_raw_map_point_position": data["map"]["point_position"][:, 0].copy(),
            "debug_raw_map_valid_mask": data["map"]["valid_mask"].copy(),
            "debug_raw_polygon_on_route": data["map"]["polygon_on_route"].copy(),
        }

    @staticmethod
    def _sample_segment_masks(
        valid_mask: np.ndarray, ratio: float, min_valid_points: int
    ) -> np.ndarray:
        mask = np.zeros_like(valid_mask, dtype=bool)
        if ratio <= 0:
            return mask

        for row_idx in range(valid_mask.shape[0]):
            valid_indices = np.flatnonzero(valid_mask[row_idx])
            valid_count = len(valid_indices)

            if valid_count < max(2, min_valid_points):
                continue

            mask_len = int(np.floor(valid_count * ratio))
            mask_len = min(valid_count - 1, max(1, mask_len))
            if mask_len <= 0:
                continue

            start = np.random.randint(0, valid_count - mask_len + 1)
            masked_indices = valid_indices[start : start + mask_len]
            mask[row_idx, masked_indices] = True

        return mask
