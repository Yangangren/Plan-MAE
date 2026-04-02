import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        print_ssl_losses: bool = False,
        ssl_loss_print_interval: int = 50,
        plot_debug_batches: bool = False,
        debug_plot_dir: str = "debug/ssl_train_plots",
        debug_plot_interval: int = 100,
        debug_plot_max_batches: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.print_ssl_losses = print_ssl_losses
        self.ssl_loss_print_interval = ssl_loss_print_interval
        self.plot_debug_batches = plot_debug_batches
        self.debug_plot_dir = debug_plot_dir
        self.debug_plot_interval = debug_plot_interval
        self.debug_plot_max_batches = debug_plot_max_batches
        self._debug_plot_count = 0

    def _get_feature_data(self, features: FeaturesType):
        feature_builders = self.model.get_list_of_required_feature()
        if len(feature_builders) != 1:
            raise KeyError(
                f"Expected exactly one feature builder, got {len(feature_builders)}."
            )

        feature_key = feature_builders[0].get_feature_unique_name()
        if feature_key not in features:
            available = ", ".join(sorted(features.keys()))
            raise KeyError(
                f"Missing feature key `{feature_key}` in batch. Available keys: {available}"
            )

        return features[feature_key].data

    def on_fit_start(self) -> None:
        if getattr(self.model, "pretrain_ssl", False):
            self.metrics = {}
            return

        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, _, _ = batch
        feature_data = self._get_feature_data(features)
        self._last_feature_data = feature_data
        res = self.forward(feature_data)

        losses = self._compute_objectives(res, feature_data)
        metrics = self._compute_metrics(res, feature_data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"]

    @staticmethod
    def _masked_regression_loss(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if not mask.any():
            return prediction.new_zeros(())
        return F.smooth_l1_loss(prediction[mask], target[mask])

    @staticmethod
    def _masked_bce_loss(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if not mask.any():
            return prediction.new_zeros(())
        return F.binary_cross_entropy_with_logits(prediction[mask], target[mask])

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        if getattr(self.model, "pretrain_ssl", False):
            if "ssl" not in data:
                raise KeyError(
                    "Missing `data['ssl']` for SSL pretraining. This usually means "
                    "the run loaded cached features from the plain NuplanFeatureBuilder "
                    "instead of SSLNuplanFeatureBuilder. Rebuild the cache with the SSL "
                    "config, or use a separate cache path / disable cache-only mode."
                )
            ssl_pred = res["ssl"]
            ssl_target = data["ssl"]

            agent_target = torch.cat(
                [
                    ssl_target["agent_position_gt"],
                    torch.stack(
                        [
                            ssl_target["agent_heading_gt"].cos(),
                            ssl_target["agent_heading_gt"].sin(),
                        ],
                        dim=-1,
                    ),
                    ssl_target["agent_velocity_gt"],
                    ssl_target["agent_shape_gt"],
                ],
                dim=-1,
            )
            map_target = torch.cat(
                [
                    ssl_target["map_point_position_gt"],
                    ssl_target["map_point_vector_gt"],
                    torch.stack(
                        [
                            ssl_target["map_point_orientation_gt"].cos(),
                            ssl_target["map_point_orientation_gt"].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )
            route_target = ssl_target["route_gt"].float()
            ego_target = agent_target[:, 0]
            agent_target = agent_target[:, 1:]

            ego_ssl_loss = self._masked_regression_loss(
                ssl_pred["ego_reconstruction"],
                ego_target,
                ssl_target["agent_hist_mask"][:, 0],
            )
            agent_ssl_loss = self._masked_regression_loss(
                ssl_pred["agent_reconstruction"],
                agent_target,
                ssl_target["agent_hist_mask"][:, 1:],
            )
            map_ssl_loss = self._masked_regression_loss(
                ssl_pred["map_reconstruction"],
                map_target,
                ssl_target["map_point_mask"],
            )
            route_ssl_loss = self._masked_bce_loss(
                ssl_pred["route_logits"],
                route_target,
                ssl_target["route_mask"],
            )

            loss = (
                self.model.ssl_ego_weight * ego_ssl_loss
                + self.model.ssl_agent_weight * agent_ssl_loss
                + self.model.ssl_map_weight * map_ssl_loss
                + self.model.ssl_route_weight * route_ssl_loss
            )

            return {
                "loss": loss,
                "ego_ssl_loss": ego_ssl_loss,
                "agent_ssl_loss": agent_ssl_loss,
                "map_ssl_loss": map_ssl_loss,
                "route_ssl_loss": route_ssl_loss,
            }

        trajectory, probability, prediction = (
            res["trajectory"],
            res["probability"],
            res["prediction"],
        )
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]

        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )

        loss = ego_reg_loss + ego_cls_loss + agent_reg_loss

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss,
            "cls_loss": ego_cls_loss,
            "prediction_loss": agent_reg_loss,
        }

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        if getattr(self.model, "pretrain_ssl", False):
            return None
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if (
            getattr(self.model, "pretrain_ssl", False)
            and self.print_ssl_losses
            and prefix == "train"
            and self.global_rank == 0
            and self.global_step % max(1, self.ssl_loss_print_interval) == 0
        ):
            printable = []
            for key, value in objectives.items():
                if torch.is_tensor(value):
                    printable.append(f"{key}={value.detach().item():.6f}")
            self.print(
                f"[ssl train step {self.global_step}] " + ", ".join(printable)
            )

        if (
            getattr(self.model, "pretrain_ssl", False)
            and self.plot_debug_batches
            and prefix == "train"
            and self.global_rank == 0
            and self.global_step % max(1, self.debug_plot_interval) == 0
        ):
            self._plot_debug_batch(prefix)

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def _plot_debug_batch(self, prefix: str) -> None:
        data = getattr(self, "_last_feature_data", None)
        if data is None or "ssl" not in data:
            return

        ssl_data = data["ssl"]
        required = [
            "raw_agent_position",
            "raw_agent_valid_mask",
            "raw_map_point_position",
            "raw_map_valid_mask",
            "raw_polygon_on_route",
        ]
        if any(key not in ssl_data for key in required):
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        os.makedirs(self.debug_plot_dir, exist_ok=True)
        batch_size = data["agent"]["position"].shape[0]

        for sample_idx in range(batch_size):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            self._draw_original_scene(
                axes[0],
                ssl_data["raw_agent_position"][sample_idx].detach().cpu(),
                ssl_data["raw_agent_valid_mask"][sample_idx].detach().cpu(),
                ssl_data["raw_map_point_position"][sample_idx].detach().cpu(),
                ssl_data["raw_map_valid_mask"][sample_idx].detach().cpu(),
                ssl_data["raw_polygon_on_route"][sample_idx].detach().cpu(),
                "Original Input",
            )
            self._draw_masked_scene(
                axes[1],
                data["agent"]["position"][sample_idx, :, : self.model.history_steps]
                .detach()
                .cpu(),
                data["agent"]["valid_mask"][sample_idx, :, : self.model.history_steps]
                .detach()
                .cpu(),
                data["map"]["point_position"][sample_idx, :, 0].detach().cpu(),
                data["map"]["valid_mask"][sample_idx].detach().cpu(),
                data["map"]["polygon_on_route"][sample_idx].detach().cpu(),
                ssl_data["agent_position_gt"][sample_idx].detach().cpu(),
                ssl_data["agent_hist_mask"][sample_idx].detach().cpu(),
                ssl_data["map_point_position_gt"][sample_idx].detach().cpu(),
                ssl_data["map_point_mask"][sample_idx].detach().cpu(),
                ssl_data["route_gt"][sample_idx].detach().cpu(),
                ssl_data["route_mask"][sample_idx].detach().cpu(),
                "Masked Input",
            )
            fig.tight_layout()
            filename = os.path.join(
                self.debug_plot_dir,
                (
                    f"{prefix}_ssl_step_{self.global_step:06d}_"
                    f"plot_{self._debug_plot_count:04d}_sample_{sample_idx:02d}.png"
                ),
            )
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            self._debug_plot_count += 1

    @staticmethod
    def _draw_original_scene(
        ax,
        agent_position: torch.Tensor,
        agent_valid_mask: torch.Tensor,
        map_point_position: torch.Tensor,
        map_valid_mask: torch.Tensor,
        polygon_on_route: torch.Tensor,
        title: str,
    ) -> None:
        for lane_idx in range(map_point_position.shape[0]):
            valid_points = map_valid_mask[lane_idx].bool()
            if not valid_points.any():
                continue
            lane_points = map_point_position[lane_idx, valid_points]
            color = "tab:orange" if polygon_on_route[lane_idx] > 0 else "lightgray"
            zorder = 1 if polygon_on_route[lane_idx] > 0 else 0
            ax.scatter(lane_points[:, 0], lane_points[:, 1], color=color, 
                       s=8, alpha=0.9, zorder=zorder)

        for agent_idx in range(agent_position.shape[0]):
            valid_steps = agent_valid_mask[agent_idx].bool()
            if not valid_steps.any():
                continue
            traj = agent_position[agent_idx, valid_steps]
            color = "tab:red" if agent_idx == 0 else "tab:blue"
            zorder = 10 if agent_idx == 0 else 5
            ax.scatter(traj[:, 0], traj[:, 1], color=color, s=12, alpha=0.9, zorder=zorder)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=24, edgecolors="black", zorder=zorder)

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlim([-30, 70])
        ax.set_ylim([-40, 40])

    @staticmethod
    def _draw_masked_scene(
        ax,
        agent_position: torch.Tensor,
        agent_valid_mask: torch.Tensor,
        map_point_position: torch.Tensor,
        map_valid_mask: torch.Tensor,
        polygon_on_route: torch.Tensor,
        agent_position_gt: torch.Tensor,
        agent_hist_mask: torch.Tensor,
        map_point_position_gt: torch.Tensor,
        map_point_mask: torch.Tensor,
        route_gt: torch.Tensor,
        route_mask: torch.Tensor,
        title: str,
    ) -> None:
        for lane_idx in range(map_point_position.shape[0]):
            visible_points = map_valid_mask[lane_idx].bool()
            masked_points = map_point_mask[lane_idx].bool()

            if visible_points.any():
                lane_points = map_point_position[lane_idx, visible_points]
                color = "tab:orange" if polygon_on_route[lane_idx] > 0 else "lightgray"
                ax.scatter(
                    lane_points[:, 0], lane_points[:, 1], color=color, s=8, alpha=0.9
                )

            if masked_points.any():
                masked_lane_points = map_point_position_gt[lane_idx, masked_points]
                masked_color = "gold" if route_gt[lane_idx] > 0 else "magenta"
                marker = "x" if route_mask[lane_idx] else "o"
                ax.scatter(
                    masked_lane_points[:, 0],
                    masked_lane_points[:, 1],
                    color=masked_color,
                    s=18,
                    alpha=0.95,
                    marker=marker,
                )

        for agent_idx in range(agent_position.shape[0]):
            visible_steps = agent_valid_mask[agent_idx].bool()
            masked_steps = agent_hist_mask[agent_idx].bool()

            if visible_steps.any():
                traj = agent_position[agent_idx, visible_steps]
                color = "tab:red" if agent_idx == 0 else "tab:blue"
                ax.scatter(traj[:, 0], traj[:, 1], color=color, s=12, alpha=0.9)
                ax.scatter(
                    traj[-1, 0], traj[-1, 1], color=color, s=24, edgecolors="black"
                )
                if not masked_steps.any():
                    ax.text(
                        traj[-1, 0] + 0.8,
                        traj[-1, 1] + 0.8,
                        "F",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                    )

            if masked_steps.any():
                masked_traj = agent_position_gt[agent_idx, masked_steps]
                masked_color = "limegreen" if agent_idx == 0 else "cyan"
                ax.scatter(
                    masked_traj[:, 0],
                    masked_traj[:, 1],
                    color=masked_color,
                    s=20,
                    alpha=0.95,
                    marker="x",
                )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlim([-30, 70])
        ax.set_ylim([-40, 40])

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
