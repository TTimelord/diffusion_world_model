from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BaseWorldModel(ModuleAttrMixin, nn.Module):
    """
    Abstract base class for a world model. A world model predicts future state (or images)
    based on current/past observations and an action sequence.

    Subclasses can implement:
        - State-based predictions
        - Image-based predictions
        - Keypoint-based predictions
    """
    def __init__(self, shape_meta: dict, **kwargs):
        """
        shape_meta: dictionary containing metadata about observation/action shapes.
                    e.g. shape_meta['obs_image']['shape'] for images,
                         shape_meta['action']['shape'] for actions.
        """
        super().__init__()
        self.shape_meta = shape_meta
        self.kwargs = kwargs

    def predict_future(self, obs_dict: Dict[str, torch.Tensor], action) -> Dict[str, torch.Tensor]:
        """
        Compute the predicted future.

        Args:
            obs_dict (Dict[str, torch.Tensor]): 
                May contain:
                  - "obs": B,To,*
                  - "action": B,Ta,Da
                  - or other keys (images, states, etc.)
        Returns:
            Dict[str, torch.Tensor]: 
                Must at least include "predicted_future" (e.g., predicted future images or states).
        """
        raise NotImplementedError()

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Set or load the normalizer for input/outputs if needed (similar to policy).
        """
        raise NotImplementedError()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the training loss for a batch of data.

        batch typically includes:
          - batch['obs'] (the past observation)
          - batch['action'] (the action sequence)
          - batch['future'] (the groundtruth future states/images)
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset any stateful internal variables if needed.
        """
        pass
