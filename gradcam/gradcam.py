import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        if target_layer is None:
            target_layer = model.features[-1]
        self.target_layer = target_layer
        self.model = model.eval()
        self.hook_handles = []
        self.act_hook = None
        self._hooks_enabled = True
        self._relu = False
        # Forward hook
        fw_hook = 'register_forward_hook'
        self.hook_handles.append(getattr(target_layer, fw_hook)(self._act_hook))
        # Last FC Layer weights
        self._fc_weights = model.classifier[-1].weight.data


    def _act_hook(self, module: nn.Module, input: Tensor, output: Tensor):
        """Activation hook"""
        if self._hooks_enabled:
            self.act_hook = output.data


    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


    def __call__(self, class_idx, prediction, normalized=False):
        # Check that forward has already occurred
        if not isinstance(self.act_hook, Tensor):
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        # Check batch size
        if self.act_hook.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.act_hook.shape[0]}")

        # compute GradCAM
        return self.compute_saliency(class_idx, prediction, normalized)


    def _get_weights(self, class_idx, prediction):
        # Return FC weights of targeted class
        return self._fc_weights[-1, :]


    def _normalize(self, batch_cams):
        raise NotImplementedError


    def compute_saliency(self, class_idx, prediction, normalized):
        weights = self._get_weights(class_idx, prediction)
        weights = weights[(...,) + (None,) * (self.act_hook.ndim - 2)]

        batch_cams = torch.nansum(weights * self.act_hook.squeeze(0), dim=0)

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams


class GradCAM(CAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        """
        GradCAM implementation similar as described in `"Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" <https://arxiv.org/pdf/1610.02391.pdf>`. Inheriting from the CAM class,
        as GradCAM is based on CAM.
        Inspired by https://github.com/frgfm/torch-cam.
        :param model: the model for which GradCAM is computed
        :param target_layer: the target layer for GradCAM
        """

        super().__init__(model, target_layer)
        self.grad_hook = None
        self._hooks_enabled = True
        self._relu = True

        # Backward hook
        bw_hook = 'register_full_backward_hook' if torch.__version__ >= '1.8.0' else 'register_backward_hook'
        self.hook_handles.append(getattr(self.target_layer, bw_hook)(self._grad_hook))


    def _grad_hook(self, module: nn.Module, input: Tensor, output: Tensor):
        """Gradient hook"""
        if self._hooks_enabled:
            self.grad_hook = output[0].data


    def _backprop(self, prediction, class_idx):
        if self.act_hook is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be HOOKED")
        loss = prediction[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)


    def _get_weights(self, class_idx, prediction):
        self.grad_hook: Tensor
        self._backprop(prediction, class_idx)

        # GAP over gradients of u x v of feature map
        return self.grad_hook.squeeze(0).flatten(1).mean(-1)





