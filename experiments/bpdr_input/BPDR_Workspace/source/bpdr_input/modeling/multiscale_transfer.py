"""Function-preserving expansion of the fitted broad-context Transformer."""
import torch

from bpdr_input.modeling.multiscale_transformer import MultiscaleTransformer


def expand_parent(parent):
    if parent.project.in_features != 8 or parent.config.scope_same_run:
        raise ValueError('Transfer requires the existing broad eight-channel parent')
    child = MultiscaleTransformer(parent.config)
    state = dict(parent.state_dict())
    weight = torch.zeros_like(child.project.weight)
    weight[:, :5] = state['project.weight'][:, :5]
    weight[:, 10:13] = state['project.weight'][:, 5:8]
    state['project.weight'] = weight
    child.load_state_dict(state, strict=True)
    return child.eval()
