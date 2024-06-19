from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartEncoder,
    BartForConditionalGeneration,
)
from transformers.models.t5.modeling_t5 import (
    T5LayerCrossAttention,
    T5ForConditionalGeneration,
)
from transformers.modeling_utils import PreTrainedModel
import types
from typing import Callable, List
import logging
import torch


# dict inheritance occurring here due to internal logging using json dumps to logg model arguments
# runtime exception otherwise
class CustomConfig(dict):
    def __init__(self):
        self.add_bias_function = None
        self.token_weights = None
        self.input_ids = None
        self.modified_forward = None
        self.bias_function = None
        self.encoder_output_hook = None
        self.weight_calc = None
        self.similarity_calc = None
        self.pad_fill = None
        self.topic_prompt = None
        self.model = None
        self.bias_head_scale = None
        self.bias_scale_trainable = None
        self.bias_scale_init = None
        self.bias_smoothing = None
        self.bias_smooth_window = None
        self.bias_smooth_sigma = None
        self.softmax_temp = None


class ModifiedModel(PreTrainedModel):
    def __init__(self, *args, **kwargs):
        self.custom_config = kwargs.pop("custom_config", None)
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(self.name_or_path)

        if self.custom_config.modified_forward is None:
            raise ValueError("Modified forward function is None.")
        if self.custom_config.bias_function is None:
            raise ValueError("Bias function is None.")

        self._modify_self(
            self,
            self.custom_config.modified_forward,
            self.custom_config.bias_function,
        )

    def __call__(self, *args, **kwds):
        weights = kwds.pop("weights", None)
        input_ids = kwds.get("input_ids")
        topic_prompt = kwds.pop("topic_prompt", None)

        if topic_prompt is not None:
            self.custom_config.topic_prompt = topic_prompt

        if self.custom_config.weight_calc and weights is None and input_ids is not None:
            weights = self.custom_config.weight_calc(input_ids)
        self._init_custom_config(input_ids, weights)

        return super().__call__(*args, **kwds)

    def generate(self, inputs=None, **kwargs):
        weights = kwargs.pop("weights", None)
        input_ids = kwargs.get("input_ids", inputs)
        topic_prompt = kwargs.pop("topic_prompt", None)

        if topic_prompt is not None:
            self.custom_config.topic_prompt = topic_prompt

        if self.custom_config.weight_calc and weights is None and input_ids is not None:
            weights = self.custom_config.weight_calc(input_ids)
        self._init_custom_config(input_ids, weights)

        if self.custom_config.input_ids is None:
            raise ValueError("Input ids not found when calling generate")

        return super().generate(inputs, **kwargs)

    def _init_custom_config(self, input_ids, weights):
        if weights is not None:
            self.custom_config.token_weights = weights
        if input_ids is not None:
            self.custom_config.input_ids = input_ids

    def _reset_custom_config(self):
        self.custom_config.token_weights = None
        self.custom_config.input_ids = None

    # Taken from https://github.com/mit-han-lab/streaming-llm
    def _modify_self(
        self,
        model,
        modified_forward: Callable,
        add_bias_function: Callable,
    ):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                self._modify_self(module, modified_forward, add_bias_function)

            self._modify_module(module, name, modified_forward, add_bias_function)

    def _modify_module(self, module, name, modified_forward, add_bias_function):
        raise NotImplemented()

class ScaleLayer(torch.nn.Module):

    def __init__(self, num_heads, init_value, single_value=False):
        super().__init__()
        self.num_heads = num_heads
        self.single_value = single_value
        if single_value:
            self.bias_param = torch.nn.Parameter(torch.tensor([init_value], dtype=torch.float32), requires_grad=True)
        else:
            self.bias_param = torch.nn.Parameter(
                torch.full((self.num_heads,), init_value, dtype=torch.float32),
                requires_grad=True,
            )

    def _get_bias_scale(self):
        if self.single_value:
            return self.bias_param.repeat(self.num_heads).squeeze()
        return self.bias_param

    def forward(self, weights_tensor, attn_weights, bsz):
        bias_scale = self._get_bias_scale()
        attn_scale = (
            torch.full((self.num_heads,), 1, device=bias_scale.device) - bias_scale
        )

        weights_scale_tensor = bias_scale.repeat(bsz).unsqueeze(1).unsqueeze(2)
        attn_scale_tensor = attn_scale.repeat(bsz).unsqueeze(1).unsqueeze(2)
        weights_res = weights_scale_tensor * weights_tensor
        attn_res = attn_scale_tensor*attn_weights
        return weights_res + attn_res


class ModifiedBartModel(ModifiedModel, BartForConditionalGeneration):

    def _modify_module(
        self,
        module,
        name,
        modified_forward,
        add_bias_function,
    ):
        if (
            isinstance(module, BartAttention) and name == "encoder_attn"
        ):  # encoder_attn refers to cross-attn in decoder layer
            module.custom_config = self.custom_config
            module.model_config = self.config
            module.custom_config.add_bias_function = add_bias_function

            scale_layer = ScaleLayer(
                module.num_heads,
                module.custom_config.bias_scale_init,
                single_value=not module.custom_config.bias_head_scale,
            )
            module.register_module("bias_scale", scale_layer)
            if not module.custom_config.bias_scale_trainable:
                freeze_weights(module.bias_scale)

            module.forward = types.MethodType(modified_forward, module)
            self.logger.debug(
                f"Added custom forward and bias function to BART cross-attention layer {name}."
            )

    def add_encoder_hook(self, encoder_output_hook: Callable):
        def _add_hook_inner(model, encoder_output_hook: Callable):
            for name, module in reversed(model._modules.items()):
                if len(list(module.children())) > 0:
                    _add_hook_inner(module, encoder_output_hook)

                if isinstance(module, BartEncoder) and encoder_output_hook is not None:
                    orig_fwd = module.forward

                    def inner_encoder_fwd(self, *args, **kwds):
                        skip_hook = kwds.pop('skip_hook', False)
                        output = orig_fwd(*args, **kwds)
                        if encoder_output_hook and not skip_hook:
                            input_ids = kwds.get('input_ids', None)
                            encoder_output_hook(output, input_ids=input_ids)
                        return output

                    module.forward = types.MethodType(inner_encoder_fwd, module)

        _add_hook_inner(self.model, encoder_output_hook)


def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False
