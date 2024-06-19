import torch
import torchvision

from tobias.model_modifications.add_bias import add_bias
from tobias.tobias_control_module import TobiasControlModule
from tobias.tobias_comparator import comparator_factory
from tobias.tobias_embedder import embedder_factory

from tobias.model_modifications.modify_model import (
    ModifiedBartModel,
    CustomConfig,
)
from tobias.model_modifications.bart_modified_forward import (
    forward_with_bias,
)

pad_fill = {"-1": -1, "0": 0, "-inf": -float("inf"), "-10": -10, "none": None}


def encoder_output_hook(
    output,
    input_ids=None,
    custom_config: CustomConfig = None,
    control_module: TobiasControlModule = None,
):
    if input_ids is None:
        input_ids = custom_config.input_ids
    word_weights = control_module.calc_weights(
        input_ids=input_ids,
        encoder_hidden_states=output,
        topic_prompt=custom_config.topic_prompt,
    )
    custom_config.token_weights = word_weights.to(output.last_hidden_state.device)


def get_model(model_args, config, tokenizer, run_args, data_args):
    torch.manual_seed(42)

    custom_config = CustomConfig()
    custom_config.modified_forward = forward_with_bias
    custom_config.bias_function = add_bias
    custom_config.add_bias = run_args.add_bias
    custom_config.bias_head_scale = run_args.bias_head_scale
    custom_config.bias_scale_init = run_args.bias_scale_init
    custom_config.bias_scale_trainable = run_args.bias_scale_trainable
    custom_config.softmax_temp = run_args.softmax_temp

    model = ModifiedBartModel.from_pretrained(
        model_args.model_name_or_path,
        custom_config=custom_config,
        local_files_only=False,
        # use_safetensors=True,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if data_args.preprocessed_weights is None or run_args.add_bias == False:
        embedder = embedder_factory(
            embedder_type=run_args.embedder, model=model, tokenizer=tokenizer
        )
        comparator = comparator_factory(
            aggregator_type=run_args.aggregator,
            similarity_metric_type=run_args.similarity_calc,
            smoothing=run_args.bias_smoothing,
            smoothing_window=run_args.bias_smooth_window,
            smoothing_sigma=run_args.bias_smooth_sigma,
        )

        control_module = TobiasControlModule(
            embedder=embedder,
            comparator=comparator,
            tokenizer=tokenizer,
            pad_fill=pad_fill[run_args.pad_fill],
            prompt_fill=pad_fill[run_args.prompt_fill],
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        def encoder_hook(hidden_states, input_ids):
            return encoder_output_hook(
                hidden_states,
                input_ids,
                custom_config=custom_config,
                control_module=control_module,
            )

        model.add_encoder_hook(encoder_hook)

    return model


def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False
