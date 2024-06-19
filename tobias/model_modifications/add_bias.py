import torch


def add_bias(module, key_states, query_states, attn_weights, hidden_states):
    bsz, tgt_len, _ = hidden_states.size()
    input_ids = module.custom_config.input_ids
    src_len = input_ids.shape[1]
    num_ids = input_ids.shape[0]


    if module.custom_config.token_weights is None:
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return key_states, query_states, attn_weights, bsz

    num_beams = bsz // num_ids

    # Get word weights and softmax temperature
    word_weights = module.custom_config.token_weights.to(attn_weights.device)
    softmax_temp = module.custom_config.softmax_temp if module.custom_config.softmax_temp is not None else None

    if softmax_temp is not None and softmax_temp > 0.0:
        word_weights = word_weights / softmax_temp

    # Apply bias to each head
    interleave_tensor = torch.full(
        (num_ids,),
        num_beams * module.num_heads,
        device=word_weights.device,
    )
    weights_tensor = (
        word_weights.repeat_interleave(interleave_tensor, dim=0)
        .unsqueeze(1)
        .repeat(1, tgt_len, 1)
    )

    attn_weights = torch.softmax(attn_weights, dim=-1)
    weights_tensor = torch.softmax(weights_tensor, dim=-1)

    attn_weights = module.bias_scale(weights_tensor, attn_weights, bsz)

    return key_states, query_states, attn_weights, bsz
