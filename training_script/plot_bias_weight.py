import numpy as np
import plotly.express as px

def get_bias_scales(model):
    """Extracts the bias scales for each head in the decoder layers"""
    decoder_layers = len(model.model._modules['decoder'].layers)
    num_heads = model.model._modules['decoder'].layers[0]._modules['encoder_attn'].bias_scale.num_heads
    all_weights = np.zeros((decoder_layers, num_heads))
    for i in range(decoder_layers):
        bias_scale = model.model._modules['decoder'].layers[i]._modules['encoder_attn'].bias_scale
        if bias_scale.single_value:
            scale = np.full((num_heads,), bias_scale.bias_param[0].item())
        else:
            scale = bias_scale.bias_param.detach().cpu().numpy()
        all_weights[i] = scale
    return all_weights

def plot_bias_scales(scales):
    fig = px.imshow(scales, color_continuous_scale='viridis', aspect="auto",
                    labels=dict(x="Head num.", y="Decoder layer", color="Value"),)
    fig.update_traces(text=np.around(scales,3), texttemplate="<b>%{text}</b>")#, textfont={'size':6.9})
    return fig