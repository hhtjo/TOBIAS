import numpy as np
import plotly.express as px
import math

def reshape_data(orignal_array, number_per_row, pad_value=0):
    number_of_rows = math.ceil(orignal_array.size/ number_per_row)
    padding_needed = (number_of_rows - (orignal_array.size % number_of_rows)) % number_of_rows
    padded_array = np.pad(orignal_array, (0, padding_needed), mode='constant')
    if padding_needed != 0:
        padded_array[-padding_needed:] = pad_value
    reshaped_array = padded_array.reshape(number_of_rows, -1)
    return reshaped_array

def get_plotly_heatmap(tokens, weights, num_per_row):
    tokens = reshape_data(tokens, num_per_row, "")
    weights = reshape_data(weights, num_per_row, 0)
    fig = px.imshow(weights, color_continuous_scale='Inferno', aspect="auto")
    fig.update_traces(text=tokens, texttemplate="<b>%{text}</b>", textfont={'size':6.9})
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig