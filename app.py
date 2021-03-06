import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

st.write(
    "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
    unsafe_allow_html=True,
)


@st.cache(allow_output_mutation=True)
def load_df(filename):
    df = pd.read_csv(filename)
    return df


def plot_loss(filename, title, model=None):
    filename = "train_logs/" + filename
    file = open(filename, "r")
    Lines = file.readlines()
    train_loss = []
    valid_loss = []
    for line in Lines:
        if "valid" in line:
            train_loss.append(line.split(" ")[7])
            valid_loss.append(line.split(" ")[11][:-1])
        if "test" in line:
            test_per = float(line.split(" ")[-1])
        if "loaded" in line:
            loaded_epoch = int(line.split(" ")[2])

    train_loss = [float(i) for i in train_loss]
    valid_loss = [float(i) for i in valid_loss]

    epoch = np.arange(1, 51, 1)

    env = ""
    if "env" in filename:
        env = "(env_corruption)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=epoch, y=train_loss, mode="lines+markers", name="train loss")
    )
    fig.add_trace(
        go.Scatter(
            x=[loaded_epoch],
            y=[train_loss[loaded_epoch - 1]],
            mode="markers",
            name="Epoch loaded",
            marker=dict(color="#f8766d", size=8),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=epoch, y=valid_loss, mode="lines+markers", name="valid loss")
    )
    fig.add_trace(
        go.Scatter(
            x=[loaded_epoch],
            y=[valid_loss[loaded_epoch - 1]],
            mode="markers",
            name="Epoch loaded",
            marker=dict(color="#f8766d", size=8),
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="{} {}<br>Test PER: {}".format(title, env, test_per),
        yaxis_range=[0, 5],
        title_x=0.5,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
    )
    #img_name = "images/{} {}.png".format(title, env)
    #if model is not None:
    #    img_name = img_name.replace("images", "images/{}".format(model))
    #fig.write_image(img_name)
    return fig


st.markdown("# Experiments")

title_list = [
    "Single channel",
    "Delay and sum beamforming",
    "MVDR beamforming",
    "Gev beamforming",
    "BeamformIt",
    "Beamforming on the fly",
    "Averaging probabilities",
    "Early concatenation",
    "Mid-concatenation",
    "Late concatenation",
]
title = st.radio("Select Experiment", tuple(title_list))

title_file_dict = {}
title_file_dict["Delay and sum beamforming"] = "train_log_das"
title_file_dict["MVDR beamforming"] = "train_log_mvdr"
title_file_dict["Gev beamforming"] = "train_log_gev"
title_file_dict["Single channel"] = "train_log_la2"
title_file_dict["Early concatenation"] = "train_log_early"
title_file_dict["BeamformIt"] = "train_log_beamformit"
title_file_dict["Beamforming on the fly"] = "train_log_fly"
title_file_dict["Late concatenation"] = "train_log_late"
title_file_dict["Averaging probabilities"] = "train_log_avg_prob"
title_file_dict["Mid-concatenation"] = "train_log_mid"

filename = title_file_dict[title] + ".txt"
filename_env = title_file_dict[title] + "_env.txt"

col1, col2 = st.beta_columns(2)
with col1:
    st.plotly_chart(plot_loss(filename_env, title))
with col2:
    st.plotly_chart(plot_loss(filename, title))

df = load_df("PER.csv")
_, col3, _ = st.beta_columns([0.25, 0.5, 0.25])
with col3:
    sort = st.radio("Sort by", ("PER (env_corrupt)", "PER"))
    if sort == "PER (env_corrupt)":
        st.table(df.sort_values(by="PER (env_corrupt)").reset_index(drop=True))
    if sort == "PER":
        st.table(df.sort_values(by="PER").reset_index(drop=True))

# DenseNet
densenet_title_list = ["Single channel", "Delay and sum beamforming", "Early concatenation"]

title_file_dict["Delay and sum beamforming (DenseNet)"] = "train_log_densenet_das_env"
title_file_dict["Early concatenation (DenseNet)"] = "train_log_densenet_early_env"
title_file_dict["Single channel (DenseNet)"] = "train_log_densenet_la2_env"

dense_dict = {}
dense_dict["Single channel"] = "2 dense blocks with 4 layers each"
dense_dict["Delay and sum beamforming"] = "2 dense blocks with 4 layers each"
dense_dict["Early concatenation"] = "2 dense blocks with 3 layers each"

col_dense1, col_dense2 = st.beta_columns(2)
df_dn = load_df("PER_DN.csv")
with col_dense1:
    st.markdown("# DenseNet")
    title_dense = st.radio("Select Experiment", tuple(densenet_title_list))
    st.markdown("<center><i>{}<br>(128, 256)</i><center>".format(dense_dict[title_dense]), unsafe_allow_html=True)
    title_dense += " (DenseNet)"
    filename_dense = title_file_dict[title_dense] + ".txt"
    st.plotly_chart(plot_loss(filename_dense, title_dense.split(" (")[0], model="DenseNet"))
    st.table(df_dn.sort_values(by="PER (env_corrupt)").reset_index(drop=True))

# ResNet
resnet_title_list = ["Single channel", "Mid-concatenation"]

title_file_dict["Single channel (ResNet)"] = "train_log_resnet_la2_env"
title_file_dict["Mid-concatenation (ResNet)"] = "train_log_resnet_mid_env"

res_dict = {}
res_dict["Single channel"] = "3 residual blocks with 3 layers each"
res_dict["Mid-concatenation"] = "3 residual blocks with 3 layers each"

df_rn = load_df("PER_RN.csv")
with col_dense2:
    st.markdown("# ResNet")
    title_res = st.radio("Select Experiment", tuple(resnet_title_list))
    st.markdown("<center><i>{}<br>(128, 128, 128)</i><center>".format(res_dict[title_res]), unsafe_allow_html=True)
    title_res += " (ResNet)"
    filename_res = title_file_dict[title_res] + ".txt"
    st.plotly_chart(plot_loss(filename_res, title_res.split(" (")[0], model="ResNet"))
    st.table(df_rn.sort_values(by="PER (env_corrupt)").reset_index(drop=True))