import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

def epoch_loaded(filename):
	file = open("epoch_loaded.txt", 'r')
	Lines = file.readlines()
	for line in Lines:
		if filename in line:
			return int(line.split(":")[-1])

def plot_loss(filename, title):
    file = open(filename, 'r')
    Lines = file.readlines()
    train_loss = []
    valid_loss = []
    for line in Lines:
        if "valid" in line:
            train_loss.append(line.split(" ")[7])
            valid_loss.append(line.split(" ")[11][:-1])
        if "test" in line:
            test_per = float(line.split(" ")[-1])

    train_loss = [float(i) for i in train_loss]
    valid_loss = [float(i) for i in valid_loss]
    
    epoch = np.arange(1, 51, 1)
    
    env = ""
    if "env" in filename:
        env = "(env_corrupt)"

    loaded_epoch = epoch_loaded(filename)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch, y=train_loss, mode="lines+markers", name="train loss"))
    fig.add_trace(go.Scatter(x=[loaded_epoch], y=[train_loss[loaded_epoch-1]], mode="markers", name="Epoch loaded", 
    			  marker=dict(color="#f8766d", size=8), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=epoch, y=valid_loss, mode="lines+markers", name="valid loss"))
    fig.add_trace(go.Scatter(x=[loaded_epoch], y=[valid_loss[loaded_epoch-1]], mode="markers", name="Epoch loaded", 
    			  marker=dict(color="#f8766d", size=8), showlegend=True, hoverinfo="skip"))
    fig.update_layout(title="{} {}<br>Test PER: {}".format(title, env, test_per), yaxis_range=[0,10] if "late" in filename else [0,3.5],
                      title_x=0.5, xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified")
    return fig


title_list = ["Delay and sum beamforming", "MVDR beamforming", "GeV beamforming", "Single channel",
			  "Early concatenation", "BeamformIt", "Beamforming on the fly", "Late concatenation"]
#title = st.selectbox("Select Experiment", title_list)
title = st.radio("Select Experiment", tuple(title_list))

title_file_dict = {}
title_file_dict["Delay and sum beamforming"] = "train_log_das"
title_file_dict["MVDR beamforming"] = "train_log_mvdr"
title_file_dict["GeV beamforming"] = "train_log_gev"
title_file_dict["Single channel"] = "train_log_la2"
title_file_dict["Early concatenation"] = "train_log_early"
title_file_dict["BeamformIt"] = "train_log_beamformit"
title_file_dict["Beamforming on the fly"] = "train_log_fly"
title_file_dict["Late concatenation"] = "train_log_late"

filename = title_file_dict[title] + ".txt"
filename_env = title_file_dict[title] + "_env.txt"

col1, col2 = st.beta_columns(2)
with col1:
	st.write(plot_loss(filename, title))
with col2:
	st.write(plot_loss(filename_env, title))