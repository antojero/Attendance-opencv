import pandas as pd
import streamlit as st
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

ts= time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp= datetime.fromtimestamp(ts).strftime("%H:%M-%S")

st_autorefresh(interval=2000,limit=100)

df = pd.read_csv("Attendance/Attendance_"+date+".csv")

st.dataframe(df.style.highlight_max(axis=0))