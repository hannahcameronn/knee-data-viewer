"""Knee Brace Session Viewer.

Run with:  py -m streamlit run viewer.py
"""
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ACTIVITIES = ["Walking", "Stairs", "Squats", "Running", "Sitting", "Other"]

st.set_page_config(page_title="Knee Brace Viewer", page_icon="🦵", layout="wide")
st.title("Knee brace session viewer")


# ---------------------------------------------------------------------------
# Angle math (Step 5)
# A quaternion is 4 numbers (w, x, y, z) describing which way a sensor points.
# ---------------------------------------------------------------------------
def quat_conj(q):
    """Undo a rotation: flip the sign of the x, y, z parts."""
    return q * np.array([1, -1, -1, -1])


def quat_mult(a, b):
    """Apply rotation b, then a, for every row."""
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.column_stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def knee_angles(df, cal_seconds):
    """Knee flexion angle (deg) for every row.

    1. Find how the shank is rotated relative to the thigh.
    2. Treat the first `cal_seconds` (standing straight) as 0 deg.
    3. Read off the rotation about the knee's hinge (x) axis.

    Step 3 assumes the knee bends like a simple hinge, and that x is the
    hinge axis. Check that against the real IMU's axes once you have it.
    """
    thigh = df[["thigh_w", "thigh_x", "thigh_y", "thigh_z"]].to_numpy()
    shank = df[["shank_w", "shank_x", "shank_y", "shank_z"]].to_numpy()
    rel = quat_mult(quat_conj(thigh), shank)

    t = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000
    standing = rel[(t < cal_seconds).to_numpy()]
    # q and -q mean the same rotation, so line up signs before averaging
    standing = standing * np.sign(standing @ standing[0])[:, None]
    cal = standing.mean(axis=0)
    cal /= np.linalg.norm(cal)

    knee = quat_mult(quat_conj(np.tile(cal, (len(rel), 1))), rel)
    angle = np.degrees(2 * np.arctan2(knee[:, 1], knee[:, 0]))
    return (angle + 180) % 360 - 180   # keep within -180 to 180


def count_bends(angle, high=45, low=15):
    """Count a bend each time the knee goes past `high` after straightening below `low`."""
    bends, bent = 0, False
    for a in angle:
        if not bent and a > high:
            bends, bent = bends + 1, True
        elif bent and a < low:
            bent = False
    return bends


def format_duration(hours):
    return f"{hours * 60:.1f} min" if hours < 1 else f"{hours:.2f} h"


# ---------------------------------------------------------------------------
# Load a session file (Step 3)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session file")
    uploaded = st.file_uploader("Choose a session CSV from the microSD card", type="csv")
    log_file = st.file_uploader(
        "Optional: your sessions log (sessions.csv)", type="csv",
        help="Add this session to your existing log instead of starting a new one.")
    cal_seconds = st.number_input("Standing-straight time at start (s)", 0.5, 10.0, 2.0, 0.5,
                                  help="The viewer treats this stretch as 0°.")

if uploaded is None:
    st.info("Choose a session file in the sidebar to see its results.")
    st.stop()

# Forget the last saved summary when a different session file is loaded
if st.session_state.get("loaded_file") != uploaded.name:
    st.session_state.pop("log", None)
    st.session_state["loaded_file"] = uploaded.name

df = pd.read_csv(uploaded)
t_s = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000
angle = knee_angles(df, cal_seconds)
logged_hours = t_s.iloc[-1] / 3600

# When was this session recorded? Use the board's clock if the file has a
# "datetime" column (needs a real-time clock, like the Adalogger's).
# Otherwise the wearer enters it in the form below.
has_clock = "datetime" in df.columns
if has_clock:
    rec_start = pd.to_datetime(df["datetime"].iloc[0])
    rec_end = pd.to_datetime(df["datetime"].iloc[-1])
    st.caption(f"Recorded {rec_start:%A, %B %d, %Y} from "
               f"{rec_start:%I:%M %p} to {rec_end:%I:%M %p}")

# ---------------------------------------------------------------------------
# Results (Steps 4 and 6)
# ---------------------------------------------------------------------------
peak_flex, peak_ext = angle.max(), angle.min()
rom = peak_flex - peak_ext
bends = count_bends(angle)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Peak flexion", f"{peak_flex:.0f}°")
c2.metric("Peak extension", f"{peak_ext:.0f}°", help="Negative means hyperextension.")
c3.metric("Range of motion", f"{rom:.0f}°")
c4.metric("Bends counted", bends)
c5.metric("Time logged", format_duration(logged_hours))

fig = px.line(x=t_s, y=angle, labels={"x": "Time (s)", "y": "Knee flexion (°)"})
st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw data"):
    st.plotly_chart(px.line(x=t_s, y=df["pressure"],
                            labels={"x": "Time (s)", "y": "Pressure reading"}),
                    use_container_width=True)
    st.dataframe(df)

# ---------------------------------------------------------------------------
# Wearer questions + save (Steps 1 and 2)
# ---------------------------------------------------------------------------
st.header("Wearer questions")
with st.form("questions"):
    if not has_clock:
        st.caption("This file has no date or time, so enter when it was recorded.")
        rec_date = st.date_input("Date recorded")
        rec_time = st.time_input("Start time", step=300)
    hours = st.number_input("Hours worn", 0.0, 24.0, step=0.5)
    activities = st.multiselect("What were you doing?", ACTIVITIES)
    pain = st.slider("Pain level (0 = none, 10 = worst)", 0, 10, 0)
    notes = st.text_area("Anything else?")
    submitted = st.form_submit_button("Save session")

if submitted:
    if not has_clock:
        rec_start = datetime.combine(rec_date, rec_time)
        rec_end = rec_start + pd.Timedelta(seconds=float(t_s.iloc[-1]))
    row = pd.DataFrame([{
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file": uploaded.name,
        "recorded_start": rec_start.strftime("%Y-%m-%d %H:%M:%S"),
        "recorded_end": rec_end.strftime("%Y-%m-%d %H:%M:%S"),
        "time_source": "device clock" if has_clock else "entered by wearer",
        "hours_reported": hours,
        "hours_logged": round(logged_hours, 3),
        "activities": "; ".join(activities),
        "pain": pain,
        "notes": notes,
        "peak_flexion_deg": round(peak_flex, 1),
        "peak_extension_deg": round(peak_ext, 1),
        "rom_deg": round(rom, 1),
        "bends": bends,
    }])
    # Combine with the wearer's existing log, if they uploaded one. Nothing is
    # kept on the server: the wearer downloads the result to their own computer.
    log = pd.concat([pd.read_csv(log_file), row]) if log_file else row
    st.session_state["log"] = log.to_csv(index=False)
    st.session_state["mismatch"] = hours > 0 and abs(hours - logged_hours) > 0.5
    st.session_state["hours"] = hours

if "log" in st.session_state:
    st.success("Session summary ready. Download it to keep it.")
    st.download_button("Download sessions.csv", st.session_state["log"],
                       file_name="sessions.csv", mime="text/csv", type="primary")
    if st.session_state["mismatch"]:
        st.warning(f"Reported {st.session_state['hours']} h worn, but the device logged "
                   f"{format_duration(logged_hours)}. Check the pressure sensor "
                   "or the wearer's estimate.")
    with st.expander("Preview"):
        st.dataframe(pd.read_csv(io.StringIO(st.session_state["log"])),
                     use_container_width=True)
