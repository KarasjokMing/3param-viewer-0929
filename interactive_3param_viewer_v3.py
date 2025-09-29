#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io, os, math, random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- 参数和指标列定义 ---
PARAM_COLS_DEFAULT = ["Gamma","Delta","theta_a1","theta_a2","theta_b2","theta_b3","phi1","phi2","phi3"]
METRIC_CANDIDATES = ["T1","T2","T3","R1","R2","R3", "score", "best_for_channel"]
ANGLE_COLS = ["theta_a1","theta_a2","theta_b2","theta_b3","phi1","phi2","phi3"]


st.set_page_config(page_title="3-Parameter Interactive Viewer (v4)", layout="wide")

def _clean_str_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("\u00a0", "", regex=False)
         .str.replace(",", ".", regex=False)
         .str.replace(r"[^0-9\.\-+eE]", "", regex=True)
    )

def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(_clean_str_series(df[c]), errors="coerce")
    return df

@st.cache_data(show_spinner="正在处理大型CSV文件...")
def load_csv_cached(content_bytes: bytes, max_rows: int) -> pd.DataFrame:
    file_buffer = io.BytesIO(content_bytes)
    file_buffer.seek(0)
    num_lines = sum(1 for _ in file_buffer) - 1
    file_buffer.seek(0)

    if num_lines <= max_rows:
        st.info(f"文件行数 ({num_lines:,}) 未超过上限，将加载所有数据。")
        df = pd.read_csv(file_buffer, low_memory=False)
    else:
        st.info(f"文件行数 ({num_lines:,}) 超过上限 ({max_rows:,})，将进行随机采样...")
        sampling_fraction = max_rows / num_lines
        df = pd.read_csv(
            file_buffer,
            skiprows=lambda i: i > 0 and random.random() > sampling_fraction,
            low_memory=False
        )
        st.success(f"随机采样完成，已加载 {len(df):,} 行数据。")

    df.columns = [c.strip() for c in df.columns]
    return df

def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def get_bytes_from_upload_or_path(uploaded_file, path_text: str):
    if uploaded_file is not None:
        return uploaded_file.read()
    if path_text and os.path.exists(path_text):
        with open(path_text, "rb") as f:
            return f.read()
    return None

# --- UI 侧边栏 ---
st.sidebar.title("数据源")
uploaded = st.sidebar.file_uploader("上传 CSV", type=["csv"])
default_path = st.sidebar.text_input("或指定本地 CSV 路径", value="")
max_load_rows = st.sidebar.number_input(
    "加载前随机采样上限", min_value=10_000, max_value=5_000_000, 
    value=1_000_000, step=50_000,
    help="如果上传的CSV行数超过此值，程序会在加载时进行随机采样以防止内存溢出。"
)
bytes_data = get_bytes_from_upload_or_path(uploaded, default_path)

if bytes_data is None:
    st.info("在左侧上传或指定 CSV 路径后开始。")
    st.stop()

df = load_csv_cached(bytes_data, max_load_rows)

# --- 核心修正：调整列识别和转换的逻辑 ---
# 1. 先找出所有我们可能关心的列
all_potential_cols = PARAM_COLS_DEFAULT + METRIC_CANDIDATES
found_cols = [c for c in all_potential_cols if c in df.columns]

# 2. 对所有找到的列，强制进行一次数字类型转换
df = coerce_numeric(df, cols=found_cols)

# 3. 在强制转换之后，再来安全地识别哪些是参数列，哪些是可用的指标列
param_cols = [c for c in PARAM_COLS_DEFAULT if c in df.columns]
metric_cols = [c for c in METRIC_CANDIDATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

# 如果默认指标列都找不到，再尝试从所有数字列中寻找
if not metric_cols:
    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    metric_cols = [c for c in all_numeric_cols if c not in param_cols]
# --- 修正结束 ---


st.sidebar.markdown("---")
st.sidebar.subheader("筛选设置")
metric = st.sidebar.selectbox("指标列 (>=阈值)", options=metric_cols, index=0 if metric_cols else -1)
thr = st.sidebar.slider("阈值", min_value=0.60, max_value=0.99999, value=0.90, step=0.0005)

with st.expander("调试：指标列统计", expanded=False):
    if metric and metric in df.columns:
        ser = df[metric]
        st.write(f"数据类型: {ser.dtype}") # 检查这里是否为 float64 或 int64
        st.write(f"非空计数: {ser.notna().sum()} / {len(ser)}")
        st.write(f"最小值: {float(np.nanmin(ser.values)) if ser.notna().any() else float('nan'):.6f}")
        st.write(f"最大值: {float(np.nanmax(ser.values)) if ser.notna().any() else float('nan'):.6f}")
        st.write(f"≥阈值数量: {int((ser >= thr).sum())}")
    else:
        st.warning(f"选择的指标列 '{metric}' 不存在或无效。")

angle_wrap = st.sidebar.checkbox("角度归一到 (-π, π]", value=True)
max_points_plot = st.sidebar.number_input("最大绘制点数", min_value=1_000, max_value=2_000_000, value=200_000, step=10_000)

st.sidebar.markdown("---")
three = st.sidebar.multiselect("选择三个参量作 3D 散点", options=param_cols, default=param_cols[:3] if len(param_cols)>=3 else param_cols, max_selections=3)
color_by = st.sidebar.selectbox("颜色编码", options=[metric]+param_cols if metric else param_cols, index=0)
dot_size = st.sidebar.slider("点大小", min_value=1, max_value=6, value=2)

with st.expander("可选：进一步数值范围筛选"):
    extra_filters = {}
    for c in param_cols:
        if c in df.columns and df[c].notna().any():
            cmin, cmax = float(df[c].min()), float(df[c].max())
            vmin, vmax = st.slider(f"{c} 范围", min_value=cmin, max_value=cmax, value=(cmin, cmax))
            if vmin > cmin or vmax < cmax:
                extra_filters[c] = (vmin, vmax)

# --- 主体逻辑 ---
if not metric:
    st.error("未能识别任何可用的指标列，请检查CSV文件。")
    st.stop()
    
subset = df.loc[df[metric] >= thr].copy()

if angle_wrap:
    for col in ANGLE_COLS:
        if col in subset.columns:
            subset[col] = wrap_angle(subset[col].to_numpy())

for c, (vmin, vmax) in extra_filters.items():
    if c in subset.columns:
        subset = subset[(subset[c] >= vmin) & (subset[c] <= vmax)]

n_all = len(df); n_near = len(subset)
st.success(f"命中子集：{n_near:,} / {n_all:,} 行 (metric={metric} ≥ {thr})")

if len(three) != 3:
    st.warning("请在侧边栏恰好选择 3 个参量。")
    st.stop()

if n_near == 0:
    st.warning("筛选后没有数据点，请降低阈值或放宽范围。")
    st.stop()

plot_df = subset.sample(n=min(n_near, max_points_plot), random_state=0) if n_near > max_points_plot else subset
st.caption(f"实际绘制点数：{len(plot_df):,}")

x, y, z = three
c = color_by
fig = go.Figure(data=[
    go.Scatter3d(
        x=plot_df[x], y=plot_df[y], z=plot_df[z],
        mode="markers",
        marker=dict(
            size=dot_size,
            color=plot_df[c] if c in plot_df.columns else None,
            colorscale="Viridis", opacity=0.8,
            colorbar=dict(title=c)
        )
    )
])
fig.update_layout(
    title=f"参数空间可视化: {metric} ≥ {thr}",
    scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z),
    height=800, margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig, use_container_width=True)

csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
st.download_button("下载当前子集的 CSV", data=csv_bytes, file_name="filtered_subset.csv", mime="text/csv")