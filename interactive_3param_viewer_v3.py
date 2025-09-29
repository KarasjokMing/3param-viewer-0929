#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_3param_viewer_v5.py
Lean Param Visualizer — 修正版 (v0.51)
Last edit: 2025-09-29  (minor UX: decouple settings & draw; point size float)
Version: 0.51
"""
import io, os, math, random, re
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---- 配置 ----
APP_VERSION = "0.51"
APP_EDIT_DATE = "2025-09-29"

DEFAULT_MAX_PLOT_POINTS = 200_000
DEFAULT_SAMPLE_SEED = 42

# ---- 辅助函数 ----
def _norm_col(col: str) -> str:
    return col.strip() if isinstance(col, str) else col

def load_csv_bytes(content: bytes) -> pd.DataFrame:
    bio = io.BytesIO(content)
    df = pd.read_csv(bio, low_memory=False)
    df.columns = [_norm_col(c) for c in df.columns]
    return df

def __looks_like_number(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except Exception:
        return False

def coerce_numeric_guess(df: pd.DataFrame, try_cols: List[str] = None) -> pd.DataFrame:
    df = df.copy()
    cols = try_cols if try_cols is not None else df.columns.tolist()
    for c in cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            sample = s.dropna().head(200).astype(str)
            if len(sample) > 0:
                cnt_num = sample.apply(lambda x: __looks_like_number(x))
                if cnt_num.sum() / max(1, len(sample)) > 0.6:
                    s = s.str.replace("\u00a0", "", regex=False).str.replace(",", ".", regex=False)
                    s = s.str.replace(r"[^\d\.\-+eE]", "", regex=True)
                    df[c] = pd.to_numeric(s, errors="coerce")
    return df

def estimate_grid(n: int) -> Tuple[int,int]:
    if n <= 0:
        return (0,0)
    r = int(math.floor(math.sqrt(n)))
    if r*r >= n:
        return (r, r)
    c = int(math.ceil(n / r))
    return (r, c)

def sample_for_plot(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df
    return df.sample(max_points, random_state=DEFAULT_SAMPLE_SEED)

def get_recommended_outputs(all_cols: List[str], numeric_cols: List[str]) -> List[str]:
    prefs = []
    for c in all_cols:
        if re.match(r'^[TR]\d+$', c, flags=re.I):
            prefs.append(c)
    for c in all_cols:
        if c not in prefs and re.search(r'[TR]\d', c, flags=re.I):
            prefs.append(c)
    kw = ['score', 'best', 'chan', 'channel', 'port', 'out', 'trans']
    for c in all_cols:
        low = c.lower()
        if c not in prefs and any(k in low for k in kw):
            prefs.append(c)
    for c in numeric_cols:
        if c not in prefs:
            prefs.append(c)
    return prefs

# ---- Streamlit UI ----
st.set_page_config(page_title="参数可视化 (精简版)", layout="wide")
st.title("参数可视化（精简版）")
st.caption(f"版本 {APP_VERSION} — 最后编辑：{APP_EDIT_DATE}")

# 简单 CSS：放大按钮
st.markdown("""<style>
div.stButton > button {height:56px; font-size:18px;}
</style>""", unsafe_allow_html=True)

# 侧栏：数据与设置（这些设置是 **pending** ，只有按 绘制 才会应用）
with st.sidebar:
    st.header("数据")
    uploaded = st.file_uploader("上传 CSV（首行为列名）", type=["csv"])
    local_path = st.text_input("或本地 CSV 路径（可选）", value="")
    st.markdown("---")
    st.header("绘图设置（修改后请按「绘制」）")
    # 注意：这几个控件只是修改 pending 配置，不会立即触发绘图计算
    pending_max_points = st.number_input("绘图时最大点数（超出将采样）", min_value=1000, max_value=2_000_000, value=DEFAULT_MAX_PLOT_POINTS, step=1000, key="pending_max_points")
    # point size float 0.1 - 12, default 1.0 (你觉得 1 正好)
    pending_point_size = st.slider("点大小（散点/3D，0.1 为最小）", 0.1, 12.0, 1.0, step=0.1, key="pending_point_size")
    pending_color_scale = st.selectbox("颜色刻度（连续）", options=[
        "Viridis","Cividis","Plasma","Inferno","Magma","Turbo","RdYlBu","Blues","Portland"
    ], index=0, key="pending_color_scale")
    st.markdown("---")
    st.write("说明：所有更改在你按「绘制」之前只会保存在待应用设置中，不会开始数据处理或绘图。")

# 读取数据（必要，轻量）
bytes_data = None
if uploaded is not None:
    bytes_data = uploaded.read()
elif local_path:
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            bytes_data = f.read()
    else:
        st.sidebar.error("本地路径不存在。")

if bytes_data is None:
    st.info("请上传 CSV 或指定本地路径。")
    st.stop()

try:
    df = load_csv_bytes(bytes_data)
except Exception as e:
    st.error(f"读取 CSV 失败：{e}")
    st.stop()

all_cols = list(df.columns)
if len(all_cols) == 0:
    st.error("CSV 未检测到列名。")
    st.stop()

# 尝试把看起来像数字的列转为数值（轻量）
df = coerce_numeric_guess(df, try_cols=all_cols)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 为 UI 生成 pending 选择（显示给用户修改）
# 推荐 outputs / inputs
recommended_outputs = get_recommended_outputs(all_cols, numeric_cols)
recommended_inputs = [c for c in all_cols if any(k in c.lower() for k in ["theta","phi","angle","delta","gamma"])]
if not recommended_inputs:
    recommended_inputs = [c for c in numeric_cols if c not in recommended_outputs][:3]

# 初始化 session_state 的 pending 和 applied 配置（只在首次加载时）
if "pending_config" not in st.session_state:
    st.session_state["pending_config"] = {
        "outputs": recommended_outputs[:3] if len(recommended_outputs)>0 else (numeric_cols[:3] if numeric_cols else []),
        "inputs": recommended_inputs[:1] if len(recommended_inputs)>0 else (numeric_cols[:1] if numeric_cols else []),
        "max_points": DEFAULT_MAX_PLOT_POINTS,
        "point_size": 1.0,
        "color_scale": "Viridis"
    }

if "applied_config" not in st.session_state:
    # initially nothing applied
    st.session_state["applied_config"] = st.session_state["pending_config"].copy()
    st.session_state["draw_request"] = False

# UI selectors bound to pending_config (these do NOT trigger heavy compute)
with st.sidebar:
    # outputs multiselect bound to session_state pending_config
    pending_outs = st.multiselect("输出（每选一个会生成一个子图）", options=all_cols, default=st.session_state["pending_config"]["outputs"], key="ui_outputs")
    pending_inputs = st.multiselect("输入（1 到 3 个）", options=all_cols, default=st.session_state["pending_config"]["inputs"], key="ui_inputs")

    # Update pending_config when user changes these UI controls
    st.session_state["pending_config"]["outputs"] = pending_outs
    st.session_state["pending_config"]["inputs"] = pending_inputs
    # And update pending numeric settings from earlier controls
    st.session_state["pending_config"]["max_points"] = pending_max_points
    st.session_state["pending_config"]["point_size"] = pending_point_size
    st.session_state["pending_config"]["color_scale"] = pending_color_scale

# Validate pending selections lightly (no heavy ops)
if len(st.session_state["pending_config"]["inputs"]) == 0:
    st.sidebar.warning("请在侧栏选择至少 1 个输入。")
if len(st.session_state["pending_config"]["inputs"]) > 3:
    st.sidebar.error("最多选择 3 个输入。")
if len(st.session_state["pending_config"]["outputs"]) == 0:
    st.sidebar.warning("请至少选择 1 个输出。")

# 绘制 / 清除 按钮（只有按绘制时会把 pending 复制到 applied 并触发绘图）
col1, col2 = st.columns([1,1])
with col1:
    if st.button("绘制", key="draw_btn"):
        # copy pending to applied
        st.session_state["applied_config"] = st.session_state["pending_config"].copy()
        st.session_state["draw_request"] = True
with col2:
    if st.button("清除", key="clear_btn"):
        st.session_state["draw_request"] = False

if not st.session_state.get("draw_request", False):
    st.info("请在侧栏设置参数（可多次调整），满意后按「绘制」一次性生成图表。")
    st.stop()

# 使用 applied_config 来驱动重计算与绘图 —— 这部分可能成本较高
cfg = st.session_state["applied_config"]
outputs = cfg["outputs"]
inputs = cfg["inputs"]
max_plot_points = cfg["max_points"]
point_size = cfg["point_size"]
color_scale = cfg["color_scale"]

# 校验 applied selections
if len(inputs) == 0:
    st.error("已应用的设置中没有输入列。请返回侧栏选择至少 1 个输入并按绘制。")
    st.stop()
if len(inputs) > 3:
    st.error("已应用的设置中输入列超过 3 个。请调整。")
    st.stop()
if len(outputs) == 0:
    st.error("已应用的设置中没有输出列。")
    st.stop()

# 构建绘图用的 DataFrame（完整计算仅在此处发生）
plot_cols_needed = inputs + outputs
missing = [c for c in plot_cols_needed if c not in df.columns]
if missing:
    st.error(f"以下列在数据中未找到：{missing}")
    st.stop()

plot_df = df[plot_cols_needed].dropna()
if plot_df.empty:
    st.error("按已应用的列去除缺失值后无可绘制数据。")
    st.stop()

if len(plot_df) > max_plot_points:
    st.warning(f"样本 {len(plot_df):,} 超过绘图上限 {max_plot_points:,}，将随机采样用于绘图。")
    plot_df = sample_for_plot(plot_df, max_plot_points)

# 绘图函数（同之前逻辑）
def plot_for_output_single(df_plot: pd.DataFrame, out_col: str, inputs: List[str], point_size_val: float, color_scale_val: str):
    if len(inputs) == 1:
        x = inputs[0]
        tmp = df_plot[[x, out_col]].dropna()
        if tmp.empty:
            fig = go.Figure(); fig.update_layout(title=f"{out_col} 无数据"); return fig
        if tmp[x].nunique() < 1000:
            agg = tmp.groupby(x)[out_col].median().reset_index().sort_values(x)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg[x], y=agg[out_col], mode="lines+markers", name=out_col))
            fig.update_layout(title=f"{out_col} vs {x}", xaxis_title=x, yaxis_title=out_col)
            return fig
        else:
            xs = tmp[x].to_numpy(); ys = tmp[out_col].to_numpy()
            bins = 200
            try:
                bins_edges = np.linspace(np.nanmin(xs), np.nanmax(xs), bins+1)
                inds = np.digitize(xs, bins_edges)
                med = [np.nanmedian(ys[inds==i]) for i in range(1, bins+1)]
                centers = (bins_edges[:-1] + bins_edges[1:]) / 2
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=centers, y=med, mode="lines", name=out_col))
                fig.update_layout(title=f"{out_col} (binned) vs {x}", xaxis_title=x, yaxis_title=out_col)
                return fig
            except Exception:
                fig = px.scatter(tmp.sample(min(len(tmp),5000)), x=x, y=out_col)
                fig.update_layout(title=f"{out_col} (sampled scatter) vs {x}")
                return fig

    elif len(inputs) == 2:
        x,y = inputs[0], inputs[1]
        tmp = df_plot[[x,y,out_col]].dropna()
        if tmp.empty:
            fig = go.Figure(); fig.update_layout(title=f"{out_col} 无数据"); return fig
        fig = px.scatter(tmp, x=x, y=y, color=out_col, color_continuous_scale=color_scale_val, title=f"{out_col} 着色", height=420)
        fig.update_traces(marker=dict(size=point_size_val, opacity=0.8))
        return fig

    elif len(inputs) == 3:
        x,y,z = inputs[0], inputs[1], inputs[2]
        tmp = df_plot[[x,y,z,out_col]].dropna()
        if tmp.empty:
            fig = go.Figure(); fig.update_layout(title=f"{out_col} 无数据"); return fig
        fig = px.scatter_3d(tmp, x=x, y=y, z=z, color=out_col, color_continuous_scale=color_scale_val, title=f"{out_col} (3D)", height=520)
        fig.update_traces(marker=dict(size=point_size_val, opacity=0.9))
        return fig
    else:
        fig = go.Figure(); fig.update_layout(title="不支持的输入维度"); return fig

# 并列布局绘图
n_out = len(outputs)
rows, cols = estimate_grid(n_out)
if rows == 0: rows, cols = 1, n_out

container = st.container()
idx = 0
for r in range(rows):
    cols_ui = container.columns(cols)
    for cidx in range(cols):
        if idx >= n_out:
            cols_ui[cidx].empty()
            continue
        out_name = outputs[idx]
        with cols_ui[cidx]:
            st.subheader(out_name)
            fig = plot_for_output_single(plot_df, out_name, inputs, point_size, color_scale)
            st.plotly_chart(fig, use_container_width=True)
            subset_csv = plot_df[[*inputs, out_name]].dropna().to_csv(index=False).encode("utf-8")
            st.download_button(f"下载 {out_name} 子集", data=subset_csv, file_name=f"{out_name}_subset.csv", mime="text/csv")
        idx += 1

st.success("绘制完成。若需更新设置，请在侧栏修改后再次按「绘制」。")
