import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 基本設定
# ============================================================
st.set_page_config(
    page_title="最急降下法による線形回帰",
    layout="wide",
)

st.title("最急降下法による線形回帰")

st.markdown(
    r"""
このアプリでは，線形回帰式

$$
    y = ax + b
$$

に対して，二乗平均誤差

$$
    z = \frac{1}{n}\sum_{i=1}^{n} \{ y_i - (ax_i+b) \}^2
$$

を最小にする係数 $a$, $b$ を調べます。
"""
)


# ============================================================
# セッション状態の初期化
# ============================================================
def init_session_state():
    if "manual_history" not in st.session_state:
        st.session_state.manual_history = []

    if "auto_history" not in st.session_state:
        st.session_state.auto_history = []

    if "last_manual" not in st.session_state:
        st.session_state.last_manual = None

    if "show_manual_surface" not in st.session_state:
        st.session_state.show_manual_surface = False

    if "last_recorded_manual_a" not in st.session_state:
        st.session_state.last_recorded_manual_a = None

    if "last_recorded_manual_b" not in st.session_state:
        st.session_state.last_recorded_manual_b = None

    if "manual_a" not in st.session_state:
        st.session_state.manual_a = 1.0

    if "manual_b" not in st.session_state:
        st.session_state.manual_b = 0.0


init_session_state()


# ============================================================
# 計算関数
# ============================================================
def mse_and_gradient(x: np.ndarray, y: np.ndarray, a: float, b: float):
    """
    z = MSE = mean((y - (a x + b))^2)

    dz/da = -2 mean(x * residual)
    dz/db = -2 mean(residual)
    """
    y_hat = a * x + b
    residual = y - y_hat

    z = np.mean(residual ** 2)
    grad_a = -2.0 * np.mean(x * residual)
    grad_b = -2.0 * np.mean(residual)

    return z, grad_a, grad_b


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    a0: float,
    b0: float,
    learning_rate: float,
    max_iter: int,
    tolerance: float,
):
    history = []

    a = float(a0)
    b = float(b0)

    for k in range(max_iter + 1):
        z, grad_a, grad_b = mse_and_gradient(x, y, a, b)
        grad_norm = np.sqrt(grad_a**2 + grad_b**2)

        history.append(
            {
                "step": k,
                "a": a,
                "b": b,
                "z": z,
                "grad_a": grad_a,
                "grad_b": grad_b,
                "grad_norm": grad_norm,
            }
        )

        if grad_norm < tolerance:
            break

        a = a - learning_rate * grad_a
        b = b - learning_rate * grad_b

    return pd.DataFrame(history)


def make_surface_grid(x, y, a_center, b_center, span_a, span_b, grid_size):
    a_min = a_center - span_a
    a_max = a_center + span_a
    b_min = b_center - span_b
    b_max = b_center + span_b

    a_values = np.linspace(a_min, a_max, grid_size)
    b_values = np.linspace(b_min, b_max, grid_size)

    A, B = np.meshgrid(a_values, b_values)
    Z = np.zeros_like(A, dtype=float)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j], _, _ = mse_and_gradient(x, y, A[i, j], B[i, j])

    return A, B, Z


def plot_mse_points(
    history_df,
    current_row,
    title="手動最適化における過去の点と現在の点",
):
    fig = go.Figure()

    if history_df is not None and len(history_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=history_df["a"],
                y=history_df["b"],
                z=history_df["z"],
                mode="markers+lines",
                marker=dict(size=5),
                line=dict(width=4),
                text=[
                    f"step={int(row.step)}<br>a={row.a:.6f}<br>b={row.b:.6f}<br>z={row.z:.6f}"
                    for row in history_df.itertuples()
                ],
                hoverinfo="text",
                name="過去の点",
            )
        )

    if current_row is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[current_row["a"]],
                y=[current_row["b"]],
                z=[current_row["z"]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="diamond"),
                text=[
                    f"現在の点<br>a={current_row['a']:.6f}<br>b={current_row['b']:.6f}<br>z={current_row['z']:.6f}"
                ],
                hoverinfo="text",
                name="現在の点",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title="z = MSE",
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def plot_mse_surface(
    x,
    y,
    history_df,
    current_row,
    a_center,
    b_center,
    span_a,
    span_b,
    grid_size,
    title,
    show_surface=True,
):
    fig = go.Figure()

    if show_surface:
        A, B, Z = make_surface_grid(
            x=x,
            y=y,
            a_center=a_center,
            b_center=b_center,
            span_a=span_a,
            span_b=span_b,
            grid_size=grid_size,
        )

        fig.add_trace(
            go.Surface(
                x=A,
                y=B,
                z=Z,
                opacity=0.75,
                colorscale="Viridis",
                showscale=True,
                name="MSE surface",
            )
        )

    if history_df is not None and len(history_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=history_df["a"],
                y=history_df["b"],
                z=history_df["z"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(width=5),
                name="更新履歴",
            )
        )

    if current_row is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[current_row["a"]],
                y=[current_row["b"]],
                z=[current_row["z"]],
                mode="markers",
                marker=dict(size=9, color="red", symbol="diamond"),
                name="現在の点",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title="z = MSE",
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def plot_regression_line(x, y, a, b):
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = a * x_line + b

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="アップロードデータ",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"y = {a:.4f}x + {b:.4f}",
            line=dict(width=4),
        )
    )

    fig.update_layout(
        title="現在の回帰直線",
        xaxis_title="x",
        yaxis_title="y",
        height=500,
    )

    return fig


def make_mse_animation(history_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["z"],
            mode="lines+markers",
            name="z = MSE",
        )
    )

    frames = []
    for k in range(len(history_df)):
        df_k = history_df.iloc[: k + 1]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=df_k["step"],
                        y=df_k["z"],
                        mode="lines+markers",
                        name="z = MSE",
                    )
                ],
                name=str(k),
            )
        )

    fig.frames = frames

    fig.update_layout(
        title="最急降下法による z の変化",
        xaxis_title="更新回数",
        yaxis_title="z = MSE",
        height=500,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "再生",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "停止",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )

    return fig


def make_regression_animation(x, y, history_df):
    x_line = np.linspace(np.min(x), np.max(x), 200)

    first = history_df.iloc[0]
    y_line_first = first["a"] * x_line + first["b"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="アップロードデータ",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line_first,
            mode="lines",
            name="回帰直線",
            line=dict(width=4),
        )
    )

    frames = []
    for k, row in history_df.iterrows():
        y_line = row["a"] * x_line + row["b"]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name="アップロードデータ",
                    ),
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="回帰直線",
                        line=dict(width=4),
                    ),
                ],
                name=str(k),
                layout=go.Layout(
                    title_text=(
                        f"回帰直線の変化：step={int(row['step'])}, "
                        f"a={row['a']:.4f}, b={row['b']:.4f}, z={row['z']:.6f}"
                    )
                ),
            )
        )

    fig.frames = frames

    y_margin = 0.1 * (np.max(y) - np.min(y) + 1e-12)

    fig.update_layout(
        title=(
            f"回帰直線の変化：step=0, "
            f"a={first['a']:.4f}, b={first['b']:.4f}, z={first['z']:.6f}"
        ),
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(range=[np.min(y) - y_margin, np.max(y) + y_margin]),
        height=550,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "再生",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "停止",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )

    return fig


# ============================================================
# CSVアップロード
# ============================================================
st.sidebar.header("データ設定")

uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロードしてください",
    type=["csv"],
)

if uploaded_file is None:
    st.info("CSVファイルがアップロードされていないため，サンプルデータを使います。")

    df = pd.DataFrame(
        {
            "x": [0.0, 0.5263, 1.0526, 1.5789, 2.1052],
            "y": [1.2512, 2.0515, 4.9124, 5.1571, 5.1918],
        }
    )
else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSVファイルを読み込めませんでした：{e}")
        st.stop()

if df.shape[1] < 2:
    st.error("CSVには少なくとも2列が必要です。")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("数値列が少なくとも2列必要です。")
    st.stop()

x_col = st.sidebar.selectbox("xとして使う列", numeric_cols, index=0)
y_col = st.sidebar.selectbox("yとして使う列", numeric_cols, index=1)

work_df = df[[x_col, y_col]].dropna().copy()
work_df.columns = ["x", "y"]

if len(work_df) < 2:
    st.error("欠損値を除いた後のデータ数が不足しています。")
    st.stop()

x = work_df["x"].to_numpy(dtype=float)
y = work_df["y"].to_numpy(dtype=float)

st.subheader("使用データ")
st.dataframe(work_df, use_container_width=True, hide_index=True)


# ============================================================
# サイドバー：描画設定
# ============================================================
st.sidebar.header("MSE曲面の描画設定")

span_a = st.sidebar.number_input("a方向の表示幅", value=5.0, min_value=0.1, step=0.5)
span_b = st.sidebar.number_input("b方向の表示幅", value=10.0, min_value=0.1, step=0.5)
grid_size = st.sidebar.slider("曲面グリッド数", min_value=20, max_value=100, value=45, step=5)

st.sidebar.header("手動最適化の設定")
manual_step_a = st.sidebar.number_input(
    "a の変化幅",
    value=0.1,
    min_value=0.000001,
    step=0.1,
    format="%.6f",
)
manual_step_b = st.sidebar.number_input(
    "b の変化幅",
    value=0.1,
    min_value=0.000001,
    step=0.1,
    format="%.6f",
)


# ============================================================
# 手動最適化タブ・自動最適化タブ
# ============================================================
manual_tab, auto_tab = st.tabs(["1. 手動最適化", "2. 自動最適化"])


with manual_tab:
    st.header("1. 手動最適化")

    a_manual = float(st.session_state.manual_a)
    b_manual = float(st.session_state.manual_b)

    z_manual, grad_a_manual, grad_b_manual = mse_and_gradient(x, y, a_manual, b_manual)
    grad_norm_manual = np.sqrt(grad_a_manual**2 + grad_b_manual**2)

    manual_changed = (
        st.session_state.last_recorded_manual_a is None
        or st.session_state.last_recorded_manual_b is None
        or not np.isclose(a_manual, st.session_state.last_recorded_manual_a)
        or not np.isclose(b_manual, st.session_state.last_recorded_manual_b)
    )

    if manual_changed:
        new_row = {
            "step": len(st.session_state.manual_history),
            "a": a_manual,
            "b": b_manual,
            "z": z_manual,
            "grad_a": grad_a_manual,
            "grad_b": grad_b_manual,
            "grad_norm": grad_norm_manual,
        }
        st.session_state.manual_history.append(new_row)
        st.session_state.last_manual = new_row
        st.session_state.last_recorded_manual_a = a_manual
        st.session_state.last_recorded_manual_b = b_manual

    manual_history_df = pd.DataFrame(st.session_state.manual_history)

    current_manual_row = {
        "a": a_manual,
        "b": b_manual,
        "z": z_manual,
        "grad_a": grad_a_manual,
        "grad_b": grad_b_manual,
        "grad_norm": grad_norm_manual,
    }

    mse_col, regression_col, setting_col = st.columns([1.25, 1.25, 1.0])

    with mse_col:
        st.markdown("#### MSE")
        if st.session_state.show_manual_surface:
            st.plotly_chart(
                plot_mse_surface(
                    x=x,
                    y=y,
                    history_df=manual_history_df,
                    current_row=current_manual_row,
                    a_center=a_manual,
                    b_center=b_manual,
                    span_a=span_a,
                    span_b=span_b,
                    grid_size=grid_size,
                    title="z = MSE の曲面，過去の点，現在の点",
                    show_surface=True,
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                plot_mse_points(
                    history_df=manual_history_df,
                    current_row=current_manual_row,
                    title="過去の点 (a, b, z) と現在の点",
                ),
                use_container_width=True,
            )

    with regression_col:
        st.markdown("#### 回帰直線")
        st.plotly_chart(
            plot_regression_line(x, y, a_manual, b_manual),
            use_container_width=True,
        )

    with setting_col:
        st.markdown("#### 係数の手動設定")
        st.number_input("a", step=manual_step_a, format="%.6f", key="manual_a")
        st.number_input("b", step=manual_step_b, format="%.6f", key="manual_b")

        if st.button("手動履歴をリセット", key="reset_manual_history"):
            st.session_state.manual_history = []
            st.session_state.last_manual = None
            st.session_state.last_recorded_manual_a = None
            st.session_state.last_recorded_manual_b = None
            st.session_state.show_manual_surface = False
            st.rerun()

        st.markdown("#### 現在の値")
        st.markdown(f"##### a = {a_manual:.4f}")
        st.markdown(f"##### b = {b_manual:.4f}")
        st.markdown(f"##### MSE = {z_manual:.4f}")

        st.markdown("#### 勾配")
        st.markdown(f"##### ∂z/∂a = {grad_a_manual:.4f}")
        st.markdown(f"##### ∂z/∂b = {grad_b_manual:.4f}")
        st.markdown(f"##### 勾配の大きさ = {grad_norm_manual:.4f}")

        st.markdown("#### 曲面表示")
        st.caption("初期状態では曲面を描画しません。必要なときだけ表示します。")
        st.session_state.show_manual_surface = st.checkbox(
            "z = f(a, b) の曲面も表示する",
            value=st.session_state.show_manual_surface,
            key="manual_surface_checkbox",
        )



    if len(manual_history_df) > 0:
        st.markdown("#### 手動更新履歴")
        st.dataframe(manual_history_df, use_container_width=True, hide_index=True)


# ============================================================
# 2. 自動最適化
# ============================================================
with auto_tab:
    st.header("2. 自動最適化")

    st.markdown(
        r"""
最急降下法では，現在の点 $(a, b)$ から，勾配と反対方向へ進みます。

$$
    a_{k+1} = a_k - \eta \frac{\partial z}{\partial a}
$$

$$
    b_{k+1} = b_k - \eta \frac{\partial z}{\partial b}
$$

ここで，$\eta$ は学習率です。
"""
    )

    auto_col1, auto_col2 = st.columns([1, 2])

    with auto_col1:
        st.markdown("#### 最急降下法の設定")

        if st.session_state.last_manual is not None:
            default_a0 = float(st.session_state.last_manual["a"])
            default_b0 = float(st.session_state.last_manual["b"])
            st.info("初期値には，最後に履歴へ追加した手動最適化の値を使います。")
        else:
            default_a0 = 1.0
            default_b0 = 0.0
            st.info("手動最適化の履歴がないため，初期値は a=1, b=0 です。")

        a0 = st.number_input("初期値 a", value=default_a0, step=0.1, format="%.6f", key="auto_a0")
        b0 = st.number_input("初期値 b", value=default_b0, step=0.1, format="%.6f", key="auto_b0")

        learning_rate = st.number_input(
            "学習率 η",
            value=0.01,
            min_value=0.000001,
            step=0.001,
            format="%.6f",
            key="learning_rate",
        )

        max_iter = st.number_input(
            "最大反復回数",
            value=100,
            min_value=1,
            max_value=10000,
            step=10,
            key="max_iter",
        )

        tolerance = st.number_input(
            "停止条件：勾配の大きさ",
            value=1e-6,
            min_value=0.0,
            format="%.8f",
            key="tolerance",
        )

        if st.button("最急降下法を実行", type="primary", key="run_auto"):
            auto_history_df = gradient_descent(
                x=x,
                y=y,
                a0=a0,
                b0=b0,
                learning_rate=learning_rate,
                max_iter=int(max_iter),
                tolerance=tolerance,
            )
            st.session_state.auto_history = auto_history_df.to_dict("records")
            st.rerun()

        if st.button("自動最適化履歴をリセット", key="reset_auto"):
            st.session_state.auto_history = []
            st.rerun()

    auto_history_df = pd.DataFrame(st.session_state.auto_history)

    with auto_col2:
        if len(auto_history_df) > 0:
            last = auto_history_df.iloc[-1]

            st.markdown("#### 最終結果")
            result_cols = st.columns(4)
            result_cols[0].metric("a", f"{last['a']:.6f}")
            result_cols[1].metric("b", f"{last['b']:.6f}")
            result_cols[2].metric("z = MSE", f"{last['z']:.6f}")
            result_cols[3].metric("勾配の大きさ", f"{last['grad_norm']:.6e}")

            st.plotly_chart(
                plot_mse_surface(
                    x=x,
                    y=y,
                    history_df=auto_history_df,
                    current_row=last,
                    a_center=last["a"],
                    b_center=last["b"],
                    span_a=span_a,
                    span_b=span_b,
                    grid_size=grid_size,
                    title="z = MSE の曲面と最急降下法の更新履歴",
                    show_surface=True,
                ),
                use_container_width=True,
            )
        else:
            st.info("まだ自動最適化は実行されていません。")

    if len(auto_history_df) > 0:
        st.markdown("#### 最急降下法による更新履歴")
        st.dataframe(auto_history_df, use_container_width=True, hide_index=True)

        st.markdown("#### a, b の変化に伴う z の変化アニメーション")
        st.plotly_chart(
            make_mse_animation(auto_history_df),
            use_container_width=True,
        )

        st.markdown("#### a, b の変化に伴う回帰直線の変化アニメーション")
        st.plotly_chart(
            make_regression_animation(x, y, auto_history_df),
            use_container_width=True,
        )

        csv_history = auto_history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="自動最適化の履歴をCSVでダウンロード",
            data=csv_history,
            file_name="gradient_descent_history.csv",
            mime="text/csv",
        )
