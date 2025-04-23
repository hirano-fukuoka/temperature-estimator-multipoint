import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- 時系列処理用関数 ---
def create_lag_features(df, n_lags=20):
    for lag in range(1, n_lags + 1):
        df[f"T_internal_lag{lag}"] = df["T_internal"].shift(lag)
    return df.dropna()

def prepare_train_data(df, n_lags=20):
    df_lag = create_lag_features(df.copy(), n_lags)
    X = df_lag[[f"T_internal_lag{i}" for i in range(1, n_lags + 1)]]
    y = df_lag["T_surface"]
    return X, y

def prepare_predict_data(df, n_lags=20):
    df_lag = create_lag_features(df.copy(), n_lags)
    X = df_lag[[f"T_internal_lag{i}" for i in range(1, n_lags + 1)]]
    return X, df_lag

def extract_cycles(df, start_col, lag_sec, span_sec, sampling=0.1):
    starts = df[df[start_col].diff() == 1].index
    lag_steps = int(lag_sec / sampling)
    span_steps = int(span_sec / sampling)

    segments = []
    for s in starts:
        t_start = s + lag_steps
        t_end = t_start + span_steps
        if t_end < len(df):
            segment = df.iloc[t_start:t_end].copy()
            segment["cycle_id"] = s
            segments.append(segment)
    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()

def predict_from_column(df, col_name, model, lag=20):
    df_temp = df[["time", col_name]].rename(columns={col_name: "T_internal"})
    X_test, df_lagged = prepare_predict_data(df_temp, n_lags=lag)
    y_pred = model.predict(X_test)
    df_lagged[f"Predicted_T_surface_{col_name}"] = y_pred
    return df_lagged[["time", f"Predicted_T_surface_{col_name}"]]

# --- Streamlit UI ---
st.set_page_config(page_title="T_surface 多点予測", layout="wide")
st.title("🌡️ 成形サイクル対応 T_surface 多点予測アプリ")

# ラグ・予測時間設定
st.sidebar.header("⏱️ 時間設定")
lag_seconds = st.sidebar.number_input("表面温度の立ち上がりまでのラグ秒数", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
predict_duration = st.sidebar.number_input("予測する時間範囲（秒）", min_value=10.0, max_value=120.0, value=55.0, step=1.0)
sampling_rate = 0.1  # 固定値、必要に応じて変更可
n_lags = 20

# 学習データ
st.header("1️⃣ 学習用データ（複数CSV, T_internal, T_surface, start_signal）")
train_files = st.file_uploader("CSVファイルを選択", type="csv", accept_multiple_files=True)
model = None

if train_files:
    train_segments = []
    for f in train_files:
        df = pd.read_csv(f)
        if set(["T_internal", "T_surface", "start_signal"]).issubset(df.columns):
            seg = extract_cycles(df, "start_signal", lag_seconds, predict_duration, sampling_rate)
            train_segments.append(seg)
        else:
            st.warning(f"{f.name} に必要な列がありません")
    
    if train_segments:
        train_df = pd.concat(train_segments, ignore_index=True)
        X_train, y_train = prepare_train_data(train_df, n_lags)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success(f"✅ 学習完了：{len(train_segments)} サイクル、{len(X_train)} サンプル")
    else:
        st.error("有効な学習データが見つかりませんでした")

# 予測データ
st.header("2️⃣ 予測対象データ（1ファイル, T_internal1～5, start_signal）")
test_file = st.file_uploader("予測CSVを選択", type="csv")

if model and test_file:
    test_df = pd.read_csv(test_file)
    internal_cols = [col for col in test_df.columns if col.startswith("T_internal")]
    if "start_signal" not in test_df.columns:
        st.error("start_signal 列が必要です")
    elif not internal_cols:
        st.error("T_internal1～T_internal5 のような列が必要です")
    else:
        # 各センサの予測
        result_segments = []
        for col in internal_cols:
            cycles = extract_cycles(test_df[["time", col, "start_signal"]].rename(columns={col: "T_internal"}),
                                    "start_signal", lag_seconds, predict_duration, sampling_rate)
            if not cycles.empty:
                pred_df = predict_from_column(cycles, "T_internal", model, lag=n_lags)
                pred_df.rename(columns={f"Predicted_T_surface_T_internal": f"Predicted_T_surface_{col}"}, inplace=True)
                result_segments.append(pred_df.set_index("time"))

        result_df = pd.concat(result_segments, axis=1).reset_index()

        # 表示
        st.subheader("📊 予測結果")
        st.dataframe(result_df.head())

        # グラフ表示（5センサ）
        st.subheader("📈 各センサの T_internal と予測 T_surface")
        fig, axes = plt.subplots(nrows=len(internal_cols), ncols=1, figsize=(10, 2.5 * len(internal_cols)), sharex=True)
        for idx, col in enumerate(internal_cols):
            ax = axes[idx]
            pred_col = f"Predicted_T_surface_{col}"
            internal_trimmed = test_df[col][len(test_df) - len(result_df):].values
            ax.plot(result_df["time"], internal_trimmed, label=col, color="tab:blue")
            ax.plot(result_df["time"], result_df[pred_col], label=pred_col, color="tab:red", linestyle="--")
            ax.set_ylabel("温度 [°C]")
            ax.set_title(f"{col} vs Predicted")
            ax.legend()
        axes[-1].set_xlabel("時間 [s]")
        st.pyplot(fig)

        # 出力
        st.subheader("💾 予測CSVのダウンロード")
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 ダウンロード", data=csv, file_name="predicted_surface.csv", mime="text/csv")
