import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- 学習＆予測用ヘルパー関数 ---
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

def extract_window_from_starts(df, start_col, lag_sec, duration_sec, sampling=0.1):
    starts = df[df[start_col].diff() == 1].index
    lag_steps = int(lag_sec / sampling)
    duration_steps = int(duration_sec / sampling)
    segments = []
    for s in starts:
        t_start = s + lag_steps
        t_end = t_start + duration_steps
        if t_end <= len(df):
            segments.append(df.iloc[t_start:t_end].copy())
    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()

def predict_from_column(df, col_name, model, lag=20):
    df_temp = df[["time", col_name]].rename(columns={col_name: "T_internal"})
    X_test, df_lagged = prepare_predict_data(df_temp, lag)
    y_pred = model.predict(X_test)
    df_lagged[f"Predicted_T_surface_{col_name}"] = y_pred
    return df_lagged[["time", f"Predicted_T_surface_{col_name}"]]

# --- Streamlit UI ---
st.set_page_config(page_title="T_surface 多点予測", layout="wide")
st.title("🌡️ 成形サイクル対応 T_surface 多点予測アプリ")

# ラグ・予測時間設定
st.sidebar.header("⏱️ 時間設定")
lag_seconds = st.sidebar.number_input("表面温度の立ち上がりラグ（秒）", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
duration_seconds = st.sidebar.number_input("予測したい時間範囲（秒）", min_value=5.0, max_value=120.0, value=55.0, step=1.0)
sampling_rate = 0.1
n_lags = 20

# === 1. 学習用データ ===
st.header("1️⃣ 学習データアップロード（複数可）")
train_files = st.file_uploader("T_internal, T_surface, start_signal を含むCSV", type="csv", accept_multiple_files=True)

model = None
if train_files:
    segments = []
    for f in train_files:
        df = pd.read_csv(f)
        if set(["T_internal", "T_surface", "start_signal"]).issubset(df.columns):
            seg = extract_window_from_starts(df, "start_signal", lag_seconds, duration_seconds, sampling_rate)
            segments.append(seg)
        else:
            st.warning(f"{f.name} に必要な列が見つかりません")

    if segments:
        train_df = pd.concat(segments, ignore_index=True)
        X_train, y_train = prepare_train_data(train_df, n_lags)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success(f"✅ モデル学習完了：{len(segments)}サイクル、{len(X_train)}サンプル")
    else:
        st.error("⚠️ 有効な学習セグメントが見つかりませんでした")

# === 2. 予測データ ===
st.header("2️⃣ 予測対象データアップロード")
test_file = st.file_uploader("T_internal1〜5, start_signal を含むCSV", type="csv")

if model and test_file:
    df_test = pd.read_csv(test_file)
    if "start_signal" not in df_test.columns:
        st.error("start_signal 列がありません")
    else:
        internal_cols = [col for col in df_test.columns if col.startswith("T_internal")]
        if not internal_cols:
            st.error("T_internal1〜5 列が必要です")
        else:
            all_preds = []
            for col in internal_cols:
                df_temp = df_test[["time", col, "start_signal"]].rename(columns={col: "T_internal"})
                segments = extract_window_from_starts(df_temp, "start_signal", lag_seconds, duration_seconds, sampling_rate)
                if not segments.empty:
                    pred_df = predict_from_column(segments, "T_internal", model, lag=n_lags)
                    pred_df.rename(columns={f"Predicted_T_surface_T_internal": f"Predicted_T_surface_{col}"}, inplace=True)
                    all_preds.append(pred_df.set_index("time"))

            if all_preds:
                result_df = pd.concat(all_preds, axis=1).reset_index()

                # === 表示 ===
                st.subheader("📊 予測結果プレビュー")
                st.dataframe(result_df.head())

                # === グラフ ===
                st.subheader("📈 入力 vs 予測（各センサ）")
                fig, axes = plt.subplots(len(internal_cols), 1, figsize=(10, 2.5 * len(internal_cols)), sharex=True)
                time = result_df["time"]
                for i, col in enumerate(internal_cols):
                    ax = axes[i]
                    pred_col = f"Predicted_T_surface_{col}"
                    original_trimmed = df_test[col][len(df_test) - len(time):].values
                    ax.plot(time, original_trimmed, label=col, color="tab:blue")
                    ax.plot(time, result_df[pred_col], label=pred_col, color="tab:red", linestyle="--")
                    ax.set_ylabel("温度 [°C]")
                    ax.set_title(f"{col} vs 予測")
                    ax.legend()
                axes[-1].set_xlabel("時間 [s]")
                st.pyplot(fig)

                # === CSV 出力 ===
                st.subheader("💾 予測CSVのダウンロード")
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 ダウンロード", data=csv_bytes, file_name="predicted_surface.csv", mime="text/csv")
