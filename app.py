import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ======= モデル処理関数 =======

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

def predict_from_column(df, col_name, model, n_lags=20):
    df_temp = df[["time", col_name]].rename(columns={col_name: "T_internal"})
    X_test, df_lagged = prepare_predict_data(df_temp, n_lags)
    y_pred = model.predict(X_test)
    df_lagged[f"Predicted_T_surface_{col_name}"] = y_pred
    return df_lagged[["time", f"Predicted_T_surface_{col_name}"]]

# ======= Streamlit UI =======

st.set_page_config(page_title="T_surface 多点予測", layout="wide")
st.title("🌡️ T_surface 多点予測アプリ")

# 学習用CSVアップロード
st.header("1️⃣ 学習用CSVをアップロード（複数可）")
train_files = st.file_uploader("T_internal, T_surface を含む CSV を複数選択", type="csv", accept_multiple_files=True)

model = None
if train_files:
    dfs = []
    for idx, f in enumerate(train_files):
        df = pd.read_csv(f)
        if "T_internal" in df.columns and "T_surface" in df.columns:
            dfs.append(df)
        else:
            st.warning(f"{f.name} に必要な列がありません")
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        X_train, y_train = prepare_train_data(combined_df)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success(f"✅ 学習完了：{len(dfs)}ファイル、{len(X_train)} サンプル")

# 予測用CSVアップロード
st.header("2️⃣ 予測用CSVをアップロード（T_internal1〜5）")
test_file = st.file_uploader("予測対象のCSV", type="csv")

if model and test_file:
    test_df = pd.read_csv(test_file)
    time_col = test_df["time"] if "time" in test_df.columns else test_df.index

    # 内部温度列を検出
    internal_cols = [col for col in test_df.columns if col.startswith("T_internal")]
    if not internal_cols:
        st.error("T_internal1 ～ T_internal5 のような列が必要です")

    # 予測処理
    all_preds = []
    for col in internal_cols:
        pred_df = predict_from_column(test_df, col, model)
        all_preds.append(pred_df.set_index("time"))

    result_df = pd.concat(all_preds, axis=1).reset_index()

    # 表示
    st.subheader("📊 予測結果プレビュー")
    st.dataframe(result_df.head())

    # === グラフ（ペア別5分割） ===
    st.subheader("📈 各センサの内部温度と予測表面温度（ペア別グラフ）")

    fig, axes = plt.subplots(nrows=len(internal_cols), ncols=1, figsize=(10, 2.5 * len(internal_cols)), sharex=True)
    time = result_df["time"]

    for idx, col in enumerate(internal_cols):
        ax = axes[idx]
        pred_col = f"Predicted_T_surface_{col}"
        internal_trimmed = test_df[col][len(test_df) - len(time):].values

        ax.plot(time, internal_trimmed, label=f"{col}", color="tab:blue")
        ax.plot(time, result_df[pred_col], label=f"{pred_col}", color="tab:red", linestyle="--")
        ax.set_ylabel("温度 [°C]")
        ax.set_title(f"{col} vs Predicted Surface")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("時間 [s]")
    st.pyplot(fig)

    # === CSV出力 ===
    st.subheader("💾 予測結果CSVのダウンロード")
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 予測結果CSVをダウンロード",
        data=csv_bytes,
        file_name="predicted_surface_multi.csv",
        mime="text/csv"
    )
