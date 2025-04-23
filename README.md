📘 取扱説明書（更新版）
概要
本アプリは、**複数の内部温度（T_internal1～5）**から、それぞれの位置の表面温度（T_surface）を個別に予測するアプリです。
機械学習（ランダムフォレスト）を用いて、学習用データに基づいてモデルを構築し、予測対象CSVに含まれる全 T_internal 系列に対して予測を行います。

入力ファイル形式
📥 学習用CSV（複数ファイル対応）

time	T_internal	T_surface
📥 予測用CSV（1ファイル）

time	T_internal1	T_internal2	T_internal3	T_internal4	T_internal5
操作手順
① 学習用CSVをアップロード（複数可）
複数のCSVを選択可能（結合して学習）

各CSVは T_internal と T_surface を持っている必要あり

② 予測用CSVをアップロード（1ファイル）
列名が T_internal1 ～ T_internal5 の形であれば何列でもOK

モデルは各列ごとにラグ特徴量を作成し、個別に Predicted_T_surface_X を出力

出力結果CSV

time	Predicted_T_surface_T_internal1	...	Predicted_T_surface_T_internal5
推奨設定
ラグ数：20（2秒相当、内部で固定）

モデル：ランダムフォレスト回帰（精度・速度・ロバスト性のバランス）
