# BasicTimeSeriesMLTools
教育・簡易的な実験目的の時系列機械学習手法の実装です。

# 環境
scikit-learn やnumpyのインストールされた環境であれば動くはずです。オススメはanaconda環境です。

関連リポジトリ
- https://github.com/kojima-r/BasicMLTool
- https://github.com/clinfo/DeepKF

## 環境

scikit-learn やnumpyのインストールされた環境であれば動くはずです。オススメはanaconda環境です。
- 上記関連リポジトリのアルゴリズムを動かすには、それらの環境条件も必要です

## サンプルの動かし方
Linux環境であれば`run.sh`スクリプトを以下のように実行すれば可能です。
```
 sh run.sh
``` 
内部的には以下の操作が行われます
### サンプルデータをダウンロード
UCIリポジトリからデータをダウンロードしてきます

### サンプルデータを整形して、sample.tsvを作成
元データから成形されたテーブルデータを作成します

### 前処理により作られるファイル
機械学習ツールを使って設定されたラベルを予測します。
デフォルトでは、5-fold cross-validationのRandomForestが実行されます。

## 自前データで上記サンプルを動かすには？
自前のデータを使うためにはsample.tsvに相当するファイルを作成する必要があります。

- sample.tsvは第０列がIDなのでグルーピングを`-g 0`
- 第１列が時間情報ですが、このサンプルでは時間情報を使わないので第1列を無視するために`-g 0` 
- 第２列がラベルなので、`-A 2`
- このサンプルファイルにはヘッダがあるので`-H`
のようにオプションを適宜、自前でつけて、以下のコマンドを実行してください
```
python BasicMLTool/classifier.py --input_file ./sample.tsv -g 0 -I 1 -A 2 -H
```
その他、オプションの詳細は https://github.com/kojima-r/BasicMLTool を参照すること

## DeepKFサンプルの動かし方
```
./sample.tsv
```
がある状態で
```
run_dkf.sh
```
を実行する。
（途中、githubのアカウント・パスワードを求められるかもしれません）
結果は、DeepKF/time_series_data/以下に保存されます。

### 内部的には
以下の手順で実行されています
- https://github.com/clinfo/DeepKF から必要であればファイルをclone（githubのアカウント・パスワードを求められるかもしれません）
- clf.pyを用いて、sample.tsvをDeepKF用のファイルに変換
- DeepKF用の設定ファイル(src4dkf/config.tmpl.json)をコピー
- DeepKFを実行する
- DeepKFの実行結果をプロットする

