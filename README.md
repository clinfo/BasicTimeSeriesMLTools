# BasicTimeSeriesMLTools
教育・簡易的な実験目的の時系列機械学習手法の実装です。

# 環境
scikit-learn やnumpyのインストールされた環境であれば動くはずです。オススメはanaconda環境です。

関連リポジトリ
- https://github.com/kojima-r/BasicMLTool
- https://github.com/clinfo/DeepKF

上記関連リポジトリのアルゴリズムを動かすには、それらの環境条件も必要です

## サンプルの動かし方(各時刻でのクラス分類問題)
Linux環境であれば`run.sh`スクリプトを以下のように実行すれば可能です。
```
 sh run.sh
``` 
内部的には以下の1~3の操作が行われます
### 1. サンプルデータをダウンロード
UCIリポジトリからデータをダウンロードしてきます
```
sh src_sampledata/get_sample_diabetes_data.sh
``` 

### 2. サンプルデータを整形して、sample.tsvを作成
元データから成形されたテーブルデータを作成します
```
python src_sampledata/make_table.py
```

### 3. 前処理により作られるファイル
機械学習ツールを使って設定されたラベルを予測します。
デフォルトでは、5-fold cross-validationのRandomForestが実行されます。
```
python BasicMLTool/classifier.py --input_file ./sample.tsv -g 0 -I 1 -A 2 -H
```

### 自前データで上記サンプルを動かすには？
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


## サンプルの動かし方(予測問題)

上記の方法でsample.tsvに相当するファイルが作成済みであることを想定します。
以下のコマンドで予測問題を解くことができます。

```
python prediction.py --input_file ./sample.tsv -i 0 -t 1 -A 2 -H --window 3  --prediction 1 --result_csv out.csv
```

- sample.tsvは第0列がID`-i 0` 
- sample.tsvは第1列が時刻`-t 1`
- 予測対象は第2列: `-A 2`
- 過去3ステップ分を利用: `--window 3`
- 予測は１ステップ先：`--prediction 1` 
- 結果をout.csvに保存する：`--result_csv out.csv`

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

## オプション一覧
### 入力ファイルの列を指定する
- `--series_time` `-t`
時刻を表す列番号を指定する

- `--series_id` `-i`
系列のIDを表す列番号を指定する

- `--group` `-g`
系列のグループを表す列番号を指定する

- `--window` `-w`
予測時に過去何ステップ分を入力特徴量に用いるかを指定する。

- `--prediction`
何ステップ先を予測するかを指定する

- `--prediction_interval`
1ステップ先の時間間隔を予測する

- `--AR`
自己回帰(auto regression)の問題を解く（入力特徴量に出力と同じ特徴量のみを利用する）

- `--delta_time_days`
1ステップにあたる単位時間を日単位で指定する。デフォルトは1日。
- `--delta_time_hours`
1ステップにあたる単位時間を時間単位で指定する。
- `--delta_time_minutes`
1ステップにあたる単位時間を分単位で指定する。


