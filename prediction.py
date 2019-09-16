import pickle
import os
import argparse
import random
import numpy as np
import warnings
import pickle
import csv
import json
import sys
from multiprocessing import Pool
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.feature_selection import SelectFromModel
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.utils import resample
sys.path.append("BasicMLTool")
# this project
from util import load_data,NumPyArangeEncoder
from classifier import train_cv_one_fold
import dateutil.parser
import datetime

def series_train_cv_one_fold(arg):
    #x,y,h,one_kf,args=arg
    #print(x.shape)
    #print(x.shape)
    #print(y.shape)
    #train_idx,test_idx=one_kf
    #print(test_idx)
    return train_cv_one_fold(arg)

def ffill(array_x,array_m,array_s):
    #(n,max_length+1,n_attr)
    n=array_x.shape[0]
    step=array_x.shape[1]
    n_attr=array_x.shape[2]
    z=np.zeros((n,n_attr),dtype=np.float32)
    z[:,:]=array_x[:,0,:]
    for s in range(step-1):
        temp=array_x[:,s+1,:]
        temp[np.isnan(temp)]=z[np.isnan(temp)]
        # mask
        temp_m=array_m[:,s+1,:]
        temp_m[~np.isnan(temp)]=1
        # next
        z[:,:]=array_x[:,s+1,:]
    for i,s in enumerate(array_s):
        array_m[i,s:,:]=0
    return array_x

def ffill_one_series(array_x,array_m):
    #(n_step,n_attr)
    step=array_x.shape[0]
    n_attr=array_x.shape[1]
    z=np.zeros((n_attr,),dtype=np.float32)
    z[:]=array_x[0,:]
    for s in range(step-1):
        temp=array_x[s+1,:]
        temp[np.isnan(temp)]=z[np.isnan(temp)]
        # mask
        temp_m=array_m[s+1,:]
        temp_m[~np.isnan(temp)]=1
        # next
        z[:]=array_x[s+1,:]
    array_m[s:,:]=0
    return array_x

def build_sequence_dict(args,x,y,opt,h):
    data_x={}
    data_y={}
    data_g={}
    data_t={}
    data_sid=[]
    for i,sid in enumerate(opt["series_id"]):
        if sid not in data_x:
            data_x[sid]=[]
            data_y[sid]=[]
            if args.group is not None:
                data_g[sid]=opt["group"][i]
            data_sid.append(sid)
            if "series_time" in opt:
                data_t[sid]=[]
        data_x[sid].append(x[i])
        if len(y)>0:
            data_y[sid].append(y[i])
        if sid in data_t:
            data_t[sid].append(opt["series_time"][i])
    return data_sid,data_x,data_y,data_g,data_t

##############################################
# --- 学習処理の全体                     --- #
##############################################
def run_train(args):
    all_result={}
    model_result=[]
    for filename in args.input_file:
        print("=================================")
        print("== Loading data ... ")
        print("=================================")
        option={}
        if args.series_time is not None:
            option["series_time"]=args.series_time
        if args.series_id is not None:
            option["series_id"]=args.series_id
        if args.group is not None:
            option["group"]=args.group
        x,y,opt,h=load_data(filename,ans_col=args.answer,ignore_col=args.ignore,header=args.header,cat_col=args.categorical,option=option)
        data_sid,data_x,data_y,data_g,data_t=build_sequence_dict(args,x,y,opt,h)
        ### display
        n_attr=x.shape[1]
        max_length=max([len(x) for _,x in data_x.items()])
        min_length=min([len(x) for _,x in data_x.items()])
        n=len(data_sid)
        print("the number of sequences:",n)
        print("the number of attributes:",n_attr)
        print("maximum length:",max_length)
        print("minimum length:",min_length)

        if args.prediction_interval:
            args.AR=True
            args.prediction=1
            args.task="regression"
        ## インターバル特徴量を作成
        for i,sid in enumerate(data_sid):
            if sid in data_t:
                seq_t=data_t[sid]
                tt=[dateutil.parser.parse(el) for el in seq_t]
                if args.delta_time_days:
                    delta_time=datetime.timedelta(days=args.delta_time_days)
                elif args.delta_time_hours:
                    delta_time=datetime.timedelta(hours=args.delta_time_hours)
                elif args.delta_time_minutes:
                    delta_time=datetime.timedelta(days=args.delta_time_minutes)
                else:
                    delta_time=datetime.timedelta(days=1)
                dtt=[el/delta_time for el in np.diff(tt)]
                seq_dt=np.concatenate([[0],dtt],axis=0)
                seq_dt=np.reshape(seq_dt,(-1,1))
                #
                seq=data_x[sid]
                seq_new=np.concatenate([seq,seq_dt], axis=1)
                data_x[sid]=seq_new
                #
                if args.prediction_interval:
                    deq_dt_y=np.concatenate([dtt,[np.nan]],axis=0)
                    data_y[sid]=deq_dt_y
        h+=["Interval"]


        ## 過去の予測対象も入力に含める
        if args.prediction is not None:
            for i,sid in enumerate(data_sid):
                seq=data_x[sid]
                seq_y=np.reshape(np.array(data_y[sid]),(-1,1))
                if args.AR:
                    seq_new=seq_y
                else:
                    seq_new=np.concatenate([seq,seq_y], axis=1)
                y_new=np.empty((seq_y.shape[0],),dtype=np.float32)
                y_new[:]=np.nan
                y_new[:seq_y.shape[0]-args.prediction]=seq_y[args.prediction:,0]
                # leaked pattern
                #y_new[:seq_y.shape[0]-args.prediction+1]=seq_y[args.prediction-1:,0]
                data_x[sid]=seq_new
                data_y[sid]=y_new
        
        if args.AR:
            h=["y"]
        else:
            h+=["y"]
        
        ## Imputation(前方)
        for i,sid in enumerate(data_sid):
            data_x[sid]=np.array(data_x[sid])
            #prev_missing_count=np.sum(np.isnan(data_x[sid]))
            array_m=np.zeros_like(data_x[sid])
            ffill_one_series(data_x[sid],array_m)
            #next_missing_count=np.sum(np.isnan(data_x[sid]))
            #print(sid,prev_missing_count,"=>",next_missing_count)

        ## Imputation(平均値)
        tbl_x=[]
        for i,sid in enumerate(data_sid):
            for vec in data_x[sid]:
                tbl_x.append(vec)
        z=np.nanmean(tbl_x,axis=0)
        for i,sid in enumerate(data_sid):
            #prev_missing_count=np.sum(np.isnan(data_x[sid]))
            seq=data_x[sid]
            for t in range(seq.shape[0]):
                seq[t,np.isnan(seq[t,:])]=z[np.isnan(seq[t,:])]
            #next_missing_count=np.sum(np.isnan(data_x[sid]))
            #print(sid,prev_missing_count,"=>",next_missing_count)
        
           
        ## Window で区切ってテーブル作成
        table_x=[]
        table_y=[]
        table_g=[]
        table_t=[]
        table_sid=[]
        window=args.window
        for i,sid in enumerate(data_sid):
            n_step=data_x[sid].shape[0]
            if n_step>=window:
                for j in range(n_step-window+1):
                    if not np.isnan(data_y[sid][j+window-1]):
                        temp_x=np.reshape(data_x[sid][j:j+window,:],(-1,))
                        table_x.append(temp_x)
                        table_y.append(data_y[sid][j+window-1])
                        if sid in data_g:
                            table_g.append(data_g[sid])
                        else:
                            table_g.append(sid)
                        table_t.append(data_t[sid][j+window-1])
        ## Window で区切ってテーブル作成
        ## ヘッダに追加
        new_h=[]
        for w in range(window):
            for el in h:
                if args.prediction:
                    new_h.append(el+"_w("+str(w-window+1-args.prediction)+")")
                else:
                    new_h.append(el+"_w("+str(w-window+1)+")")
        h=new_h
        ## 最終的なデータサイズ表示
        x=np.array(table_x)
        y=np.array(table_y)
        g=np.array(table_g)
        print("x:",x.shape)
        print("y:",y.shape)
        print("g:",g.shape)
        cnt={}
        for e in table_y:
            if e not in cnt:
                cnt[e]=0
            cnt[e]+=1
        print("label:",cnt)
        ## データから２クラス問題か多クラス問題化を決めておく
        if args.task=="auto":
            if len(np.unique(y))==2:
                args.task="binary"
            else:
                args.task="multiclass"
        if args.task!="regression":
            y=y.astype(dtype=np.int64)
        
        ##
        ## cross-validation を並列化して行う
        ##
        print("=================================")
        print("== Starting cross-validation ... ")
        print("=================================")
        kf=sklearn.model_selection.GroupKFold(n_splits=args.splits)
        results=[]
        for s in kf.split(x,y,g):
            result=series_train_cv_one_fold((x,y,h,s,args))
            results.append(result)
#pool = Pool(processes=args.splits)
#results = pool.map(series_train_cv_one_fold, [(x,y,h,s,args) for s in kf.split(x,y,g)])
        
        ##
        ## cross-validation の結果をまとめる
        ## ・各評価値の平均・標準偏差を計算する
        ##
        cv_result={"cv": [r[0] for r in results]}
        model_result.append([r[1] for r in results])
        print("=================================")
        print("== Evaluation ... ")
        print("=================================")
        if args.task=="regression":
            score_names=["r2","mse"]
        else:
            score_names=["accuracy","f1","precision","recall","auc"]
        for score_name in score_names:
            scores=[r[0][score_name] for r in results]
            test_mean = np.nanmean(np.asarray(scores))
            test_std = np.nanstd(np.asarray(scores))
            print("Mean %10s on test set: %3f (standard deviation: %3s)"
                % (score_name,test_mean,test_std))
            cv_result[score_name+"_mean"]=test_mean
            cv_result[score_name+"_std"]=test_std
        ##
        ## 全体の評価
        ##
        test_y=[]
        pred_y=[]
        for result in cv_result["cv"]:
            test_y.extend(result["test_y"])
            pred_y.extend(result["pred_y"])
        if args.task!= "regression":
            conf=sklearn.metrics.confusion_matrix(test_y, pred_y)
            cv_result["confusion"]=conf
        cv_result["task"]=args.task
        ##
        ## 結果をディクショナリに保存して返値とする
        ##
        all_result[filename]=cv_result
    return all_result,model_result


############################################################
# --- mainの関数：コマンド実行時にはここが呼び出される --- #
############################################################
if __name__ == '__main__':
    ##
    ## コマンドラインのオプションの設定
    ##
    parser = argparse.ArgumentParser(description = "Classification")
    parser.add_argument("--grid_search",default=False,
        help = "enebled grid search", action="store_true")
    parser.add_argument("--feature_selection",default=False,
        help = "enabled feature selection", action="store_true")
    parser.add_argument("--input_file","-f",nargs='+',default=None,
        help = "input filename (txt/tsv/csv)", type = str)
    parser.add_argument("--trials",default=3,
        help = "Trials for hyperparameters random search", type = int)
    parser.add_argument("--splits","-s", default=5,
        help = "number of splits for cross validation", type = int)
    parser.add_argument("--param_search_splits","-p", default=3,
        help = "number of splits for parameter search", type = int)
    parser.add_argument('--header','-H',default=False,
        help = "number of splits", action='store_true')
    parser.add_argument('--answer','-A',
        help = "column number of answer label", type=int)
    parser.add_argument('--categorical','-C',nargs='*',default=[],
        help = "column numbers for categorical data", type=int)
    parser.add_argument('--ignore','-I',nargs='*',default=[],
        help = "column numbers for ignored data", type=int)
    parser.add_argument("--model",default="rf",
        help = "method (rf/svm/rbf_svm/lr)", type = str)
    parser.add_argument("--task",default="auto",
        help = "task type (auto/binary/multiclass/regression)", type = str)
    parser.add_argument('--output_json',default=None,
        help = "output: json", type=str)
    parser.add_argument('--output_csv',default=None,
        help = "output: csv", type=str)
    parser.add_argument('--output_model',default=None,
        help = "output: pickle", type=str)
    parser.add_argument('--seed',default=20,
        help = "random seed", type=int)
    parser.add_argument('--num_features',default=None,
        help = "select features", type=int)
    parser.add_argument("--fci",default=False,
        help = "enabled forestci", action="store_true")
    parser.add_argument('--data_sample',default=None,
        help = "re-sample data", type=int)
    parser.add_argument('--series_time','-t',default=None,
        help = "column number of time step (Time series)", type=int)
    parser.add_argument('--series_id','-i',default=None,
        help = "column number of time (Time series)", type=int)
    parser.add_argument('--group','-g',default=None,
        help = "column number of group", type=int)
    parser.add_argument('--window','-w',default=1,
        help = "window size for time series prediction", type=int)
    parser.add_argument('--prediction',default=None,
        help = "k-step look-ahead prediction in time series prediction", type=int)
    parser.add_argument('--prediction_interval',default=False,
        help = "interval prediction in time series prediction", action="store_true")
    parser.add_argument("--AR",default=False,
        help = "auto regression model", action="store_true")
    parser.add_argument('--delta_time_days',default=None,
        help = "delta time in time series prediction", type=int)
    parser.add_argument('--delta_time_hours',default=None,
        help = "delta time in time series prediction", type=int)
    parser.add_argument('--delta_time_minutes',default=None,
        help = "delta time in time series prediction", type=int)
    
    ##
    ## コマンドラインのオプションによる設定はargsに保存する
    ##
    args = parser.parse_args()
    ##
    ## 乱数初期化
    ##
    np.random.seed(args.seed) 

    ##
    ## 学習開始
    ##
    all_result,model_result=run_train(args)
    ##
    ## 結果を簡易に表示
    ##
    if args.task=="regression":
        score_names=["r2","mse"]
    else:
        score_names=["accuracy","auc"]
    print("=================================")
    print("== summary ... ")
    print("=================================")
    metrics_names=sorted([m+"_mean" for m in score_names]+[m+"_std" for m in score_names])
    print("\t".join(["filename"]+metrics_names))
    for key,o in all_result.items():
        arr=[key]
        for name in metrics_names:
            arr.append("%2.4f"%(o[name],))
        print("\t".join(arr))
        
    ##
    ## 結果をjson ファイルに保存
    ## 予測結果やcross-validationなどの細かい結果も保存される
    ##
    if args.output_json:
        print("[SAVE]",args.output_json)
        fp = open(args.output_json, "w")
        json.dump(all_result,fp, indent=4, cls=NumPyArangeEncoder)
    
    ##
    ## 結果をcsv ファイルに保存
    ##
    if args.output_csv:
        print("[SAVE]",args.output_csv)
        fp = open(args.output_csv, "w")
        if args.task=="regression":
            score_names= ["r2","mse"]
        else:
            score_names= ["accuracy","f1","precision","recall","auc"]
        metrics_names=sorted([m+"_mean" for m in score_names]+[m+"_std" for m in score_names])
        fp.write("\t".join(["filename"]+metrics_names))
        fp.write("\n")
        for key,o in all_result.items():
            arr=[key]
            for name in metrics_names:
                arr.append("%2.4f"%(o[name],))
            fp.write("\t".join(arr))
            fp.write("\n")
    ##
    ## 学習済みモデルをpickle ファイルに保存
    ##
    if args.output_model:
        print("[SAVE]",args.output_model)
        with open(args.output_model, 'wb') as f:
            pickle.dump(model_result, f)
    
