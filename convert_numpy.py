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
def series_train_cv_one_fold(arg):
	x,y,h,one_kf,args=arg
	print(x.shape)
	print(y.shape)
	train_idx,test_idx=one_kf
	print(test_idx)


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
		#if args.data_sample is not None:
		#	x,y,g = resample(x,y,g,n_samples=args.data_sample)

		data_x={}
		data_y={}
		data_g={}
		data_sid=[]
		for i,sid in enumerate(opt["series_id"]):
			if sid not in data_x:
				data_x[sid]=[]
				data_y[sid]=[]
				if args.group is not None:
					data_g[sid]=opt["group"][i]
				data_sid.append(sid)
			data_x[sid].append(x[i])
			data_y[sid].append(y[i])
		n_attr=x.shape[1]
		max_length=max([len(x) for _,x in data_x.items()])
		min_length=min([len(x) for _,x in data_x.items()])
		n=len(data_sid)
		print("the number of sequences:",n)
		print("the number of attributes:",n_attr)
		print("maximum length:",max_length)
		print("minimum length:",min_length)
		array_x=np.zeros((n,max_length+1,n_attr),dtype=np.float32)
		array_m=np.zeros((n,max_length+1,n_attr),dtype=np.float32)
		array_y=np.zeros((n,max_length+1),dtype=np.float32)
		array_s=np.zeros((n,),dtype=np.int32)
		info={"id":[],"attr":[],"group":[]}
		info["id"]=opt["series_id"]
		for i,sid in enumerate(data_sid):
			x=data_x[sid]
			x=np.array(x)
			array_x[i,:len(x),:]=x
			m=array_m[i,:len(x),:]
			m[~np.isnan(x)]=1
			y=data_y[sid]
			array_y[i,:len(y)]=np.array(y)
			array_s[i]=len(x)
			if args.group is not None:
				info["group"].append(data_g[sid])

		## Imputation
		prev_missing_count=np.sum(np.isnan(array_x))
		#print(np.sum(np.isnan(array_x)))
		#print(np.sum(array_m))
		ffill(array_x,array_m,array_s)
		missing_count=np.sum(np.isnan(array_x))
		z=np.nanmean(array_x,axis=(0,1))
		zz=np.zeros_like(array_x)
		zz[:,:,:]=z
		array_x[np.isnan(array_x)]=zz[np.isnan(array_x)]
		#print(np.sum(np.isnan(array_x)))
		#print(np.sum(array_m))

		## Save
		path="data/"
		filename=path+"data.npy"
		np.save(filename,array_x)
		print("[SAVE]",filename)
		filename=path+"mask.npy"
		np.save(filename,array_m)
		print("[SAVE]",filename)
		filename=path+"label.npy"
		np.save(filename,array_y)
		print("[SAVE]",filename)
		filename=path+"steps.npy"
		np.save(filename,array_s)
		print("[SAVE]",filename)
		filename=path+"info.json"
		fp=open(filename,"w")
		json.dump(info,fp)
		print("[SAVE]",filename)
		x=array_x
		y=array_y
		g=np.array(info["group"])
		quit()
		## 欠損値を補完(平均)
		"""
		m=np.nanmean(x,axis=0)
		h=np.array(h)[~np.isnan(m)]
		print(len(h))
		print(h)
		imr = Imputer(missing_values=np.nan, strategy='mean', axis=0)
		x = imr.fit_transform(x)
		"""
		print("x:",x.shape)
		print("y:",y.shape)
		## 標準化
		#sc = StandardScaler()
		#x = sc.fit_transform(x)
		
		#if g is not None:
		#	print("g:",g.shape)
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
		if args.group is not None:
			kf=sklearn.model_selection.GroupKFold(n_splits=args.splits)
			pool = Pool(processes=args.splits)
			results = pool.map(series_train_cv_one_fold, [(x,y,h,s,args)for s in kf.split(x,y,info["group"])])
		else:
			kf=sklearn.model_selection.KFold(n_splits=args.splits, shuffle=True)
			pool = Pool(processes=args.splits)
			results = pool.map(series_train_cv_one_fold, [(x,y,h,s,args)for s in kf.split(x)])

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
	
