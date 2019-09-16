
name=time_series_data
export  CUDA_VISIBLE_DEVICES=3
cd `dirname $0`

if [ ! -e DeepKF ]; then
git clone https://github.com/clinfo/DeepKF.git
fi


mkdir -p data
python convert_numpy.py --input_file ./sample.tsv -i 0 -t 1 -A 2 -H

mkdir -p ./DeepKF/${name}
rm -r ./DeepKF/${name}/data
mv data ./DeepKF/${name}/


#rm -r ./DeepKF/${name}/model
mkdir -p ./DeepKF/${name}/model
mkdir -p ./DeepKF/${name}/model/result/
mkdir -p ./DeepKF/${name}/model/sim/

cd DeepKF



cp ../src4dkf/config.tmpl.json ${name}/config.json

python dmm.py --config ${name}/config.json train,infer,filter,field
#python dmm.py --config ./${name}/config.result.json filter,field

python script/plot_p.py --config ./${name}/config.json all 
python script/plot.py --config ./${name}/config.json all
python script/plot_vec.py ./${name}/config.json all

