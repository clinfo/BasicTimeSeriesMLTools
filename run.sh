cd `dirname $0`

if [ ! -e Diabetes-Data ]; then
sh src_sampledata/get_sample_diabetes_data.sh
fi

if [ ! -e BasicMLTool ]; then
git clone https://github.com/kojima-r/BasicMLTool.git
fi

python src_sampledata/make_table.py 

python BasicMLTool/classifier.py --input_file ./sample.tsv -g 0 -I 1 -A 2 -H

