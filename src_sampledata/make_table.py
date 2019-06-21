
from datetime import datetime
import glob
items={
    33: "Regular insulin dose",
    34: "NPH insulin dose",
    35: "UltraLente insulin dose",
    48: "Unspecified blood glucose measurement",
    57: "Unspecified blood glucose measurement",
    58: "Pre-breakfast blood glucose measurement",
    59: "Post-breakfast blood glucose measurement",
    60: "Pre-lunch blood glucose measurement",
    61: "Post-lunch blood glucose measurement",
    62: "Pre-supper blood glucose measurement",
    63: "Post-supper blood glucose measurement",
    64: "Pre-snack blood glucose measurement",
    66: "Typical meal ingestion",
    67: "More-than-usual meal ingestion",
    68: "Less-than-usual meal ingestion",
    69: "Typical exercise activity",
    70: "More-than-usual exercise activity",
    71: "Less-than-usual exercise activity",
    72: "Unspecified special event",
    }
item_list=[33,34,35,48,57,58,59,60,61,62,63,64,66,67,68,69,70,71,72]
target_name="Hypoglycemic symptoms"
target=65
save_filename="sample.tsv"

def discritize_time(t):
    t=t.replace(minute=0)
    t=t.replace(hour=0)
    return t

all_data={}
for filename in glob.glob('Diabetes-Data/data-*'):
    all_data[filename]=[]
    for line in open(filename):
        all_data[filename].append(line.strip().split("\t"))

all_time_data={}
for key,data in all_data.items():
    idx=key.rindex("-")
    patient_id=key[idx+1:]
    time_data={}
    for line in data:
        if len(line)==4:
            t=line[0]+" "+line[1]
            try:
                t = datetime.strptime(t, '%m-%d-%Y %H:%M')
            except:
                #何故かデータに06-31-1991がある
                continue
            t=discritize_time(t)
            if t not in time_data:
               time_data[t]={}
            try:
                x=float(line[3])
                time_data[t][int(line[2])]=x
            except:
                print("[E]",line[3])
                continue
    all_time_data[patient_id]=time_data

table_data=[]
for patient_id,time_data in all_time_data.items():
    for t,data in time_data.items():
        v=[]
        for item_code in item_list:
            if item_code in data:
                v.append(data[item_code])
            else:
                v.append("")
        flag=0
        if target in data:
            flag=1
        table_data.append([patient_id,t,flag]+v)

names=[items[v] for v in item_list]
print("[SAVE]",save_filename)
fp=open(save_filename,"w")
s="\t".join(["patient_id","time",target_name]+names)
fp.write(s)
fp.write("\n")
for v in table_data:
    s="\t".join(map(str,v))
    fp.write(s)
    fp.write("\n")


