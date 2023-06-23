import subprocess
import os
import pickle
from tqdm import tqdm
import math

def TMalign(path1, path2):
    try:
        result = subprocess.run(["./TMalign", path1, path2], capture_output=True, text=True)
        ret = result.stdout.split('\n')
        ret = [r for r in ret if r.startswith('TM-score')]
        return float(ret[1].split()[1])
    except:
        return 0


'''
    in:
        pdbs_arr: 输入样本的pdbs文件路径list, 
        sp_label_arr: 预测的对应超族标签list
    
    out:
        这一批pdbs的平均TMScore
'''
def getTMScore(pdbs_arr, sp_id_arr):    
    id2label_path = '/home/tangwuguo/datasets/pdb_scope_id2label.pkl'    
    fa_path = './astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa'
    pkl_file = './fa.pkl'
    
    # load ground_spf dict
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            grouped_spf = pickle.load(f)
    else:        
        grouped_spf = load_fa(fa_path, dump_path=pkl_file)
    
    # id2label
    if os.path.exists(id2label_path)==False:
        print(f'Error: file-{id2label_path} not exist.')
        return
    with open(id2label_path, '+rb') as f:
        id2label_dict = pickle.load(f)
    spfs = [id2label_dict[id.item()] for id in sp_id_arr]
    
    # caculate avg TMScore  
    names = []
    scores = []
    sum_score = 0
    bar = tqdm(zip(pdbs_arr, spfs), ncols=100)
    for idx, item in enumerate(bar):
        score = 0
        o, spf = item[0].strip(), item[1].strip()
        if spf in grouped_spf:
            for p in grouped_spf[spf]:
                score = max(score, TMalign(o, p))        
        
        name = o[o.rfind('/')+1:]
        names.append(name)
        scores.append(score)
        sum_score += score        
        bar.set_description(f'Num{idx+1}/{len(pdbs_arr)}')
        bar.set_postfix({"name":name, "spf":spf, "score":score})
        bar.update()       
    score = sum_score / len(pdbs_arr)
    print('Average Score: ', score)
    return list(zip(names,scores)), score


def load_fa(fa_path, dump_path):    
    with open(fa_path, 'r+') as f:
        lines = f.readlines() # read all clustered lines
    lines = [l[1:].split(' ')[:2] for l in lines if l.startswith(">")] # extract all superfamily and PDB id
    pairs = [(key[:-2], value) for (value, key) in lines] # rearrange to (key, value) pairs
    key2paths = [(key, f"pdbstyle-2.08/{value[2:4]}/{value}.ent") for (key, value) in pairs] # parse the pdb file paths

    grouped_spf = {} # grouped superfamilies
    for k,p in key2paths:
        if k not in grouped_spf:
            grouped_spf[k] = []
        grouped_spf[k].append(p)
    
    with open(dump_path, 'wb') as f:
        pickle.dump(grouped_spf, f)    
    return grouped_spf

