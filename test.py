import subprocess
import sys

def TMalign(path1, path2):
    try:
        result = subprocess.run(["./TMalign", path1, path2], capture_output=True, text=True)
        ret = result.stdout.split('\n')
        ret = [r for r in ret if r.startswith('TM-score')]
        return float(ret[1].split()[1])
    except:
        return 0

def main():
    with open('astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa') as f:
        lines = f.readlines() # read all clustered lines
    lines = [l[1:].split(' ')[:2] for l in lines if l.startswith(">")] # extract all superfamily and PDB id
    pairs = [(key[:-2], value) for (value, key) in lines] # rearrange to (key, value) pairs
    key2paths = [(key, f"pdbstyle-2.08/{value[2:4]}/{value}.ent") for (key, value) in pairs] # parse the pdb file paths

    grouped_spf = {} # grouped superfamilies
    for k,p in key2paths:
        if k not in grouped_spf:
            grouped_spf[k] = []
        grouped_spf[k].append(p)


    ori, tgt = sys.argv[1], sys.argv[2]
    with open(ori) as ori_file, open(tgt) as tgt_file:
        ori_paths, spfs = ori_file.readlines(), tgt_file.readlines()
        scores = []
        for o, spf in zip(ori_paths, spfs):
            score = 0
            o, spf = o.strip(), spf.strip()
            if spf in grouped_spf:
                for p in grouped_spf[spf]:
                    score = max(score, TMalign(o, p))

            print('Score for ', o, ' is ', score)
            scores.append(score)
        print('Average Score: ', sum(scores) / len(ori_paths))

if __name__ == "__main__":
    main()
