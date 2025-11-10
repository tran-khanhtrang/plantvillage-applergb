import os, csv, random, argparse, pathlib
def collect(root):
    items=[]
    for cls in sorted(os.listdir(root)):
        d=os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')):
                items.append((os.path.join(d,fn), cls))
    return items
def split(items, seed=42, ratios=(0.8,0.1,0.1)):
    random.Random(seed).shuffle(items)
    n=len(items); n_tr=int(n*ratios[0]); n_va=int(n*ratios[1])
    return items[:n_tr], items[n_tr:n_tr+n_va], items[n_tr+n_va:]
def write_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['path','label'])
        for p,lbl in rows: w.writerow([pathlib.Path(p).as_posix(), lbl])
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--outdir', default='manifests')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    items=collect(args.root)
    if not items: raise SystemExit(f'Không tìm thấy ảnh trong {args.root}')
    tr,va,te=split(items, seed=args.seed)
    write_csv(tr, os.path.join(args.outdir,'apple_train.csv'))
    write_csv(va, os.path.join(args.outdir,'apple_val.csv'))
    write_csv(te, os.path.join(args.outdir,'apple_test.csv'))
    classes=sorted({lbl for _,lbl in items})
    with open(os.path.join(args.outdir,'classes.txt'),'w',encoding='utf-8') as f: f.write('\n'.join(classes))
    print(f"Total={len(items)} | train={len(tr)}, val={len(va)}, test={len(te)}")
    print('Wrote manifests to', args.outdir)
