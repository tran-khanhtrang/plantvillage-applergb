import argparse, os, csv, torch
from PIL import Image
from torchvision import transforms, models
def load(ckpt):
    ckpt=torch.load(ckpt, map_location='cpu')
    model=models.resnet18(); model.fc=torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt['classes']
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--folder', required=True); ap.add_argument('--out', default='predictions.csv'); args=ap.parse_args()
    model, classes=load(args.ckpt); tf=transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    rows=[['path','top1_class','top1_prob','top3']]
    with torch.no_grad():
        for fn in sorted(os.listdir(args.folder)):
            if not fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')): continue
            pth=os.path.join(args.folder, fn); x=tf(Image.open(pth).convert('RGB')).unsqueeze(0)
            probs=torch.softmax(model(x), dim=1)[0]; p1,i1=torch.max(probs,0); topk=torch.topk(probs, k=min(3,len(classes)))
            rows.append([pth, classes[int(i1)], f"{float(p1):.4f}", '; '.join([f"{classes[idx]}:{float(p):.4f}" for p,idx in zip(topk.values, topk.indices)])])
    with open(args.out,'w',newline='',encoding='utf-8') as f: csv.writer(f).writerows(rows); print('Saved:', args.out)
