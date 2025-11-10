import os, argparse, json, random, numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CsvImageDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, train=True, img_size=256):
        df = pd.read_csv(csv_path)
        self.paths = df['path'].tolist()
        self.labels = [class_to_idx[l] for l in df['label'].tolist()]
        aug_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(),
        ])
        aug_val = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.tf = aug_train if train else aug_val
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = Image.open(self.paths[i]).convert('RGB')
        x = self.tf(x); y = self.labels[i]
        return x, y

@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    y_true=[]; y_pred=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        with (torch.amp.autocast('cuda') if (amp and device=='cuda') else torch.no_grad()):
            logits=model(x)
        pred=logits.argmax(1)
        y_true+=y.cpu().tolist(); y_pred+=pred.cpu().tolist()
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred, average='macro')
    return acc,f1

def main(args):
    set_seed(42)
    classes=[l.strip() for l in open(args.classes,'r',encoding='utf-8').read().splitlines() if l.strip()]
    class_to_idx={c:i for i,c in enumerate(classes)}
    os.makedirs(args.out, exist_ok=True)
    tb_dir=os.path.join(args.out,'tb'); os.makedirs(tb_dir, exist_ok=True)
    writer=SummaryWriter(tb_dir)
    with open(os.path.join(args.out,'class_to_idx.json'),'w',encoding='utf-8') as f:
        json.dump(class_to_idx,f,ensure_ascii=False,indent=2)

    tr_ds=CsvImageDataset(args.train_csv, class_to_idx, train=True, img_size=args.size)
    va_ds=CsvImageDataset(args.val_csv,   class_to_idx, train=False, img_size=args.size)
    tr_ld=DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    va_ld=DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc=nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler=ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    criterion=nn.CrossEntropyLoss()

    use_amp = (device=='cuda') and args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    global_step=0; best_f1=0.0; best_path=os.path.join(args.out,'best_resnet18.pt'); epochs_no_improve=0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar=tqdm(tr_ld, desc=f"Epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits=model(x); loss=criterion(logits, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits=model(x); loss=criterion(logits, y)
                loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.item()), lr=opt.param_groups[0]['lr'])
            writer.add_scalar('train/loss', float(loss.item()), global_step); global_step+=1

        acc,f1=evaluate(model, va_ld, device, amp=use_amp)
        writer.add_scalar('val/accuracy', acc, epoch); writer.add_scalar('val/macro_f1', f1, epoch)
        scheduler.step(f1); print(f"[VAL] acc={acc:.4f}  f1={f1:.4f}")
        if f1>best_f1 + 1e-6:
            best_f1=f1; epochs_no_improve=0
            torch.save({'model':model.state_dict(),'classes':classes}, best_path); print("  ✓ Saved:", best_path)
        else:
            epochs_no_improve+=1
        if epochs_no_improve>=args.early_patience:
            print(f"Early stopping at epoch {epoch}"); break
    writer.close(); print("Done. Best F1:", best_f1)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_csv', default='manifests/apple_train.csv')
    ap.add_argument('--val_csv',   default='manifests/apple_val.csv')
    ap.add_argument('--classes',   default='manifests/classes.txt')
    ap.add_argument('--out',       default='runs/rgb_resnet18_gpu')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--bs',   type=int, default=32)
    ap.add_argument('--lr',   type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--early_patience', type=int, default=6)
    ap.add_argument('--amp', action='store_true', help='Enable mixed precision on GPU')
    args=ap.parse_args(); main(args)
