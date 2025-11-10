import argparse, torch
from torchvision import models
def load(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location='cpu')
    model=models.resnet18(); model.fc=torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt['classes']
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', default='model.ts'); args=ap.parse_args()
    model, classes=load(args.ckpt); scripted=torch.jit.script(model); scripted.save(args.out); print('Exported TorchScript:', args.out, '| classes:', classes)
