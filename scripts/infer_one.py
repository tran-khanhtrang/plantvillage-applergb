import argparse, torch
from PIL import Image
from torchvision import transforms, models
def load(ckpt):
    ckpt=torch.load(ckpt, map_location='cpu')
    model=models.resnet18(); model.fc=torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt['classes']
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt', default='runs/rgb_resnet18_gpu/best_resnet18.pt'); ap.add_argument('--image', required=True); args=ap.parse_args()
    model, classes=load(args.ckpt); tf=transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    x=tf(Image.open(args.image).convert('RGB')).unsqueeze(0); 
    with torch.no_grad(): probs=torch.softmax(model(x), dim=1)[0]
    topk=torch.topk(probs, k=3); print("Top-3:")
    for p,idx in zip(topk.values, topk.indices): print(f"  {classes[idx]} : {float(p):.4f}")
