import argparse, torch
from torchvision import models
def load(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location='cpu')
    model=models.resnet18(); model.fc=torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt['classes']
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', default='model.onnx'); ap.add_argument('--opset', type=int, default=14); args=ap.parse_args()
    model, classes=load(args.ckpt); dummy=torch.randn(1,3,256,256)
    torch.onnx.export(model, dummy, args.out, opset_version=args.opset, input_names=['input'], output_names=['logits'], dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}})
    print('Exported ONNX:', args.out, '| classes:', classes)
