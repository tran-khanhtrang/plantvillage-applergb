import argparse, torch, numpy as np
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

def load_model(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location='cpu')
    model=models.resnet18(); model.fc=torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt['classes']

def preprocess(img_path, size=256):
    tf=transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    pil=Image.open(img_path).convert('RGB'); x=tf(pil).unsqueeze(0)
    return x, pil

def grad_cam(model, x, target_layer_name='layer4'):
    activations = {}; gradients = {}
    def fwd_hook(module, inp, out): activations['value'] = out.detach()
    def bwd_hook(module, grad_in, grad_out): gradients['value'] = grad_out[0].detach()
    target_layer = dict(model.named_modules())[target_layer_name]
    hf = target_layer.register_forward_hook(fwd_hook); hb = target_layer.register_full_backward_hook(bwd_hook)
    logits = model(x); pred_idx = int(torch.argmax(logits, dim=1)); score = logits[0, pred_idx]; model.zero_grad(); score.backward()
    acts = activations['value']; grads = gradients['value']; weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True); cam = torch.relu(cam)[0,0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8); hf.remove(); hb.remove(); return cam, pred_idx

def save_heatmap(cam, pil_img, out_path, overlay_path=None, alpha=0.45):
    cam_img = Image.fromarray((cam*255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR); cam_np = np.array(cam_img)
    plt.figure(figsize=(6,6)); plt.imshow(cam_np, cmap='jet'); plt.axis('off'); plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0); plt.close()
    if overlay_path:
        plt.figure(figsize=(6,6)); plt.imshow(pil_img); plt.imshow(cam_np, cmap='jet', alpha=alpha); plt.axis('off'); plt.tight_layout(); plt.savefig(overlay_path, dpi=150, bbox_inches='tight', pad_inches=0); plt.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--image', required=True)
    ap.add_argument('--out', default='heatmap.png'); ap.add_argument('--overlay', default=None); args=ap.parse_args()
    model, classes=load_model(args.ckpt); x, pil = preprocess(args.image); cam, pred_idx = grad_cam(model, x, target_layer_name='layer4')
    save_heatmap(cam, pil, args.out, args.overlay); print('Predicted:', classes[pred_idx]); print('Saved heatmap to', args.out); 
    if args.overlay: print('Saved overlay to', args.overlay)
