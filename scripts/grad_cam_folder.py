import argparse, os, glob, math
import torch, numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from torchvision import models, transforms

def load_model(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location='cpu')
    # TrangTK
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

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
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8); hf.remove(); hb.remove(); return cam, pred_idx, torch.softmax(logits,dim=1)[0, pred_idx].item()

def save_heatmap(cam, pil_img, out_path, overlay_path=None, alpha=0.45, label=None, prob=None):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR)
    cam_np = np.array(cam_img)

    # Heatmap riêng
    plt.figure(figsize=(4,4))
    plt.imshow(cam_np, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Overlay có chú thích
    if overlay_path:
        overlay = pil_img.copy()
        plt.figure(figsize=(4,4))
        plt.imshow(overlay)
        plt.imshow(cam_np, cmap='jet', alpha=alpha)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Vẽ text dự đoán lên overlay
        img_overlay = Image.open(overlay_path).convert("RGB")
        draw = ImageDraw.Draw(img_overlay)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        if label and prob is not None:
            text = f"{label} ({prob*100:.1f}%)"
            bbox = draw.textbbox((10, 10), text, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-3, bbox[2]+5, bbox[3]+3], fill=(0, 0, 0, 160))
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)

        img_overlay.save(overlay_path)

# def save_heatmap(cam, pil_img, out_path, overlay_path=None, alpha=0.45):
#     cam_img = Image.fromarray((cam*255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR); cam_np = np.array(cam_img)
#     plt.figure(figsize=(4,4)); plt.imshow(cam_np, cmap='jet'); plt.axis('off'); plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0); plt.close()
#     if overlay_path:
#         plt.figure(figsize=(4,4)); plt.imshow(pil_img); plt.imshow(cam_np, cmap='jet', alpha=alpha); plt.axis('off'); plt.tight_layout(); plt.savefig(overlay_path, dpi=150, bbox_inches='tight', pad_inches=0); plt.close()

def make_grid(images, grid_cols, pad=6, bg=(255,255,255)):
    if not images: return None
    w,h = images[0].size
    rows = math.ceil(len(images)/grid_cols)
    grid = Image.new("RGB", (grid_cols*w + pad*(grid_cols+1), rows*h + pad*(rows+1)), bg)
    k=0
    for r in range(rows):
        for c in range(grid_cols):
            if k>=len(images): break
            x = pad + c*(w+pad); y = pad + r*(h+pad)
            grid.paste(images[k].resize((w,h)), (x,y)); k+=1
    return grid

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    model, classes=load_model(args.ckpt)
    # Limit to 4 Apple classes if present
    target_classes = [c for c in classes if c.startswith("Apple___")]
    if target_classes and len(target_classes)==4:
        classes = target_classes

    per_class=args.per_class
    all_class_grids_overlay=[]
    all_class_grids_heat=[]
    for cls in classes:
        cls_dir=os.path.join(args.data_root, cls)
        if not os.path.isdir(cls_dir): 
            print(f"[WARN] Missing class folder: {cls_dir}")
            continue
        out_cls_dir = os.path.join(args.out_dir, cls.replace(':','_'))
        os.makedirs(out_cls_dir, exist_ok=True)
        images = sorted([p for p in glob.glob(os.path.join(cls_dir, "*")) if p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))])
        images = images[:per_class]
        overlays=[]; heats=[]
        for p in images:
            x,pil=preprocess(p, size=args.size)
            # cam, pred_idx, prob = grad_cam(model, x, target_layer_name=args.layer)
            # base=os.path.splitext(os.path.basename(p))[0]
            # heat=os.path.join(out_cls_dir, f"{base}_heat.png")
            # ovl =os.path.join(out_cls_dir, f"{base}_overlay.png")
            # save_heatmap(cam, pil, heat, ovl, alpha=0.45)
            cam, pred_idx, prob = grad_cam(model, x, target_layer_name=args.layer)
            pred_label = classes[pred_idx] if pred_idx < len(classes) else f"cls_{pred_idx}"
            base = os.path.splitext(os.path.basename(p))[0]
            heat = os.path.join(out_cls_dir, f"{base}_heat.png")
            ovl  = os.path.join(out_cls_dir, f"{base}_overlay.png")
            save_heatmap(cam, pil, heat, ovl, alpha=0.45, label=pred_label, prob=prob)
            overlays.append(Image.open(ovl).convert("RGB"))
            heats.append(Image.open(heat).convert("RGB"))
        # per-class grids
        if overlays:
            grid_ov = make_grid(overlays, grid_cols=min(3, len(overlays)))
            grid_ht = make_grid(heats,    grid_cols=min(3, len(heats)))
            if grid_ov: grid_ov.save(os.path.join(args.out_dir, f"overlay_grid_{cls}.png"))
            if grid_ht: grid_ht.save(os.path.join(args.out_dir, f"heatmap_grid_{cls}.png"))
            # downsized for poster
            all_class_grids_overlay.append(grid_ov.resize((600, int(600*grid_ov.height/grid_ov.width))) if grid_ov else None)
            all_class_grids_heat.append(grid_ht.resize((600, int(600*grid_ht.height/grid_ht.width))) if grid_ht else None)

    
    from datetime import datetime

# Summary poster (4 classes x 2 columns: overlay | heatmap)
# Summary poster (N classes x 2 columns: Overlay | Heatmap)
    if len(all_class_grids_overlay) == len(classes) and len(classes) > 0:
        box_w = 600
        pad = 20
        w = 2*box_w + 3*pad
        h = sum(img.height for img in all_class_grids_overlay) + (len(classes)+3)*pad + 60
        poster = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(poster)

        # Header info
        try:
            header_font = ImageFont.truetype("arial.ttf", 26)
            title_font = ImageFont.truetype("arial.ttf", 22)
        except:
            header_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        header = "Grad-CAM Visualization Summary"
        sub = f"Model: {os.path.basename(args.ckpt)}    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        hw = header_font.getbbox(header)[2]
        draw.text(((w - hw)//2, pad), header, fill=(0,0,0), font=header_font)
        draw.text((pad, pad+40), sub, fill=(50,50,50), font=title_font)
        y = pad + 80

        # Column titles
        draw.text((pad + 250, y), "Overlay", fill=(0,0,0), font=title_font)
        draw.text((pad + 600 + pad + 250, y), "Heatmap", fill=(0,0,0), font=title_font)
        y += 40

        # Draw class sections
        for i, cls in enumerate(classes):
            ov = all_class_grids_overlay[i]
            ht = all_class_grids_heat[i]
            th = title_font.getbbox(cls)[3] - title_font.getbbox(cls)[1]

            # Class title (centered)
            draw.text((pad, y), cls, fill=(0,0,120), font=title_font)
            y += th + 8

            # Paste images
            poster.paste(ov, (pad, y))
            poster.paste(ht, (pad + box_w + pad, y))

            # Frame rectangles
            draw.rectangle([pad-3, y-3, pad+box_w+3, y+ov.height+3], outline=(180,180,180), width=2)
            draw.rectangle([pad+box_w+pad-3, y-3, pad+2*box_w+pad+3, y+ht.height+3], outline=(180,180,180), width=2)

            y += max(ov.height, ht.height) + pad

        # Save
        poster_path = os.path.join(args.out_dir, "cam_summary_poster.png")
        poster.save(poster_path)
        print("✅ Saved upgraded poster:", poster_path)


    
    # Summary poster (4 classes x 2 columns: overlay | heatmap)
    # if len(all_class_grids_overlay)==len(classes)==4:
    #     w = 2*600 + 3*10
    #     h = sum(img.height for img in all_class_grids_overlay) + (len(classes)+1)*10
    #     poster = Image.new("RGB", (w, h), (255,255,255))
    #     y=10
    #     font=None
    #     try:
    #         font = ImageFont.truetype("arial.ttf", 22)
    #     except:
    #         pass
    #     for i,cls in enumerate(classes):
    #         ov=all_class_grids_overlay[i]; ht=all_class_grids_heat[i]
    #         draw=ImageDraw.Draw(poster)
    #         title = f"{cls}"

    #         if font:
    #             draw.text((10, y), title, fill=(0,0,0), font=font)
    #             # getbbox trả về (x0, y0, x1, y1) → chiều cao = y1 - y0
    #             bbox = font.getbbox(title)
    #             th = bbox[3] - bbox[1]
    #         else:
    #             draw.text((10, y), title, fill=(0,0,0))
    #             th = 22



    #         # if font:
    #         #     draw.text((10, y), title, fill=(0,0,0), font=font)
    #         #     th = font.getsize(title)[1]

    #         # else:
    #         #     draw.text((10, y), title, fill=(0,0,0))
    #         #     th = 22

    #         y += th + 6
    #         poster.paste(ov, (10, y))
    #         poster.paste(ht, (10+600+10, y))
    #         y += max(ov.height, ht.height) + 10
    #     poster_path=os.path.join(args.out_dir, "cam_summary_poster.png")
    #     poster.save(poster_path)
    #     print("Saved poster:", poster_path)

    print("Done. Outputs in", args.out_dir)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', required=True, help='Root with class subfolders')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--per_class', type=int, default=6)
    ap.add_argument('--layer', default='layer4')
    args=ap.parse_args()
    main(args)
