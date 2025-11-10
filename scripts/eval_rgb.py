import argparse, os, torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_model(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location='cpu')
    model=models.resnet18()
    model.fc=torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt['classes']

def save_confusion_matrix(cm, classes, out_png):
    plt.figure(figsize=(6,6)); plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar()
    tick_marks=range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right'); plt.yticks(tick_marks, classes)
    plt.tight_layout(); plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.savefig(out_png, bbox_inches='tight', dpi=150); plt.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/rgb_resnet18_gpu/best_resnet18.pt')
    ap.add_argument('--test_csv', default='manifests/apple_test.csv')
    ap.add_argument('--out', default='runs/rgb_resnet18_gpu/eval_test')
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, classes=load_model(args.ckpt)
    tf=transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    df=pd.read_csv(args.test_csv)
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for _,r in df.iterrows():
            img=Image.open(r['path']).convert('RGB')
            x=tf(img).unsqueeze(0)
            p=torch.softmax(model(x), dim=1)[0]
            y_true.append(classes.index(r['label'])); y_pred.append(int(torch.argmax(p)))
    report=classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(os.path.join(args.out,'classification_report.txt'),'w',encoding='utf-8') as f: f.write(report)
    import pandas as pd
    pd.DataFrame(classification_report(y_true,y_pred,target_names=classes,output_dict=True,digits=4)).to_csv(os.path.join(args.out,'classification_report.csv'))
    cm=confusion_matrix(y_true,y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(args.out,'confusion_matrix.csv'))
    save_confusion_matrix(cm, classes, os.path.join(args.out,'confusion_matrix.png'))
    print(report); print('Saved TXT/CSV/PNG to', args.out)
