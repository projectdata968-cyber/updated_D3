import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import datetime
# from sklearn.metrics import average_precission_score
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from data import D3_dataset_AP
from models import D3_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script with configurable parameters.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--gpu-id', type=str, default="0",
                        help='CUDA GPU device ID(s), e.g., "0" or "1,2,3" (default: "5")')
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'cos'],
                        help='Loss function type (default: l2)')
    parser.add_argument('--encoder', type=str, default='XCLIP-16', 
                        help='Encoder model name (default: XCLIP-16)',
                        choices=['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32', 'DINO-base', 'DINO-large', 'ResNet-18', 'VGG-16', 'EfficientNet-b4', 'MobileNet-v3'])
    parser.add_argument('--real-csv', type=str, default=None,
                        help='Path to the real data CSV file ')
    parser.add_argument('--fake-csv', type=str, default=None,
                        help='Path to the fake/synthetic data CSV file')
    args = parser.parse_args()

    seed = args.seed
    gpu_id = args.gpu_id
    loss_type = args.loss
    encoder_type = args.encoder
    real_csv = args.real_csv
    fake_csv = args.fake_csv

    # real_csv = 'datasets/csv/t1.csv'
    # fake_csv = 'datasets/csv/t2.csv' 
    
    print(f"Starting AP evaluation for {encoder_type} with {loss_type} loss")
    print(f"Real CSV: {real_csv}")
    print(f"Fake CSV: {fake_csv}")
    
    # Load Model
    model = D3_model(encoder_type=encoder_type, loss_type=loss_type).cuda()
    model.eval()
    
    # Load Dataset
    eval_dataset = D3_dataset_AP(real_csv=real_csv, fake_csv=fake_csv, max_len=1000)
    print(f"Total samples: {len(eval_dataset)}")
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True,
        drop_last=False
    )
    
    # Eval
    # Eval
    y_true, y_pred, video_names = [], [], []
    print("Individual Video Scores (Higher = More likely Real):")
    print("-" * 30)
    
    with torch.no_grad():
        # Add 'i' to track the index if your dataset doesn't return filenames
        for i, (batch_frames, batch_label, batch_video_name) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            batch_inputs = batch_frames.cuda()
            _, _, batch_dis_std = model(batch_inputs)
            
            score = batch_dis_std.cpu().item()
            label = "Fake" if batch_label.item() == 1 else "Real"
            
            # Print the individual result
            print(f"Sample {i+1}: Video: {batch_video_name[0]} | Score = {score:.4f} | Source: {label}")
            
            y_pred.append(score)
            y_true.append(batch_label.item())
            video_names.append(batch_video_name[0])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 2. Binarize predictions
    # Note: Your AP calculation uses (1 - y_true), implying:
    # Label 0 = Real, Label 1 = Fake. 
    # Your score logic: Higher = More likely Real.
    # Therefore, we use a threshold to decide if a video is "Real" (Class 1 for metrics)
    y_true_binary = 1 - y_true # Inverting so 'Real' is the positive class (as per the AP logic)
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {threshold}")
    y_pred_binary = (y_pred >= threshold).astype(int)

    # 3. Calculate Metrics
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary)
    rec = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    ap_score = average_precision_score(y_true_binary, y_pred)
    # ap_score = average_precision_score(1-y_true, y_pred)
    
    result_str = (
    f"--- Evaluation Results ---\n"
    f"Encoder: {encoder_type} | Loss: {loss_type}\n"
    f"Total Samples: {len(y_true)}\n"
    f"Threshold used: {threshold}\n"
    f"--------------------------\n"
    f"Accuracy:  {acc:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall:    {rec:.4f}\n"
    f"F1-Score:  {f1:.4f}\n"
    f"AP Score:  {ap_score:.4f}\n"
  )
    
    print("\n" + "="*50)
    print(result_str.strip())
    print("="*50)

    # --- Identify Errors ---
    print("\nMisclassified Videos:")
    for name, true, pred in zip(video_names, y_true_binary, y_pred_binary):
        if true != pred:
            status = "Fake labeled as Real (FP)" if pred == 1 else "Real labeled as Fake (FN)"
            print(f" - {name}: {status}")

    # --- Confusion Matrix Plotting ---
    # labels=[1, 0] keeps 'Real' as the first row/column
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Real', 'Pred Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title(f'Confusion Matrix\n{args.encoder} (Acc: {acc:.2f})')
    
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/cm_{timestamp}.png")
    print(f"\nConfusion matrix saved to results/cm_{timestamp}.png")

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_file = f"results/result_{timestamp}.txt"

    # with open(output_file, 'w') as f:
    #     f.write(result_str)

    # print(f"\nResults saved to {output_file}")
