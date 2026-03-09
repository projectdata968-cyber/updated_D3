
import os
import torch
import argparse
import shutil
import random
import numpy as np
from models import D3_model
from data.datasets import read_video, set_preprocessing
import albumentations as A

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # A.set_random_seed(seed) # This line was problematic and is now removed.

def main():
    seed_everything(42) # Call the seeding function here

    parser = argparse.ArgumentParser(description='D3 Forensic Video Detector')
    parser.add_argument('--video', type=str, required=True, help='Path to test video')
    # Using your empirically discovered threshold for L2 + XCLIP-16
    parser.add_argument('--threshold', type=float, default=2.7)
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--loss', type=str, default='l2')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # 1. Setup temporary workspace for the video2frame utility
    temp_dir = f"forensic_temp_{video_name}"
    video_in = os.path.join(temp_dir, "video", "input")
    os.makedirs(video_in, exist_ok=True)
    shutil.copy(args.video, video_in)

    # 2. Extract Frames (preserving the 8/16 frame logic from datasets.py)
    print(f"[*] Extracting temporal features from {os.path.basename(args.video)}...")
    os.system(f"python utils/video2frame.py --dataset-path {temp_dir} > /dev/null 2>&1")

    # 3. Preprocessing (Aligns with your PhD-level research standards)
    # This automatically includes your 10% center crop and XCLIP normalization
    trans = set_preprocessing(None, None)
    frames_path = os.path.join(temp_dir, "frames", "input", video_name)

    try:
        video_tensor = read_video(frames_path, trans)
        # Reshape to [1, T, C, H, W] for the model
        video_tensor = video_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # 4. Initialize Model
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    model.eval()

    # 5. Predict
    with torch.no_grad():
        # returns: (features, acceleration_avg, volatility_std)
        _, _, score_tensor = model(video_tensor)
        score = score_tensor.cpu().item()

    # 6. Classification Logic (Real = Positive Class)
    # High Score (High Volatility) = Authentic Physics
    # Low Score (Smoothness) = AI Diffusion
    is_real = score >= args.threshold
    verdict = "REAL" if is_real else "AI-GENERATED"

    print("" + "-"*45)
    print(" D3 FORENSIC ANALYSIS RESULT")
    print("-"*45)
    print(" File:      " + os.path.basename(args.video))
    print(" Volatility: " + f"{score:.6f}") # Use f-string for float formatting
    print(" Threshold:  " + f"{args.threshold:.6f}") # Use f-string for float formatting
    print(" Verdict:    " + verdict)
    print("-"*45 + "")

    # 7. Cleanup
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
