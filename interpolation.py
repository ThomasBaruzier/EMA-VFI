import os
import cv2
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

def interpolate_frames(I0, I2, model, padder, factor):
    TTA = model.TTA
    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I0_, I2_ = padder.pad(I0_, I2_)
    num_frames = factor + 1
    time_list = [(i + 1) * (1. / num_frames) for i in range(num_frames - 1)]
    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=time_list, fast_TTA=TTA)
    return [(padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1] for pred in preds]

def process_video(input, output_video, model, factor=1, fps=None, threads=1):
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        print(f"Error: Could not open the input video ({input}).")
        return

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps, num_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    padder = InputPadder((1, 3, height, width), divisor=32)
    output_fps = fps or input_fps * (factor + 1)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
    frames = [frame for ret, frame in iter(lambda: cap.read(), (False, None))]

    def process_frame(i):
        interpolated_frames = interpolate_frames(frames[i], frames[i + 1], model, padder, factor)
        output_frames = [frames[i]] + [cv2.cvtColor(interpolated_frame, cv2.COLOR_RGB2BGR) for interpolated_frame in interpolated_frames]
        return output_frames

    with ThreadPoolExecutor(max_workers=threads) as executor:
        processed_frames = list(tqdm(executor.map(process_frame, range(len(frames) - 1)), desc="Processing frames", total=len(frames) - 1, unit="frame"))

    for frame_set in processed_frames:
        for frame in frame_set:
            out.write(frame)

    out.write(frames[-1])
    cap.release()
    out.release()
    print(f"Video processing completed. Output video saved as {output_video}.")

def main(args):
    assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'
    TTA = args.model == 'ours_t'
    cfg.MODEL_CONFIG['LOGNAME'] = args.model
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32 if TTA else 16,
        depth=[2, 2, 2, 4, 4] if TTA else [2, 2, 2, 2, 2]
    )

    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()
    model.TTA = TTA

    process_video(args.input, args.output, model, factor=args.factor, fps=args.fps, threads=args.threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to the input video file")
    parser.add_argument('--output', required=True, help="Path to the output video file")
    parser.add_argument('--factor', type=int, default=1, help="Number of frames that will be guessed for each pair of frames")
    parser.add_argument('--fps', type=int, help="Number of frames per second of the interpolated video")
    parser.add_argument('--model', default='ours_t', type=str)
    parser.add_argument('--threads', default=1, type=int, help="Number of threads to use for processing frames")
    args = parser.parse_args()
    main(args)
