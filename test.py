import wave
import math
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm

from models.wavenet import WaveNet
from models.convnet import ConvNet
from utils import wav_to_tensor, np_to_wav

parser = ArgumentParser()
parser.add_argument("--model", default="wavenet", help="Model definition [wavenet, convnet]", choices=["wavenet", "convnet"])
parser.add_argument("--input_audio", default="audio/validation_DI.wav", help="Path to audio to reamp")
parser.add_argument("--model_weights", default="weights/wavenet.model", help="Path to .model file")
parser.add_argument("--output_audio_path", default="output.wav", help="Path for output audio")
parser.add_argument("--device", default=0, help="Pytorch device")

def get_model(model):
    if(model == "wavenet"):
        return WaveNet()
    elif(model == "convnet"):
        return ConvNet()
    else:
        return WaveNet()
    
def get_sample_rate(audio_path):
    with wave.open(audio_path, "rb") as wave_file:
        return wave_file.getframerate()

def main(args):
    sample_rate = get_sample_rate(args.input_audio)
    overlap = 1000
    num_input_samples = 12000
    model = get_model(args.model).cuda() 
    model.load_state_dict(torch.load(args.model_weights))
    source = wav_to_tensor(args.input_audio).cuda()
    num_batches = math.ceil((len(source) / (num_input_samples + overlap)))
    wav_output = np.ndarray((1,))
    outputs = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(range(num_batches)):
            start = batch * num_input_samples
            end = start + num_input_samples
            start = max(start - overlap, 0)
            end += overlap
            output = model(source[None, start:end])
            output = output.cpu().numpy()[overlap:-overlap]
            wav_output = np.concatenate([wav_output, output])
            outputs.append(output)

    np_to_wav(wav_output, args.output_audio_path, rate=sample_rate)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
