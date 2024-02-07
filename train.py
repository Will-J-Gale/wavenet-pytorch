import math
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

from models.wavenet import WaveNet
from models.convnet import ConvNet
from utils import wav_to_tensor, DataGenerator

parser = ArgumentParser()
parser.add_argument("--model", default="wavenet", help="Model to train [wavenet, convnet]", choices=["wavenet", "convnet"])
parser.add_argument("--input_audio", default="audio/train/high_gain/high_gain_DI.wav", help="Path to input audio")
parser.add_argument("--output_audio", default="audio/train/high_gain/high_gain_AMP.wav", help="Path to reamped audio")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
parser.add_argument("--device", default=0, help="Pytorch device")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
parser.add_argument("--learning_rate", default=0.004, help="Optimizer learing rate")
parser.add_argument("--lr_decay", default=0.993, help="Learning rate decay")
parser.add_argument("--validation_slice_index", default=5362960, type=int, help="index to slice validation dataset")
parser.add_argument("--model_output_size", default=8192, help="Number of samples per training pair")
parser.add_argument("--model_name", default="wavenet.model", help="Name of saved model")

def get_model(model):
    if(model == "wavenet"):
        return WaveNet()
    elif(model == "convnet"):
        return ConvNet()
    else:
        return WaveNet()

def main(args):
    model = get_model(args.model).to(args.device) 
    input_size = model.receptive_field + args.model_output_size
    optimizer = Adam(model.parameters(), args.learning_rate)
    lr_decay = ExponentialLR(optimizer, args.lr_decay)
    mse_loss = MSELoss()
    
    input_audio = wav_to_tensor(args.input_audio).to(args.device)
    output_audio = wav_to_tensor(args.output_audio).to(args.device)
    assert(len(input_audio) == len(output_audio))

    train_x = input_audio[:args.validation_slice_index]
    train_y = output_audio[:args.validation_slice_index]
    test_x = input_audio[args.validation_slice_index:]
    test_y = output_audio[args.validation_slice_index:]

    train_data_generator = DataGenerator(
        train_x, 
        train_y, 
        input_size=input_size, 
        output_size=args.model_output_size, 
        batch_size=args.batch_size, 
        device=args.device
    )

    min_loss = math.inf

    progress = tqdm(total=args.epochs)
    for _ in range(args.epochs):
        train_data_generator.shuffle()

        epoch_losses = []
        for x_batch, y_batch in train_data_generator:
            optimizer.zero_grad()
            prediction = model(x_batch, pad_start=False)
            loss = mse_loss(prediction, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        
        lr_decay.step()
        avg_epoch_loss = np.mean(epoch_losses)
        progress.set_description(f"Avg epoch loss: {avg_epoch_loss:.6f}")

        if(avg_epoch_loss < min_loss):
            min_loss = avg_epoch_loss
            torch.save(model.state_dict(), args.model_name)
        
        progress.update()

    plot_start = 100000
    plot_end = plot_start + 10000
    test_x_slice = test_x[plot_start:plot_end]
    test_y_slice = test_y[plot_start:plot_end].cpu().numpy()
    test_x_slice = test_x_slice[None, :]

    model.eval()
    prediction = model(test_x_slice.to(args.device)).cpu().detach().numpy().squeeze()

    print(prediction)
    plt.figure("Validation")
    plt.plot(test_y_slice)
    plt.plot(prediction)
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)