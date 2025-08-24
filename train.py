
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import pickle

import importlib

from two_layer_apc import APC
from utils import load_mnist, load_fashion, load_omni, load_affnist


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="mnist or coil100",default='mnist', type=str)
parser.add_argument("--outdir", help="Directory for results and log files", default='.', type=str)
parser.add_argument("--load", help="load model",default=0, type=int)
parser.add_argument("--epochs", help="# of training epochs",default=100, type=int)
parser.add_argument("--eval", help="evaluate",default=0, type=int)
args = parser.parse_args()
DATASET = args.dataset
OUTDIR = args.outdir
LOAD = args.load
EPOCHS = args.epochs
EVAL = args.eval


if not os.path.exists('./models/{}/'.format(OUTDIR)):
	os.mkdir('./models/{}/'.format(OUTDIR))


if LOAD:
	fh = open('./models/{}/'.format(OUTDIR)+'config.pkl','rb')
	CFG = pickle.load(fh)
	fh.close()
else:
	CFG = importlib.import_module('.'+DATASET+'_config','configs').cfg


if DATASET=='mnist':
	tr_dl, vd_dl, ts_dl = load_mnist(CFG)
elif DATASET=='fashion':
	tr_dl, vd_dl, ts_dl = load_fashion(CFG)
elif DATASET=='omni':
	tr_dl, vd_dl, ts_dl = load_omni(CFG)
elif DATASET=='affnist':
	tr_dl, vd_dl, ts_dl = load_affnist(CFG)
	

fh = open('./models/{}/'.format(OUTDIR)+'config.pkl','wb')
pickle.dump(CFG,fh)
fh.close()

BATCH_SIZE = CFG.BATCH_SIZE
H,W,C = CFG.dims

# Create PyTorch model
model = APC(CFG)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Print model summary (PyTorch equivalent)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Summary:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

if LOAD:
	model.load_state_dict(torch.load('./models/{}/model/weights.pth'.format(OUTDIR), map_location=device))

# Training function
def train_epoch(model, dataloader, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # The APC model expects (images, labels) as input
        inputs = (data, target)
        
        # Use the model's train_step method
        result = model.train_step(inputs)
        total_loss += result['loss']
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # The APC model expects (images, labels) as input
            inputs = (data, target)
            
            # Use the model's test_step method
            result = model.test_step(inputs)
            # For evaluation, we might want to compute MSE loss
            pred = result['pred']
            mse_loss = torch.mean((pred - data) ** 2)
            total_loss += mse_loss.item()
    
    return total_loss / len(dataloader)

if EVAL:
	model.std1 = 0.0
	model.std2 = 0.0
	
	# Evaluate on test set
	test_loss = evaluate(model, ts_dl, device)
	print(f"Test Loss: {test_loss:.6f}")
	exit()
	
else:
	# Training loop
	for epoch in range(EPOCHS):
		train_loss = train_epoch(model, tr_dl, device)
		
		# Validation every 50 epochs
		if (epoch + 1) % 50 == 0:
			val_loss = evaluate(model, vd_dl, device)
			print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
		else:
			print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.6f}')
	
	# Save model weights
	if not os.path.exists('./models/{}/model/'.format(OUTDIR)):
		os.makedirs('./models/{}/model/'.format(OUTDIR))
	torch.save(model.state_dict(), './models/{}/model/weights.pth'.format(OUTDIR))
	
	# Final evaluation on test set
	test_loss = evaluate(model, ts_dl, device)
	print(f"Final Test Loss: {test_loss:.6f}")