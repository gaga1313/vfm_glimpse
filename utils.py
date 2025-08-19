
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def hyperfanin_for_kernel(fanin, varin=1.0, relu=True, bias=True):
    """
    PyTorch equivalent of TensorFlow hyperfanin_for_kernel initializer.
    Returns a function that can be used with nn.init.xavier_uniform_.
    """
    def init_fn(tensor):
        hfanin, _ = tensor.shape
        variance = (1/varin) * (1/fanin) * (1/hfanin)
        
        if relu:
            variance *= 2.0
        if bias:
            variance /= 2.0
        
        variance = np.sqrt(3 * variance)
        
        # Use uniform initialization
        torch.nn.init.uniform_(tensor, -variance, variance)
        # Alternative: normal initialization
        # torch.nn.init.normal_(tensor, 0, variance)
    
    return init_fn


def hyperfanin_for_bias(varin=1.0, relu=True):
    """
    PyTorch equivalent of TensorFlow hyperfanin_for_bias initializer.
    Returns a function that can be used with nn.init.xavier_uniform_.
    """
    def init_fn(tensor):
        hfanin, _ = tensor.shape
        variance = (1/2) * (1/varin) * (1/hfanin)
        
        if relu:
            variance *= 2.0
        
        variance = np.sqrt(3 * variance)
        
        # Use uniform initialization
        torch.nn.init.uniform_(tensor, -variance, variance)
        # Alternative: normal initialization
        # torch.nn.init.normal_(tensor, 0, variance)
    
    return init_fn


def load_affnist(CFG):
    dat_val_train = loadmat('./Data/affnist/training_and_validation_batches/1.mat')['affNISTdata'][0][0]
    dat_test = loadmat('./Data/affnist/test_batches/1.mat')['affNISTdata'][0][0]

    x_val_train = np.transpose(dat_val_train[2])
    y_val_train = np.transpose(dat_val_train[5]).reshape([-1,])
    
    x_test = np.transpose(dat_test[2])
    y_test = np.transpose(dat_test[5]).reshape([-1,])
    
    x_val_train, x_test = x_val_train / 255.0, x_test / 255.0

    x_val_train = x_val_train.reshape([-1, 40, 40, 1]).astype(np.float32)
    x_test = x_test.reshape([-1, 40, 40, 1]).astype(np.float32)
    
    y_val_train = y_val_train.astype(np.int64)  # PyTorch uses int64 for labels
    y_test = y_test.astype(np.int64)
    
    H = W = 40
    
    x_train = x_val_train[:50000]
    y_train = y_val_train[:50000]
    
    x_valid = x_val_train[50000:]
    y_valid = y_val_train[50000:]
    
    assert x_train.shape[0] == 50000
    assert y_train.shape[0] == 50000
    assert x_valid.shape[0] == 10000
    assert y_valid.shape[0] == 10000
    assert x_test.shape[0] == 10000
    assert y_test.shape[0] == 10000
    
    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_train).permute(0, 3, 1, 2)  # NHWC to NCHW
    y_train = torch.from_numpy(y_train)
    x_valid = torch.from_numpy(x_valid).permute(0, 3, 1, 2)
    y_valid = torch.from_numpy(y_valid)
    x_test = torch.from_numpy(x_test).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test)
    
    # Create PyTorch datasets and dataloaders
    tr_ds = TensorDataset(x_train, y_train)
    vd_ds = TensorDataset(x_valid, y_valid)
    ts_ds = TensorDataset(x_test, y_test)
    
    tr_dl = DataLoader(tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    vd_dl = DataLoader(vd_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    ts_dl = DataLoader(ts_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    
    return tr_dl, vd_dl, ts_dl


def load_omni(CFG):
    x_train = np.load('./Data/omni/omni_train.npy')
    x_valid = np.load('./Data/omni/omni_valid.npy')
    x_test = np.load('./Data/omni/omni_test.npy')
    x_transfer = np.load('./Data/omni/omni_transfer.npy')
    
    x_orig = np.concatenate([x_train, x_valid, x_test], axis=0)
    
    x_train = x_orig[:, :17]
    x_test = x_orig[:, 17:]
    
    H, W = x_train.shape[-2:]
    C = 1

    x_train = x_train.reshape([-1, H, W, C]).astype(np.float32)
    x_test = x_test.reshape([-1, H, W, C]).astype(np.float32)
    x_transfer = x_transfer.reshape([-1, H, W, C]).astype(np.float32)
    
    y_train = np.ones((x_train.shape[0],))
    y_test = np.ones((x_test.shape[0],))
    y_transfer = np.ones((x_transfer.shape[0],))
    
    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_train).permute(0, 3, 1, 2)  # NHWC to NCHW
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test).long()
    x_transfer = torch.from_numpy(x_transfer).permute(0, 3, 1, 2)
    y_transfer = torch.from_numpy(y_transfer).long()
    
    # Create PyTorch datasets and dataloaders
    tr_ds = TensorDataset(x_train, y_train)
    ts_ds = TensorDataset(x_test, y_test)
    tf_ds = TensorDataset(x_transfer, y_transfer)
    
    tr_dl = DataLoader(tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    ts_dl = DataLoader(ts_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    tf_dl = DataLoader(tf_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    
    return tr_dl, ts_dl, tf_dl


def load_fashion(CFG):
    # Load Fashion MNIST using torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full dataset
    full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Combine train and test for custom split
    x_train = full_dataset.data.float() / 255.0
    y_train = full_dataset.targets
    x_test = test_dataset.data.float() / 255.0
    y_test = test_dataset.targets
    
    # Add channel dimension
    x_train = x_train.unsqueeze(1)  # Add channel dimension
    x_test = x_test.unsqueeze(1)
    
    # Combine data
    data = torch.cat([x_train, x_test], dim=0)
    labs = torch.cat([y_train, y_test], dim=0)
    
    label_set = CFG.label_set
    
    if label_set is not None:
        set_idx = torch.where(torch.isin(labs, torch.tensor(label_set)))[0]
        data = data[set_idx]
        labs = labs[set_idx]
        
        for lab in range(len(label_set)):
            labs[torch.where(labs == label_set[lab])[0]] = lab
    
    N = data.shape[0]
    
    num_split = int(np.floor((1 - 0.9) * N))
    N_ts = num_split
    N_vd = num_split
    N_tr = N - 2 * num_split
    
    torch.manual_seed(1337)
    
    shuffle_idx = torch.randperm(N)
    
    x_test = data[shuffle_idx[:num_split]]
    x_valid = data[shuffle_idx[num_split:(2 * num_split)]]
    x_train = data[shuffle_idx[(2 * num_split):]]
    
    y_test = labs[shuffle_idx[:num_split]]
    y_valid = labs[shuffle_idx[num_split:(2 * num_split)]]
    y_train = labs[shuffle_idx[(2 * num_split):]]
    
    # Create PyTorch datasets and dataloaders
    tr_ds = TensorDataset(x_train, y_train)
    vd_ds = TensorDataset(x_valid, y_valid)
    ts_ds = TensorDataset(x_test, y_test)
    
    tr_dl = DataLoader(tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    vd_dl = DataLoader(vd_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    ts_dl = DataLoader(ts_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    
    return tr_dl, vd_dl, ts_dl


def load_mnist(CFG):
    # Load MNIST using torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full dataset
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Combine train and test for custom split
    x_train = full_dataset.data.float() / 255.0
    y_train = full_dataset.targets
    x_test = test_dataset.data.float() / 255.0
    y_test = test_dataset.targets
    
    # Add channel dimension
    x_train = x_train.unsqueeze(1)  # Add channel dimension
    x_test = x_test.unsqueeze(1)
    
    # Combine data
    data = torch.cat([x_train, x_test], dim=0)
    labs = torch.cat([y_train, y_test], dim=0)
    
    label_set = CFG.label_set
    
    if label_set is not None:
        set_idx = torch.where(torch.isin(labs, torch.tensor(label_set)))[0]
        data = data[set_idx]
        labs = labs[set_idx]
        
        for lab in range(len(label_set)):
            labs[torch.where(labs == label_set[lab])[0]] = lab
    
    N = data.shape[0]
    
    num_split = int(np.floor((1 - 0.9) * N))
    N_ts = num_split
    N_vd = num_split
    N_tr = N - 2 * num_split
    
    torch.manual_seed(1337)
    
    shuffle_idx = torch.randperm(N)
    
    x_test = data[shuffle_idx[:num_split]]
    x_valid = data[shuffle_idx[num_split:(2 * num_split)]]
    x_train = data[shuffle_idx[(2 * num_split):]]
    
    y_test = labs[shuffle_idx[:num_split]]
    y_valid = labs[shuffle_idx[num_split:(2 * num_split)]]
    y_train = labs[shuffle_idx[(2 * num_split):]]
    
    # Create PyTorch datasets and dataloaders
    tr_ds = TensorDataset(x_train, y_train)
    vd_ds = TensorDataset(x_valid, y_valid)
    ts_ds = TensorDataset(x_test, y_test)
    
    tr_dl = DataLoader(tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    vd_dl = DataLoader(vd_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    ts_dl = DataLoader(ts_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)
    
    return tr_dl, vd_dl, ts_dl


def convert_node(x):
    """Convert PyTorch tensors to numpy arrays"""
    for t in range(len(x)):
        if isinstance(x[t], torch.Tensor):
            x[t] = x[t].detach().cpu().numpy()
    return x


def generate_grid(scale, resolution, dims):
    x_C = np.linspace(-scale, scale, resolution)
    
    axes = []
    
    for dim in range(dims):
        axes += [x_C]
    
    packed_grid = np.meshgrid(*axes)
    
    np_grid = packed_grid[0].reshape([-1, 1])
    
    for dim in range(dims - 1):
        np_grid = np.concatenate([np_grid, packed_grid[dim + 1].reshape([-1, 1])], axis=1)
    
    return np.float32(np_grid)


def extract_box(tgrid):
    Ax = tgrid[0, 0, 0].reshape([-1, 1])
    Ay = tgrid[0, 0, 1].reshape([-1, 1])
    Bx = tgrid[0, -1, 0].reshape([-1, 1])
    By = tgrid[0, -1, 1].reshape([-1, 1])
    Cx = tgrid[-1, 0, 0].reshape([-1, 1])
    Cy = tgrid[-1, 0, 1].reshape([-1, 1])
    Dx = tgrid[-1, -1, 0].reshape([-1, 1])
    Dy = tgrid[-1, -1, 1].reshape([-1, 1])

    x_vals = np.concatenate([Ax, Bx, Cx, Dx], axis=-1)
    y_vals = np.concatenate([Ay, By, Cy, Dy], axis=-1)

    return x_vals, y_vals
