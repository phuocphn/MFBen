import h5py
import pdb
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pointnet
import numpy as np
import utils
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

config = dict(
    learing_rate=5e-4,
    batch_size=256,
    epochs=5000,
    m=3,
    scheduler=None,
    validation_step=1,
    num_cells=int(4000),
    dataset_path="/mnt/home/pham/data-gen/MLCAD/ds-04/2dobs+fixFluidType+fixShape.hdf5",
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_dataset(path=None, split=[0.8, 0.2]):
    hdf5_file = h5py.File(path, "r")
    data = hdf5_file["interior_data"][:, ...]
    
    ni = {'u_max': 0.0528556, 'u_min': -0.0104957, 'v_max': 0.0263741, 'v_min': -0.026564, 'p_max': 0.0014619, 'p_min': -0.000719237}
    u_min, u_max = ni['u_min'], ni['u_max']
    v_min, v_max = ni['v_min'], ni['v_max']
    p_min, p_max = ni['p_min'], ni['p_max']


    assert data.shape[-1] == 5
    data[:,:,2] = (data[:,:,2] - u_min)/(u_max - u_min)
    data[:,:,3] = (data[:,:,3] - v_min)/(v_max - v_min)
    data[:,:,4] = (data[:,:,4] - p_min)/(p_max - p_min)

    train_len = int(split[0] * data.shape[0])
    test_len = int(split[0] * data.shape[0])
    
    train_set = data[:train_len, :, :]
    test_set = data[train_len:(train_len + test_len), :, :]
    train_set = torch.permute(torch.tensor(train_set),(0, 2, 1))
    test_set = torch.permute(torch.tensor(test_set),(0, 2, 1))

    return train_set, test_set, ni

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make(config):

    model= pointnet.PointNetSegHead(num_points=config.num_cells, m=config.m)
    model = model.to(device)
    print (f"num of training parameters: {count_parameters(model)}")
    print (f"use {device} for training")


    train_set, test_set, normalization_info = read_dataset(config.dataset_path)
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set[:1,:, :], batch_size=1, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learing_rate)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, 
    #                                          step_size_up=1000, cycle_momentum=False)

    return model, train_loader, val_loader, test_loader, criterion, optimizer, normalization_info


def train_batch(X, targets, model, optimizer, criterion, ni):
    X, targets = X.to(device), targets.to(device)
    
    model.train()
    pred,_,_ = model(X)
    loss = criterion(pred, targets)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    return loss


def train(model, train_loader, val_loader, test_loader, criterion, optimizer, config, ni):

    wandb.watch(model, criterion, log="all", log_freq=10)
    total_batches = len(train_loader) * config.epochs
    
    example_ct = 0 # number of examples seen
    batch_ct = 0

    for epoch in tqdm(range(config.epochs)):
        print ("Epoch: ", epoch)
        for train_data in train_loader:
            x_train = train_data[:, 0:2, :config.num_cells]

            targets = train_data[:, 2:, :config.num_cells]
            targets = torch.permute(targets, (0, 2, 1))

            loss = train_batch(x_train, targets, model, optimizer, criterion, ni)
            example_ct +=len(x_train)
            batch_ct += 1

            if ((batch_ct + 1) % 1) == 0:
                train_log(loss, example_ct, epoch)

        #validate
        if epoch % config.validation_step == 0:
            validate(model, train_loader, criterion, epoch, config, ni)
            #predict(model, test_loader, criterion, epoch, ni)


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "train_loss": loss}, step=example_ct)
    print(f"train_loss after {str(example_ct).zfill(6)} examples: {loss:.4f}")

def validate(model, val_loader,criterion, epoch, config, ni):
    model.eval()
    losses = utils.AverageMeter('Validation Loss:', ':.4e')
    
    for i, val_data in enumerate(val_loader):
        x_val = val_data[:, 0:2, :config.num_cells]
        targets = val_data[:, 2:, :config.num_cells]
        targets = torch.permute(targets, (0, 2, 1))
        x_val, targets = x_val.to(device), targets.to(device)

        with torch.no_grad():
            pred, _, _ = model(x_val)
            loss = criterion(pred, targets)
            losses.update(loss.item(), x_val.size(0)) 
    
    wandb.log({"val_loss": losses.avg, "epoch": epoch})
    print(f"val_loss: {losses.avg:.4f}, epoch: {epoch}")

    iid = 0
    save_path = "plots/" + str(epoch) + ".png"
    dump_prediction(x_coord=x_val.transpose(2,1)[iid,:,-2].cpu(), 
            y_coord=x_val.transpose(2,1)[iid,:,-1].cpu(), 
            y_true=targets[iid, :, :].cpu(), 
            y_pred=pred[iid,:,:].cpu(), 
            loss=loss,
            config=config,
            save_path=save_path)


def dump_prediction(x_coord, y_coord, y_true, y_pred, loss, config, save_path):
    extent = -0.25, 0.65, -0.1, 0.1
    plt.suptitle('Comparision of OpenFOAM vs Deep Learning\nMean Squared Error: {0:0.5f}'.format(loss.item()), fontsize=13)
    plt.subplot(211)
    


    ux_true = y_true[:config.num_cells,0]
    uy_true = y_true[:config.num_cells,1]
    p_true = y_true[:config.num_cells,2]

    ux_pred = y_pred[:config.num_cells,0]
    uy_pred = y_pred[:config.num_cells,1]
    p_pred = y_pred[:config.num_cells,2]

    x_coord = x_coord[:config.num_cells]
    y_coord = y_coord[:config.num_cells]

    plt.ylabel('OpenFOAM', fontsize=15)
    plt.scatter(x_coord, y_coord, c=p_true, cmap='viridis')

    plt.subplot(212)
    plt.ylabel('PointNet CFD', fontsize=15)
    plt.scatter(x_coord, y_coord, c=p_pred, cmap='viridis')

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.85)
    plt.savefig(save_path)
    plt.close()

def model_pipeline(hyperparameters):
    with wandb.init(project="dd-cfd", config=hyperparameters):
        config = wandb.config

        #make the model, data, and optimization problem
        model, train_loader, val_loader, test_loader, criterion, optimizer, normalization_info = make(config)
        print (model)

        train(model, train_loader, val_loader, test_loader, criterion, optimizer, config, normalization_info)

    return model

model_pipeline(config)
