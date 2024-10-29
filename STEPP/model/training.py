import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from STEPP.model.mlp import ReconstructMLP
import numpy as np
from STEPP.utils.misc import make_results_folder
from STEPP.utils.testing import test_feature_reconstructor_with_model
import time
import wandb
import sys
import os


# Data loader
class FeatureDataset:
    def __init__(self, feature_dir, stack=False, transform=None, target_transform=None, batch_size=None) -> None:
        self.feature_dir = feature_dir
        self.transform = transform
        self.batch_size = batch_size
        self.target_transform = target_transform      

        if stack:
            #from the folder, load all numpy files and combine them into one big numpy array
            #loop through all files in the folder
            for root, dirs, files in os.walk(self.feature_dir):
                for file in files:
                    if file.endswith('.npy'):
                        #load the numpy file
                        if not hasattr(self, 'avg_features'):
                            self.avg_features = np.load(os.path.join(root, file)).astype(np.float32)
                        else:
                            self.avg_features = np.concatenate((self.avg_features, np.load(os.path.join(root, file)).astype(np.float32)), axis=0)
                        print(self.avg_features.shape)
            self.avg_features = self.avg_features[~np.isnan(self.avg_features).any(axis=1)]
        else:
            self.avg_features = np.load(self.feature_dir).astype(np.float32)
            self.avg_features = self.avg_features[~np.isnan(self.avg_features).any(axis=1)]

    def __len__(self) -> int:
        return len(self.avg_features)

    def __getitem__(self, idx: int):
        if self.batch_size:
            feature = self.avg_features[idx:idx+self.batch_size]
            print(feature.shape)
        else:
            feature = self.avg_features[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            feature = self.target_transform(feature)
        return feature

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.training_start_time = time.strftime("%Y%m%d-%H%M")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        results_folder = make_results_folder('trained_model')
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), results_folder + f'/all_ViT_small_ump_input_700_small_nn_checkpoint_{self.training_start_time}.pth')
        self.val_loss_min = val_loss


class TrainFeatureReconstructor():

    def __init__(self, path, batch_size=32, epochs=1, learning_rate=1e-3):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")
        self.input_dim = 384
        # self.hidden_dim = [256, 128, 64, 32, 64, 128, 256] #big nn
        self.hidden_dim = [256, 64, 32, 16, 32, 64, 256] # small nn
        # self.hidden_dim = [1024, 512, 256, 64, 32, 16, 32, 64, 256, 512, 1024] #huge nn
        # self.hidden_dim = [256, 32] # wvn nn
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.data_path = path
        self.stack = True
        self.early_stopping = EarlyStopping(patience=10, verbose=True)

    # Training loop
    def train_loop(self, train_dataloader, loss_fn, optimizer):
        self.model.train()
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data in train_dataloader:
                inputs = targets = data.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs
                                     )
                loss = loss_fn(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Print statistics
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")

            meta = {'epoch': epoch, 'loss': epoch_loss}
            if (epoch+1) % 10 == 0:
                test_dict = self.test_loop(self.model, loss_fn)
                meta.update(test_dict)
                self.early_stopping(test_dict["test_loss"], self.model)
            wandb.log(meta)
            if self.early_stopping.early_stop:
                print("Early stopping")
                exit()
            

        print('Finished Training')

    def test_loop(self, model, loss_fn):
        # Set the model to evaluation mode
        model.eval()
        dataloader = self.test_dataloader
        num_batches = len(dataloader)
        test_loss = 0

        # Ensure no gradients are computed during test mode
        with torch.no_grad():
            for X in dataloader:
                X = X.to(self.device)
                # Forward pass: compute the model output          
                recon_X = model(X)
                # Compute the loss
                test_loss += loss_fn(recon_X, X).item()

        # Compute the average loss over all batches
        test_loss /= num_batches
        print(f"Test Error: \n Avg MSE Loss: {test_loss:>8f} \n")

        # test on one validation image
        mode = 'segment_wise'
        test_image_path = '/home/sebastian/ARIA/aria_recordings/Richmond_forest/mps_Richmond_forest_09_vrs/rgb/000388.png'
        figure = test_feature_reconstructor_with_model(mode,self.model, test_image_path)
        return dict(test_loss=test_loss, test_plot=figure)

    def data_split(self, dataset, train_split=0.8):
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size
        # training_data = dataset[:1000]
        # test_data = training_data
        training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    def main(self):
        
        # Creating DataLoader
        dataset = FeatureDataset(self.data_path, self.stack)

        # Model instantiation
        self.model = ReconstructMLP(self.input_dim, self.hidden_dim).to(self.device)
        print(self.model)

        # Loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Splitting the data
        train_dataloader, self.test_dataloader = self.data_split(dataset)

        # Training the model
        self.train_loop(train_dataloader, loss_fn, optimizer)

        # Testing the model
        self.test_loop(self.model, loss_fn)


if __name__ == '__main__':
    wandb.init(project='STEPP')

    path_to_features = f'/home/sebastian/Documents/code/seb_trav/results/all_non_ump/'
    TrainFeatureReconstructor(path_to_features, epochs=1000000).main()