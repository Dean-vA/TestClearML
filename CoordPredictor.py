#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from clearml import Task
task = Task.init(project_name='My Project', task_name='My Experiment')

class Coord_Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        import torch    
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(476288, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%%
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Coord_Predictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    model.to(device)
    #%%
    data = np.load('Object_coord_data.npz')
    ims = data['arr_0']
    seg_ims = data['arr_1']
    coords = data['arr_2']
    for i in range(10):
        data = np.load('Object_coord_data'+str(i)+'.npz')
        ims = np.vstack((ims,(data['arr_0'])))
        seg_ims = np.vstack((seg_ims,data['arr_1']))
        coords = np.vstack((coords,data['arr_2']))
    data = (ims,coords)
    #%%
    batch_size = 32
    batches = int(np.floor(len(ims)/batch_size))
    print(batches)
    #%%
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        for batch in range(batches):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(torch.Tensor(inputs[batch*batch_size:(batch+1)*batch_size]).moveaxis(3,1).to(device)/255)
            labels_tensor = torch.Tensor(labels[batch*batch_size:(batch+1)*batch_size,[0,1,2]]).to(device)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #print(running_loss)
            if batch % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 10:.8f}')
                running_loss = 0.0
            

    print('Finished Training')
    path = 'coordpredictXYZ.pt'
    torch.save(model.state_dict(), path)
    print('Model state dictionary save as: ' + path)

# %%
if __name__ == "__main__":
    main()
