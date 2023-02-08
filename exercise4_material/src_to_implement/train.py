import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

validation_size = 0.2
batch_size=32
learning_rate = 1e-3
epochs = 20
early_stop_patience = 10

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
df = pd.read_csv('data.csv', sep=';')
train_df, val_df = train_test_split(df, test_size=validation_size)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_df, 'train')
val_dataset = ChallengeDataset(val_df, 'val')
train_dl = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_test_dl = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# create an instance of our ResNet model
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-imple bmented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
crit = t.nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
trainer = Trainer(model=model, crit=crit, optim=optimizer, train_dl=train_dl, val_test_dl=val_test_dl, early_stopping_patience=early_stop_patience, cuda=True)

# go, go, go... call fit on trainer
res = trainer.fit(epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')