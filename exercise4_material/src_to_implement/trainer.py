import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()# reset gradients
        output = self._model(x)  # forward pass through the network
        loss = self._crit(output, y)  # calculate the loss
        loss.backward()  # compute gradients by backward propagation
        self._optim.step()  # update weights
        return loss.item()  # return the loss


    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        with t.no_grad():
            output = self._model(x)  # forward pass through the network
            loss = self._crit(output, y)  # calculate the loss
            # predictions = t.argmax(output, dim=1)  # get the class predictions
        # return loss.item(), predictions  # return the loss and the predictions
        return loss.item(), output

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model.train()  # set the model to training mode

        total_loss = 0
        for x, y in tqdm(self._train_dl):
            if self._cuda:
                x, y = x.cuda(), y.cuda()  # transfer the batch to the GPU
            loss = self.train_step(x, y)
            total_loss += loss
        avg_loss = total_loss / len(self._train_dl)
        return avg_loss  # return the average loss for the epoch

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval() # set eval mode.
        with t.no_grad(): # disable gradient computation.
            total_loss = 0
            y_true = []
            y_pred = []
            for x, y in self._val_test_dl: # iterate through the validation set
                if self._cuda: # perform a validation step
                    x = x.cuda() #images
                    y = y.cuda() #labels
                loss, pred = self.val_test_step(x, y)
                total_loss += loss
                y_true.extend(y.cpu().numpy().tolist()) # save the predictions and the labels for each batch
                # y_pred.extend(self._model(x).cpu().numpy().tolist())
                y_pred.extend((pred.cpu() > 0.5).numpy().tolist())
        avg_loss = total_loss / len(self._val_test_dl) # calculate the average loss and average metrics of your choice.
        avg_f1_score = f1_score(y_true, y_pred, average='weighted')
        print(f"Validation Loss: {avg_loss:.4f}, F1 Score: {avg_f1_score:.4f}")
        return avg_loss, avg_f1_score # return the loss and print the calculated metrics


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        # create a list for the train and validation losses, and create a counter for the epoch
        #TODO
        train_losses = []
        val_losses = []
        val_min_loss = float("inf")
        epoch_counter = 0
        patience_counter =0
        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            #TODO
            if epoch_counter >= epochs:
                print("Training finished")
                break
            print(f"Epoch: {epoch_counter}")

            epoch_loss = self.train_epoch()# train for an epoch
            train_losses.append(epoch_loss)

            val_loss, metrics = self.val_test() # calculate the validation loss and metrics
            val_losses.append(val_loss)

            # save the model (can be restricted to epochs with improvement)
            save_best_only = True
            if save_best_only:
                if val_loss < val_min_loss:
                    print(f" Model is saved. Loss is decreased from {val_min_loss} to {val_loss}...")
                    self.save_checkpoint(0)
                    patience_counter = 0
                else :
                    patience_counter +=1
                    # print(f" Model isn't saved. Loss is changed from {val_min_loss} to {val_loss}...")
            else:
                self.save_checkpoint(epoch_counter)
                print(f" Model is saved.")

            val_min_loss = min(val_losses)
            # check whether early stopping should be performed
            if patience_counter >= self._early_stopping_patience:
                print("Early stopping----")
                break
            epoch_counter += 1

        return train_losses, val_losses # return the train and validation losses
