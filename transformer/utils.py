import time
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class TransformerSet(Dataset):
    """Transformer dataset object

   This dataset object ensures that the transformer network is
   feed correctly and also saves the true targets values for
   posterior evaluation.

    Args:
        x_matrix : Matrix containing the past steps of a univariate
                   time series. With shape (time_steps, window_of_features)

        y_matrix : Matrix containing the future steps of a univariate time series.
                   With shape (time_steps, window_of_features)

        n_time_steps: Number of timesteps used in the entry of the transformer.
    """
    def __init__(self, x_matrix, y_matrix, n_time_steps, encoder_only=False):
        self.encoder_only = encoder_only
        x_matrix = x_matrix.reshape(-1, n_time_steps, 1)

        if encoder_only:
            self.encoder_input = x_matrix

        else:
            n_encoder_input_steps = n_time_steps // 2
            self.encoder_input = x_matrix[:, :n_encoder_input_steps]
            self.decoder_input = x_matrix[:, n_encoder_input_steps:]
        self.true_target = y_matrix

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        if self.encoder_only:
            return self.encoder_input[idx], self.true_target[idx]
        else:
            return self.encoder_input[idx], self.decoder_input[
                idx], self.true_target[idx]


class TransformerTimeSet(Dataset):
    """Transformer dataset object

   This dataset object ensures that the transformer network is
   feed correctly and also saves the true targets values for
   posterior evaluation.

    Args:
        x_encoder : Matrix containing the past steps of a univariate
                   time series. With shape (time_steps, window_of_features)

        x_decoder : Matrix containing the past steps of a univariate
                   time series. With shape (time_steps, window_of_features)

        y_matrix : Matrix containing the future steps of a univariate time series.
                   With shape (time_steps, window_of_features)

        n_encoder_time_steps: Number of timesteps used in the encoder of the transformer.

        n_decoder_time_steps: Number of timesteps used in the decoder of the transformer.

        pred_size: Size of prediction in the tail of the y_matrix
    """
    def __init__(self, x_encoder, x_decoder, y_matrix, n_encoder_time_steps,
                 n_decoder_time_steps):

        x_encoder = x_encoder.reshape(-1, n_encoder_time_steps, 1)
        x_decoder = x_decoder.reshape(-1, n_decoder_time_steps, 1)
        self.encoder_input = x_encoder
        self.decoder_input = x_decoder
        self.true_target = y_matrix

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[
            idx], self.true_target[idx]


def batch_train(model,
                epoch,
                batch_size,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                set_size,
                device,
                encoder_only=False,
                informer_pred_sz=None):
    """Train the model in batch
    """
    model.train()  # Turn on the train mode
    batch_loss = 0.
    total_loss = 0.
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        if encoder_only:
            src, tgt_out = batch[0], batch[1]
            src = Variable(torch.Tensor(src.float())).to(device)
            tgt_out = Variable(torch.Tensor(tgt_out.float())).to(device)
        else:
            src, tgt_in, tgt_out = batch[0], batch[1], batch[2]
            src = Variable(torch.Tensor(src.float())).to(device)
            tgt_in = Variable(torch.Tensor(tgt_in.float())).to(device)
            tgt_out = Variable(torch.Tensor(tgt_out.float())).to(device)

        optimizer.zero_grad()
        if encoder_only:
            output = model(src)
        else:
            output = model((src, tgt_in))

        if informer_pred_sz is not None:
            output = output[:, -informer_pred_sz:]
            tgt_out = tgt_out[:, -informer_pred_sz:]

        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        total_loss += batch_loss
        log_interval = int(set_size / batch_size / 5)
        if i % log_interval == 0 and i > 0:
            cur_loss = batch_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(epoch, i, set_size // batch_size,
                                        scheduler.get_lr()[0],
                                        elapsed * 1000 / log_interval,
                                        cur_loss))
            batch_loss = 0
            start_time = time.time()

    return total_loss


def batch_val(model,
              val_loader,
              criterion,
              device,
              encoder_only=False,
              informer_pred_sz=None):
    model.eval()
    batch_loss = 0.
    total_loss = 0.
    for batch in val_loader:
        if encoder_only:
            src, tgt_out = batch[0], batch[1]
            src = Variable(torch.Tensor(src.float())).to(device)
            tgt_out = Variable(torch.Tensor(tgt_out.float())).to(device)
            output = model(src)
        else:
            src, tgt_in, tgt_out = batch[0], batch[1], batch[2]
            src = Variable(torch.Tensor(src.float())).to(device)
            tgt_in = Variable(torch.Tensor(tgt_in.float())).to(device)
            tgt_out = Variable(torch.Tensor(tgt_out.float())).to(device)
            output = model((src, tgt_in))

        if informer_pred_sz is not None:
            output = output[:, -informer_pred_sz:]
            tgt_out = tgt_out[:, -informer_pred_sz:]
        loss = criterion(output, tgt_out)
        batch_loss += loss.item()
        total_loss += batch_loss

    return total_loss