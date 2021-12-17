import torch
import math
from torch import nn
from torch._C import device
from torch.nn.modules import linear


class PositionalEncoding(nn.Module):
    """Vanilla PositionalEncoding encoding

    According to the Attention is all you need paper, this positional
    encoding inserts the notion of position in the input features.

    Args:
        d_model : Hidden dimensionality of the input.
        max_len : Maximum length of a sequence to expect
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerTimeSeries(nn.Module):
    """Transformer model adapted to time series data

    This model follows the implementation from [1], which adapted the original
    model from [2] to perform a time series prediction task instead of the original
    task of NLP. The adaptation to the time series task removed the input
    embedding from the encoder but maintained the positional encoding process.

    References:
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    [2] Wu, Neo, et al. "Deep transformer models for time series forecasting: The influenza
        prevalence case." arXiv preprint arXiv:2001.08317 (2020).

    Args:
        device            : Torch device to process the model
        encoder_vector_sz : Size of the encoder input vector, which also define the
                            number the encoder's input features.
        encoder_mask      : Either use or not a mask to avoid look ahead in the encoder.
        num_layers        : Number of decoder and encoder layers.
        nhead             : Number of parallel multi attention heads.
        n_time_steps      : Number of timesteps used as inputs in the encoder and
                            decoder.
        d_model           : Model's vector dimmension.
        max_len : Array of objects with simulation data

    """
    def __init__(self,
                 device,
                 encoder_vector_sz=1,
                 encoder_mask=False,
                 encoder_feedback=False,
                 n_enc_layers=1,
                 nhead=2,
                 n_encoder_time_steps=30,
                 output_vector_sz=5,
                 d_model=100,
                 dropout=0.5,
                 n_input_time_steps=1,
                 iterative=True):

        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.device = device
        self.encoder_mask = encoder_mask
        self.encoder_feedback = encoder_feedback
        self.n_encoder_time_steps = n_encoder_time_steps
        self.output_vector_sz = output_vector_sz
        self.n_input_time_steps = n_input_time_steps
        self.iterative = iterative
        # Positional encoding used in the decoder and encoder
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder and encoder layers
        # Note that this implementation follows the data input format of
        # (batch, timestamps, features).
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_enc_layers)
        # Encoder input projection
        self.encoder_projection = nn.Linear(encoder_vector_sz, d_model)

        # Transformer output layer
        # Note that the output format of this model is (batch, pred_steps), where
        # as a time series predictor the pred_steps contain the estimation of future
        # behavior of one variable in time.
        self.linear = nn.Linear(d_model, 1, bias=True)
        self.linear2 = nn.Linear(n_encoder_time_steps,
                                 output_vector_sz,
                                 bias=True)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, in_sz, iterative=True):
        """Generates decoder mask

        This mask prevents the look ahead behavior in the decoder or encoder
        process.

        Args:
            sz : Size of timestep matrix mask.
        """
        # Provides mask for multi-step iterative scenarios
        if iterative:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0,
                                            float('-inf')).masked_fill(
                                                mask == 1, float(0.0))

        # Provides mask for the multi-step direct scenarios
        else:
            mask = torch.zeros((sz, sz))
            mask[in_sz:, in_sz:] = 1
            mask = mask.float().masked_fill(mask == 1,
                                            float('-inf')).masked_fill(
                                                mask == 0, float(0.0))

        return mask

    def encoder_process(self, src):
        """Encoder process of the Transformer network.

        According to [1], the tranformer encoder produces an encoded version
        of the entry series with the positional information and a self learned
        projection of the input data, generating a encoder matrix of dmodel
        dimension.

        Args:
            src: Input data of  the encoder, with shape
                 (batch, n_time_steps, encoder_vector_sz)
        """
        x = src
        x = self.encoder_projection(x)
        x = self.pos_encoder(x)

        if self.encoder_mask:
            # Mask to avoid look ahead behavior in the encoder process
            mask = self._generate_square_subsequent_mask(
                self.n_encoder_time_steps,
                self.n_input_time_steps,
                iterative=self.iterative).to(self.device)
        else:
            mask = None

        x = self.transformer_encoder(x, mask=mask)

        return x

    def init_weights(self):
        """Initialize the output layer weights.
        """
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """Send the information through the transformer network.
        """
        src = x
        src = self.encoder_process(src)
        out = self.linear(src)
        out = out.reshape(out.shape[0], out.shape[1])
        out = self.linear2(out)
        # out = self.out_fcn(out)

        return out

    def encoder_attention(self, src, layer_idx=0):
        """Return the attention matrix given an x input

        Args:
            src : Input tensor with shape (batch, timesteps, dmodel)
        """
        x = src
        x = self.encoder_projection(x)
        x = self.pos_encoder(x)

        if self.encoder_mask:
            # Mask to avoid look ahead behavior in the decoder process
            mask = self._generate_square_subsequent_mask(
                src.shape[1],
                self.n_input_time_steps,
                iterative=self.iterative).to(self.device)
        else:
            mask = None
        return self.transformer_encoder.layers[layer_idx].self_attn(
            x, x, x, attn_mask=mask)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = TransformerTimeSeries(device,
                                  n_encoder_time_steps=197,
                                  encoder_vector_sz=39,
                                  output_vector_sz=5).to(device)
    src = torch.rand(10, 197, 39).to(device)

    out = model(src)
    print(f'output {out.shape}')


if __name__ == '__main__':
    main()