import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_unit, output_size, num_lstm_layer):
        super().__init__()

        # input args
        self.in_channels = in_channels
        self.hidden_unit = hidden_unit
        self.output_size = output_size
        self.num_lstm_layer = num_lstm_layer

        # layer
        self.linear_input_layer = nn.Sequential(nn.Linear( self.in_channels, self.hidden_unit),
                                                nn.ReLU())

        self.lstm = nn.LSTM(input_size=self.hidden_unit, 
                            hidden_size=self.hidden_unit, 
                            num_layers=self.num_lstm_layer, 
                            batch_first=True)

        self.linear_output_layer = nn.Sequential(   nn.Linear(hidden_unit, self.output_size), 
                                                    nn.ReLU())

    def forward(self, x):

        # input shapes
        batch_size = x.shape[0]
        num_seqs = x.shape[1]
        seq_lengths = x.shape[2]

        # pass through layer
        x = self.linear_input_layer(x)
        x = x.reshape(batch_size * num_seqs, seq_lengths, -1)
        _, (hidden_state, _) = self.lstm(x)
        x = hidden_state[-1].reshape(batch_size, num_seqs, -1)
        return self.linear_output_layer(x)
