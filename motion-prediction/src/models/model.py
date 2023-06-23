from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.vectornet_modules.TrajDecoder import TrajDecoder
# from src.models.vectornet_modules.SelfAttention import SelfAttention
# from src.models.vectornet_modules.Subgraph import SubGraph

from src.models.generic_modules.MLPEncoder import MLPEncoder
from src.models.generic_modules.MLPDecoder import MLPDecoder
from src.models.generic_modules.LSTMEncoder import LSTMEncoder

########################################################################################################################


class BaseClass(nn.Module):
    def __init__(self, predicted_steps, num_predictions, loss_fun):
        super().__init__()

        # init passed args
        self.loss_fun = loss_fun
        self.predicted_steps = predicted_steps*2
        self.K = num_predictions
        self.output_size = self.predicted_steps*self.K

        # if probability is learned by model -> output is mean and std, therefore 2x the output size
        if loss_fun == 'GNLLL':
            self.output_size *= 2

    def predict_best(self, input_data, y, return_loss=False):

        # get model output
        out = self.forward(input_data)
        # expand dim of target to k predictions
        y = y.repeat(self.K, 1, 1).permute(1, 0, 2)

        # train with GNLLL
        if self.loss_fun == 'GNLLL':
            preds, log_std = out.chunk(2, dim=2)
            loss = F.gaussian_nll_loss(
                input=preds, target=y, var=log_std.exp(), reduction='none').mean(dim=2)
            loss_best, pred_best = self.get_best(preds, loss)

        # train with MSE loss
        elif self.loss_fun == 'MSE':
            preds = out
            loss = F.mse_loss(input=preds, target=y,
                              reduction='none').mean(dim=2)
            loss_best, pred_best = self.get_best(preds, loss)

        #  loss not implemented
        else:
            raise NotImplementedError

        # return loss and prediction if passed
        if return_loss:
            return pred_best, loss_best
        else:
            return pred_best

    def predict_all(self, input_data):
        # get model output
        if self.loss_fun == 'GNLLL':
            out, _ = self.forward(input_data).chunk(2, dim=2)
        else:
            out = self.forward(input_data)
        # reshape to [num_predictions, batch_size, 2*pred_len] so first dim is for each prediction
        return out.view(self.K, out.shape[0], -1)

    def get_best(self, preds, loss):
        # get the smalles losses
        loss_best, _ = loss.topk(k=1, dim=1, largest=False)
        # filter best predictions according to smallest error
        pred_best = None

        best_loss = loss[0]
        best_loss_idx = 0
        for i in range(len(loss)):
            if loss[i] < best_loss:
                best_loss = loss[i]
                best_loss_idx = i

        pred_best = preds[best_loss_idx]
        # return best predictions and their loss
        return loss_best.mean(), pred_best

    @abstractmethod
    def forward(self, x):
        pass

########################################################################################################################


# class VectorNet(BaseClass):
#     def __init__(self, predicted_steps, num_predictions, loss_fun, num_subgraph_layers=3, hidden_unit=64):
#         super().__init__(predicted_steps, num_predictions, loss_fun)

#         # init passed args
#         self.num_subgraph_layers = num_subgraph_layers
#         self.hidden_unit = hidden_unit

#         # VectorNet modules (model takes as input features [x_start, y_start, x_end, y_end, polyline type, timestep])
#         self.sub_graph = SubGraph(in_channels=6,
#                                   num_subgraph_layers=self.num_subgraph_layers,
#                                   hidden_unit=self.hidden_unit)

#         self.self_attention = SelfAttention(in_channels=self.hidden_unit * 2,
#                                             hidden_unit=self.hidden_unit)

#         self.traj_decoder = TrajDecoder(in_channels=self.hidden_unit,
#                                         hidden_unit=self.hidden_unit,
#                                         out_channels=self.output_size)

#     def forward(self, input_data):
#         # input_data:   [x, cluster]
#         # x:            [batch_dim, input_features]
#         # cluster:      [batch_dim, index]

#         # input data consists of features and cluster
#         x, cluster = input_data[0], input_data[1]

#         # pass through VectorNet layer
#         x = self.sub_graph(x, cluster)  # pass through subgraph
#         x = self.self_attention(x)  # global graph attention
#         # decode 0th polyline for prediction as polyline 0 is ego agent
#         x = self.traj_decoder(x[:, 0])
#         # reshape to [batch_dim, num_predictions, ...]
#         return x.view(x.shape[0], self.K, -1)

########################################################################################################################


class LSTM(BaseClass):
    def __init__(self, predicted_steps, num_predictions, loss_fun, num_lstm_layer=3, hidden_unit=128):
        super().__init__(predicted_steps, num_predictions, loss_fun)

        # init passed args
        self.num_lstm_layer = num_lstm_layer
        self.hidden_unit = hidden_unit

        # modules (model takes as input features [x_start, y_start, x_end, y_end])
        self.encoder = MLPEncoder(in_channels=4,
                                  output_size=self.hidden_unit)

        self.lstm = nn.LSTM(input_size=self.hidden_unit,
                            hidden_size=self.hidden_unit,
                            num_layers=self.num_lstm_layer,
                            batch_first=True)

        self.decoder = MLPDecoder(in_channels=self.hidden_unit,
                                  output_size=self.output_size)

    def forward(self, input_data):
        # input data: [[batch_dim, num_agents, seq_len, input_features]]

        # input data is only agent data so remove outer list dimension
        if not isinstance(input_data, list):
            input_data = [input_data]

        x = input_data[0]

        # input shapes
        batch_size = x.shape[0]
        num_agents = x.shape[1]
        seq_lengths = x.shape[2]

        # Pass the input sequences through decoder
        x = self.encoder(x)

        # reshape x so agents are flattend over batch dim and pass through lstm layer
        x = x.reshape(batch_size * num_agents, seq_lengths, -1)
        _, (hidden_state, _) = self.lstm(x)

        # reshape last output back to  [batch_dim, num_agents, seq_len, ... ] and get trajectory
        x = hidden_state[-1].reshape(batch_size, num_agents, -1)
        x = self.decoder(x[:, 0])  # only decode agent 0

        # reshape to [batch_dim, num_predictions, ...]
        return x.view(x.shape[0], self.K, -1)

    # coord_idx: 24 values
    def predict_single_coordinate(self, input_data, target, coord_idx):
        pred = self.predict_best(input_data, target)
        return pred[:, coord_idx]


########################################################################################################################

class VectorLSTM(BaseClass):
    def __init__(self, predicted_steps, num_predictions, loss_fun, num_subgraph_layers=3, num_lstm_layer=1, hidden_unit=64):
        super().__init__(predicted_steps, num_predictions, loss_fun)

        # init passed args
        self.num_subgraph_layers = num_subgraph_layers
        self.num_lstm_layer = num_lstm_layer
        self.hidden_unit = hidden_unit

        # modules
        self.lstm_encoder = LSTMEncoder(in_channels=4,
                                        hidden_unit=self.hidden_unit,
                                        # hidden_unit*2 to match output size of the subgraph
                                        output_size=self.hidden_unit*2,
                                        num_lstm_layer=self.num_lstm_layer)

        self.sub_graph = SubGraph(in_channels=5,
                                  num_subgraph_layers=self.num_subgraph_layers,
                                  hidden_unit=self.hidden_unit)

        self.self_attention = SelfAttention(in_channels=self.hidden_unit * 2,
                                            hidden_unit=self.hidden_unit)

        self.traj_decoder = TrajDecoder(in_channels=self.hidden_unit,
                                        hidden_unit=self.hidden_unit,
                                        out_channels=self.output_size)

    def forward(self, input_data):
        # input_data:   [x_agent, x_map, cluster]
        # x_agent:      [batch_dim, num_agents, seq_len, input_features]
        # x_map:        [batch_dim, input features]
        # cluster_map:  [batch_dim, index]

        # input data consists of features and cluster
        x_agent, x_map, cluster = input_data[0], input_data[1], input_data[2]

        # pass through layer
        # Pass agents through lstm encoder
        x_agent = self.lstm_encoder(x_agent)
        x_map = self.sub_graph(x_map, cluster)  # pass map through subgraph
        x = torch.cat([x_agent, x_map], dim=1)  # concat data
        x = self.self_attention(x)              # pass through global attention
        # decode 0th polyline for prediction as polyline 0 is ego agent
        x = self.traj_decoder(x[:, 0])

        # reshape to [batch_dim, num_predictions, ...]
        return x.view(x.shape[0], self.K, -1)
