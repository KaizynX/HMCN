import torch
import torch.nn as nn
from typing import List


class Dense(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: nn.Module, dropout_rate: float = None):
        super().__init__()
        self.label_size = output_size
        self.activation = activation
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class HMCNFModel(nn.Module):
    def __init__(
        self,
        features_num: int,
        classes_num: int,
        class_num_list: List[int],
        beta: float = 0.5,
        dropout_rate: float = 0.1,
        relu_size: float = 384,
    ):
        super().__init__()
        self.D = features_num
        self.C = classes_num
        self.H = len(class_num_list)
        self.beta = beta

        self.W_G_list = nn.ModuleList(
            [Dense(self.D, relu_size, nn.ReLU(), dropout_rate)]
            + [Dense(self.D + relu_size, relu_size, nn.ReLU(), dropout_rate) for _ in range(self.H - 1)]
            # + [Dense(self.D + relu_size, self.C, nn.Sigmoid(), dropout_rate)]
            + [Dense(self.D + relu_size, self.C, nn.Sigmoid())]
        )
        self.W_T_list = nn.ModuleList([Dense(relu_size, relu_size, nn.ReLU(), dropout_rate) for _ in class_num_list])
        # self.W_L_list = nn.ModuleList([Dense(relu_size, C_h, nn.Sigmoid(), dropout_rate) for C_h in class_num_list])
        self.W_L_list = nn.ModuleList([Dense(relu_size, C_h, nn.Sigmoid()) for C_h in class_num_list])

    def forward(self, x):
        A_G = None
        P_L_total = None
        for i in range(self.H):
            if i == 0:
                A_G = self.W_G_list[i](x)
            else:
                A_G = self.W_G_list[i](torch.cat([A_G, x], dim=1))
            A_L = self.W_T_list[i](A_G)
            P_L = self.W_L_list[i](A_L)
            P_L_total = P_L if P_L_total is None else torch.cat([P_L_total, P_L], dim=1)
        P_G = self.W_G_list[-1](torch.cat([A_G, x], dim=1))
        P_F = self.beta * P_G + (1 - self.beta) * P_L_total
        return P_F
