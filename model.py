from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule)
import torch

MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, het_mode = None):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        het_mode = het_mode)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class MixModel(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, het_mode = None):

        super().__init__()
        self.submodels = nn.ModuleList()
        self.het_mode = het_mode
        if het_mode == 'mix':
            modes_list = ['heterophily', 'homophily', 'original']
        else:
            modes_list = [het_mode]
        for mode in modes_list:
            self.submodels.append(Model(model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, het_mode = mode))
        self.lin = nn.Linear(in_features= len(modes_list)  *output_dim , out_features=output_dim )
    def forward (self, graph, x):
        if self.het_mode == 'mix':
            outs = []
            for model in self.submodels:
                out = model(graph, x)
                outs.append(out) 
            
            outs = torch.cat(outs, dim = 1)
            out = self.lin(outs)
        else:
            model = self.submodels[0]
            out = model(graph, x)
        # breakpoint()
        return out