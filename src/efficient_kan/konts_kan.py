import torch
import torch.nn.functional as F
from .spline import basis, second_derivative
import math
import torch.nn as nn
import random
import numpy as np



# For results reproduction
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class MMEncoder(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.encoder = nn.Linear(in_feature, out_feature)
    def forward(self, x):
        return self.encoder(x)
    
    
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        spline_order=3,
        scale_noise=0.1,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-2, 2],
        groups=8,
        need_relu=True
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        assert in_features % groups == 0 or groups == -1, 'Please select the group number divisible to input shape'
        
        if groups == -1:
            # if == -1, means no groups
            # print(in_features)
            self.groups = in_features
        else:
            self.groups = groups
            
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
                ).expand(self.groups, -1).contiguous().unsqueeze(0)
        self.h = h
        self.register_buffer("grid", grid)
        
        self.spline_lin_c = torch.nn.Parameter(
                            torch.Tensor(1, self.groups, grid_size + spline_order)
                            )
        self.spline_weight = torch.nn.Parameter(
                            torch.Tensor(in_features, out_features)
                            )
        
        self.num_neurons_in_g = in_features // self.groups
        
        # self.base_weight = torch.nn.Parameter(
        #                     torch.Tensor(in_features, out_features)
        #                     )
        
        self.grid_range = grid_range
        self.shortcut = nn.SiLU()
        self.grid_bias = torch.nn.Parameter(torch.empty_like(grid).uniform_(-h/8, h/8))
        self.scale_noise = scale_noise
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps
        self.need_relu = need_relu
        self.reset_parameters()



    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_spline)
        torch.nn.init.kaiming_uniform_(self.spline_lin_c, a=math.sqrt(5) * self.scale_spline)
        # torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_spline)
        # torch.nn.init.kaiming_uniform_(self.grid_bias, a=math.sqrt(5) * self.scale_spline)



    def _knots(self):
        knots = (self.grid + self.grid_bias).sort(dim=-1).values
        return knots.clamp(self.grid_range[0] - self.h * self.spline_order,
                           self.grid_range[-1] + self.h * self.spline_order)

    def _group_inputs(self, x):
        return x.reshape(-1, self.num_neurons_in_g, self.groups, 1)

    def b_splines(self, x):
        return basis(self._group_inputs(x), self._knots(), self.spline_order)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)        # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)                        # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)           # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()


    def forward(self, x: torch.Tensor, return_curvature=False):
        assert x.size(-1) == self.in_features
        grouped = self._group_inputs(x)
        knots = self._knots()
        values = basis(grouped, knots, self.spline_order)
        activation = (values * self.spline_lin_c).sum(-1).reshape_as(x)
        if self.need_relu:
            activation = activation + self.shortcut(x)
        output = activation @ self.spline_weight
        if return_curvature:
            d2 = second_derivative(grouped, knots, self.spline_lin_c, self.spline_order)
            return output, d2.sum(dim=(-1, -2)).mean()
        return output


class Knots_KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=20,
        spline_order=3,
        scale_noise=0.1,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-3, 3],
        groups=8,
        need_relu=True
        ):
        super(Knots_KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_hid = len(layers_hidden)
        self.layers = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()
        for idx, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layer_norm.append(
                nn.LayerNorm(in_features)
            )
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_spline=scale_spline,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    groups=groups,
                    need_relu=need_relu
                )
            )
    
    
    
    def forward(self, x: torch.Tensor, update_grid=False, normalize=False, return_curvature=False):
        if normalize:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-6).detach() 
            x /= stdev
        
        curvature = x.new_zeros(())
        hid_x = x
        for layer, norm in zip(self.layers, self.layer_norm):
            hid_x = norm(hid_x)
            if return_curvature:
                hid_x, term = layer(hid_x, return_curvature=True)
                curvature = curvature + term
            else:
                hid_x = layer(hid_x)

        if normalize:
            hid_x = hid_x * stdev
            hid_x = hid_x + means
        if return_curvature:
            return hid_x, curvature
        return hid_x



class MLP(torch.nn.Module):
    def __init__(self, layers_hidden, activation='ReLU'):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        
        if activation == 'ReLU':
            non_linear = nn.ReLU
        elif activation == 'LeakyReLU':
            non_linear = nn.LeakyReLU
        elif activation == 'GeLU':
            non_linear = nn.GELU
        elif activation == 'SiLU':
            non_linear = nn.SiLU
        
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:-1]):
            self.layers.append(nn.LayerNorm(in_features))
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(non_linear())
        self.layers.append(nn.Linear(layers_hidden[-2], layers_hidden[-1]))
    
    
    def forward(self, x, normalize=False):
        if normalize:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
            
        for layer in self.layers:
            x = layer(x)
        
        if normalize:
            x = x * stdev
            x = x + means
        return x