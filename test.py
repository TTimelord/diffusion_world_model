from escnn import nn
from escnn import gspaces  
import escnn.group as group

import torch

torch.manual_seed(32)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        r2_act = gspaces.rot2dOnR2(N=4) 
        self.r2_act = gspaces.rot2dOnR2(N=4)
        self.conv1 = nn.R2Conv(
            in_type=nn.FieldType(r2_act,  1*[r2_act.trivial_repr]),
            out_type=nn.FieldType(r2_act,  3*[r2_act.regular_repr]),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = torch.nn.SiLU()
            # nn.PointwiseMaxPool2D(nn.FieldType(r2_act,  3*[r2_act.regular_repr]), 2),
            # nn.IIDBatchNorm2d(nn.FieldType(r2_act,  3*[r2_act.regular_repr])),
        self.conv2 = nn.R2Conv(
            in_type=nn.FieldType(r2_act,  3*[r2_act.regular_repr]),
            out_type=nn.FieldType(r2_act,  1*[r2_act.trivial_repr]),
            kernel_size=3,
            padding=1
        )
    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]))
        x = self.conv1(x).tensor
        x = self.relu(x)
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 3 * [self.r2_act.regular_repr]))
        out = self.conv2(x)
        return out.tensor


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.r2_act = gspaces.rot2dOnR2(N=4)  
        self.conv1 = nn.R2Conv(
            in_type=nn.FieldType(self.r2_act,  1*[self.r2_act.trivial_repr]),
            out_type=nn.FieldType(self.r2_act,  4*[self.r2_act.regular_repr]),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.maxpool = nn.PointwiseMaxPool2D(nn.FieldType(self.r2_act,  4*[self.r2_act.regular_repr]), 2)
        
        self.group = gspaces.no_base_space(group.CyclicGroup(N=4))
        self.in_type = nn.FieldType(self.group, 4 * [self.group.regular_repr])
        self.out_type = nn.FieldType(self.group, 4 * [self.group.trivial_repr])
        
        self.fc = nn.Linear(self.in_type, self.out_type)
        
    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]))
        x = self.maxpool(self.conv1(x)).tensor.reshape(1, -1)
        x = nn.GeometricTensor(x, self.in_type)
        x = self.fc(x).tensor
        # out = nn.GeometricTensor(out, self.in_type)
        # out = self.fc(out)
        return x


class Net3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.G = group.so2_group(maximum_frequency=2)
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = self.gspace.type(self.G.standard_representation())
        
        activation1 = nn.FourierELU(
            self.gspace,
            channels=4, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=1).irreps, # include all frequencies up to L=1
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 6 equally distributed points
            type='regular', N=4,   
        )
        
        # self.out_type = nn.FieldType(self.gspace, 4*[self.gspace.regular_repr])
        # self.out_type = self.gspace.type(self.G.regular_representation)
        # self.out_type = self.gspace.type(self.G.regular_representation)
        
        # self.group = gspaces.no_base_space(CyclicGroup(N=4))
        self.fc1 = nn.Linear(self.in_type, activation1.in_type)
        self.fc2 = nn.Linear(activation1.in_type, self.in_type)
        
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        out = self.fc1(x)
        print(out.tensor)
        out = self.fc2(out)
        return out.tensor
    
class Net4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.group = gspaces.no_base_space(group.CyclicGroup(N=4))
        
        self.fc1 = nn.Linear(nn.FieldType(self.group, 2 * [self.group.trivial_repr]), nn.FieldType(self.group, 4 * [self.group.regular_repr]))
        self.fc2 = nn.Linear(nn.FieldType(self.group, 4 * [self.group.regular_repr]), nn.FieldType(self.group, 2 * [self.group.trivial_repr]))
        
    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.group, 2 * [self.group.trivial_repr]))
        out = self.fc1(x)
        print(out)
        out = self.fc2(out)
        return out.tensor

# net = Net()
# x = torch.Tensor([
#     [0.0, 1.0, 2.0, 3.0],
#     [2.0, 3.0, 1.0, 3.0],
#     [1.5, 2.5, 3.5, 3.0],
#     [1.5, 2.5, 3.5, 3.0],
# ])

# out = net(x.reshape(1, 1, 4, 4)).squeeze()
# out_r = net(x.rot90(k=1, dims=(0, 1)).reshape(1, 1, 4, 4)).squeeze()

# print(out)
# print(out_r)


# net2 = Net2()

# x = torch.Tensor([
#     [0.0, 1.0],
#     [2.0, 3.0],
# ])

# out = net2(x.reshape(1, 1, 2, 2)).squeeze()
# out_r = net2(x.rot90(k=1, dims=(0, 1)).reshape(1, 1, 2, 2)).squeeze()

# print(out)
# print(out_r)


net3 = Net3()
x = torch.tensor(
    [
        [1.0, 2.0]
    ]
)
out = net3(x)

x_r = torch.tensor(
    [
        [-1.0, -2.0]
    ]
)

out_r = net3(x_r)

# print(out)
# print(out_r)


# net4 = Net4()
# x = torch.tensor(
#     [
#         [1.0, 2.0]
#     ]
# )
# out = net4(x)

# x_r = torch.tensor(
#     [
#         [-2.0, 1.0]
#     ]
# )

# out_r = net4(x_r)

# print(out)
# print(out_r)