

import math
import torch
import torch.nn as nn
import torch.nn.functional as FUNC

#Defining the bottlenect block for the network
class BottleneckBlock(nn.Module):
    def __init__(self, Next_feat_size, Output_feat_size, DROPOUT=0.0):
        super(BottleneckBlock, self).__init__()
        internal_feat_size = Output_feat_size * 4
        self.bn1 = nn.BatchNorm2d(Next_feat_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(Next_feat_size, internal_feat_size, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_feat_size)
        self.conv2 = nn.Conv2d(internal_feat_size, Output_feat_size, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = DROPOUT
    
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = FUNC.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = FUNC.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

#defining the transition block, includes reduction(compression) and dropout
class TransitionBlock(nn.Module):
    def __init__(self, Next_feat_size, Output_feat_size, DROPOUT=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(Next_feat_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(Next_feat_size, Output_feat_size, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = DROPOUT
    
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = FUNC.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return FUNC.avg_pool2d(out, 2)

# Defining the dense block for making dense layers
class DenseBlock(nn.Module):
    def __init__(self, nb_layers, Next_feat_size, GRT_rate, block, DROPOUT=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, Next_feat_size, GRT_rate, nb_layers, DROPOUT)
    
    def _make_layer(self, block, Next_feat_size, GRT_rate, nb_layers, DROPOUT):
        layers = []
        for i in range(nb_layers):
            layers.append(block(Next_feat_size + i * GRT_rate, GRT_rate, DROPOUT))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth=120, tot_classes=10,
                    GRT_rate=16,
                    reduction=0.6,
                    DROPOUT=0.2):
        super(DenseNet3, self).__init__()

        Next_feat_size = 2 * GRT_rate
        n = (depth - 4) // 3
        
        n = n // 2
        block = BottleneckBlock
        
        
        # CONV layer after input and before the Dense Block
        self.conv1 = nn.Conv2d(3, Next_feat_size, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # DENSE BLOCK 1
        self.block1 = DenseBlock(n, Next_feat_size, GRT_rate, block, DROPOUT)
        Next_feat_size = int(Next_feat_size + n * GRT_rate)
        self.trans1 = TransitionBlock(Next_feat_size, int(math.floor(Next_feat_size * reduction)), DROPOUT=DROPOUT)
        Next_feat_size = int(math.floor(Next_feat_size * reduction))
        
        # DENSE BLOCK 2
        self.block2 = DenseBlock(n, Next_feat_size, GRT_rate, block, DROPOUT)
        Next_feat_size = int(Next_feat_size + n * GRT_rate)
        self.trans2 = TransitionBlock(Next_feat_size, int(math.floor(Next_feat_size * reduction)), DROPOUT=DROPOUT)
        Next_feat_size = int(math.floor(Next_feat_size * reduction))
        
        # DENSE BLOCK 3
        self.block3 = DenseBlock(n, Next_feat_size, GRT_rate, block, DROPOUT)
        Next_feat_size = int(Next_feat_size + n * GRT_rate)
        
        # FULLY CONNECTED AND FINAL GLOBAL AVERAGE POOLING
        self.bn1 = nn.BatchNorm2d(Next_feat_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(Next_feat_size, tot_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.in_planes = Next_feat_size

        for Mod in self.modules():
            if isinstance(Mod, nn.Conv2d):
                n = Mod.kernel_size[0] * Mod.kernel_size[1] * Mod.out_channels
                Mod.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(Mod, nn.BatchNorm2d):
                Mod.weight.data.fill_(1)
                Mod.bias.data.zero_()
            elif isinstance(Mod, nn.Linear):
                Mod.bias.data.zero_()
    
    #Forward propagation of constructed network
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(-1, self.in_planes)
        return self.fc(out)




