# -*- coding:utf-8 -*-
"""Model definition of ETNas."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

MODEL_MAPPINGS = {'ET-NAS-A': '2-_32_2-11-112-1121112.pth', 
                'ET-NAS-B': '031-_32_1-1-221-11121.pth', 
                'ET-NAS-C': '011-_32_2-211-2-111122.pth', 
                'ET-NAS-D': '031-_64_1-1-221-11121.pth', 
                'ET-NAS-E': '10001-_64_4-111-11122-1111111111111112.pth', 
                'ET-NAS-F': '011-_64_21-211-121-11111121.pth', 
                'ET-NAS-G': '10001-_64_4-111111111-211112111112-11111.pth', 
                'ET-NAS-H': '211-_64_41-211-121-11111121.pth', 
                'ET-NAS-I': '02031-a02_64_111-2111-21111111111111111111111-211.pth', 
                'ET-NAS-J': '211-_64_411-2111-21111111111111111111111-211.pth', 
                'ET-NAS-K': '02031-a02_64_1121-111111111111111111111111111-21111111211111-1.pth', 
                'ET-NAS-L': '23311-a02c12_64_211-2111-21111111111111111111111-211.pth'}

def conv3x3(in_channel, out_channel, stride=1, groups=1, bias=False):
    """Construct a convolution 3x3 layer with SAME padding, and dilation=1.

    :param in_channel: number of input channels
    :type in_channel: int
    :param out_channel: number of output channels
    :type out_channel: int
    :param stride: stride
    :type stride: int or tuple of int
    :param groups: number of groups
    :type groups: int
    :param bias: whether bias is needed
    :type bias: bool
    :return: 3x3 convolution layer with specified value of parameters
    :rtype: nn.Conv2d
    """
    if groups == 0:
        raise ValueError('Number of groups cannot be 0.')
    if in_channel % groups != 0:
        raise ValueError('In channel "{}" is not a multiple of groups: "{}"'.format(
            in_channel, groups))
    if out_channel % groups != 0:
        raise ValueError('Out channel "{}" is not a multiple of groups: "{}"'.format(
            out_channel, groups))

    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1,
                     stride=stride, groups=groups, bias=bias)


def conv1x1(in_channel, out_channel, stride=1, bias=False):
    """Construct a convolution 1x1 layer without padding, groups=1, and dilation=1.

    :param in_channel: number of input channels
    :type in_channel: int
    :param out_channel: number of output channels
    :type out_channel: int
    :param stride: stride
    :type stride: int or tuple of int
    :param bias: whether bias is needed
    :type bias: bool
    :return: 1x1 convolution layer with specified value of parameters
    :rtype: nn.Conv2d
    """
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=bias)


def conv3x3_base(in_channel, out_channel, stride=1, base_channel=1):
    """Construct a convolution 3x3 layer of specified base channel with SAME padding, dilation=1.

    :param in_channel: number of input channels
    :type in_channel: int
    :param out_channel: number of output channels
    :type out_channel: int
    :param stride: stride
    :type stride: int or tuple of int
    :param base_channel: base channel, which is "in_channel // groups"
    :type base_channel: int
    :return: 3x3 convolution layer with specified value of parameters
    :rtype: nn.Conv2d
    """
    return conv3x3(in_channel, out_channel, stride, in_channel // base_channel)


def conv3x3_sep(in_channel, out_channel, stride):
    """Construct a depthwise-separable 3x3 layer, which contains a 3x3 conv layer and a 1x1 conv layer.

    :param in_channel: number of input channels
    :type in_channel: int
    :param out_channel: number of output channels
    :type out_channel: int
    :param stride: stride
    :return: depthwise-separable layer with with specified value of parameters
    :rtype: nn.Sequential
    """
    return nn.Sequential(
        conv3x3(in_channel, in_channel, stride, groups=in_channel),
        conv1x1(in_channel, out_channel))


# All the OPS available.
OPS = {
    'conv3': conv3x3,
    'conv1': conv1x1,
    'conv3_sep': conv3x3_sep,
    'conv3_grp2': partial(conv3x3, groups=2),
    'conv3_grp4': partial(conv3x3, groups=4),
    'conv3_base1': partial(conv3x3_base, base_channel=1),
    'conv3_base32': partial(conv3x3_base, base_channel=32)
}


def create_op(opt_name, in_channel, out_channel, stride=1):
    """Create an op in OPS, which always has a BN layer followed.

    :param opt_name: name of op, which is a key in OPS
    :type opt_name: str
    :param in_channel: number of input channels
    :type in_channel: int
    :param out_channel: number of output channels
    :type out_channel: int
    :param stride: stride
    :type stride: int or tuple of int
    :return: specified op followed by BN
    :rtype: nn.Sequential
    """
    layer = OPS[opt_name](in_channel, out_channel, stride)
    bn = nn.BatchNorm2d(out_channel)

    # Initialization
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)

    return nn.Sequential(layer, bn)


class AddBlock(nn.Module):
    """Skip connection with "add" operator.

    """
    def __init__(self, channels, strides, num1, num2):
        """Construct add block, which connect place {num1} and place {num2}.

        :param channels: channels of all places
        :type channels: list of int
        :param strides: strides of all places
        :type strides: list of int
        :param num1: place that connects from
        :type num1: int
        :param num2: place that connects to
        :type num2: int
        """
        super(AddBlock, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.conv = None

        # If the two places have different channels or strides, add a 1x1 conv layer to make them same.
        if strides[num1] != strides[num2] or channels[num1] != channels[num2]:
            self.conv = create_op('conv1', channels[num1], channels[num2], strides[num2] // strides[num1])

    def forward(self, x):
        """Calculate the forward propagation of block.

        :param x: tensors of all places
        :type x: list of tensors
        :return: tensors of all places after the skip connection
        :rtype: list of tensors
        """
        x1, x2 = x[self.num1], x[self.num2]
        if self.conv is not None:
            x1 = self.conv(x1)
        x[self.num2] = x1 + x2
        return x


class ConcatBlock(nn.Module):
    """Skip connection with "concat" operator.

    """
    def __init__(self, channels, strides, num1, num2):
        """Construct concat block, which connect place {num1} and place {num2}.

        :param channels: channels of all places
        :type channels: list of int
        :param strides: strides of all places
        :type strides: list of int
        :param num1: place that connects from
        :type num1: int
        :param num2: place that connects to
        :type num2: int
        """
        super(ConcatBlock, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.conv = None

        # If the two places have different strides, add a 1x1 conv layer to make them same.
        if strides[num1] != strides[num2]:
            self.conv = create_op('conv1', channels[num1], channels[num1], strides[num2] // strides[num1])

        # Add the channel number of num1 to that of num2
        channels[self.num2] += channels[self.num1]

    def forward(self, x):
        """Calculate the forward propagation of block.

        :param x: tensors of all places
        :type x: list of tensors
        :return: tensors of all places after the skip connection
        :rtype: list of tensors
        """
        x1, x2 = x[self.num1], x[self.num2]
        if self.conv is not None:
            x1 = self.conv(x1)
        x[self.num2] = torch.cat([x1, x2], 1)
        return x


class EncodedBlock(nn.Module):
    """Block with given encoding.

    """
    def __init__(self, block_encoding, in_channel, op_names, stride=1, channel_increase=1):
        """Construct a block with given encoding.

        :param block_encoding: encoding of the block
        :type block_encoding: str
        :param in_channel: number of input channels
        :type in_channel: int
        :param op_names: list of available ops (for decoding)
        :type op_names: list of str
        :param stride: stride of the block
        :type stride: int
        :param channel_increase: channel increase multiplier
        :type channel_increase: int
        """
        super(EncodedBlock, self).__init__()
        out_channel = in_channel * channel_increase
        layer_channels = [in_channel]

        # layer_encoding and connection_encoding are splitted by '-'
        if '-' not in block_encoding:
            block_encoding = block_encoding + '-'
        layer_encoding, connection_encoding = block_encoding.split('-')
        # out channel of the last layer always equals to that of the block, so leave out a '2' in encoding
        layer_encoding = layer_encoding + "2"
        num_layers = len(layer_encoding) // 2

        # Each connection contains 3 characters. Sort them by output places.
        connect_parts = [connection_encoding[i:i + 3] for i in range(0, len(connection_encoding), 3)]
        connect_parts = sorted(connect_parts, key=lambda x: x[2])
        # There is always an add connection from start to end of the block
        connect_parts.append("a0{}".format(num_layers))
        connect_index = 0

        # Always add stride to the first layer which is not conv 1x1.
        stride_place = 0
        while stride_place + 1 < num_layers and layer_encoding[stride_place * 2] == '1':
            stride_place += 1
        strides = [1] * (stride_place + 1) + [stride] * (num_layers - stride_place)

        self.module_list = nn.ModuleList()
        for i in range(num_layers):
            # Layer modules
            layer_modules = nn.ModuleList()
            layer_op_name = op_names[int(layer_encoding[i * 2])]
            layer_in_channel = layer_channels[-1]
            layer_out_channel = out_channel * 2 ** int(layer_encoding[i * 2 + 1]) // 4
            layer_channels.append(layer_out_channel)
            layer_stride = stride if i == stride_place else 1
            layer = create_op(layer_op_name, layer_in_channel, layer_out_channel, layer_stride)
            layer_modules.append(layer)

            # Zero init BN of the last layer
            if i + 1 == num_layers:
                nn.init.constant_(layer[-1].weight, 1)

            # Connection blocks
            while connect_index < len(connect_parts) and int(connect_parts[connect_index][2]) == i + 1:
                block_class = AddBlock if connect_parts[connect_index][0] == 'a' else ConcatBlock
                block = block_class(
                    layer_channels, strides, int(connect_parts[connect_index][1]), i + 1)
                layer_modules.append(block)
                connect_index += 1

            self.module_list.append(layer_modules)

    def forward(self, x):
        """Calculate the forward propagation of block.

        :param x: input tensor
        :type x: tensor
        :return: output tensor
        :rtype: tensor
        """
        outs = [x]
        current = x

        for module_layer in self.module_list:
            for i, layer in enumerate(module_layer):
                if i == 0:
                    # Layer modules
                    outs.append(layer(current))
                else:
                    # Skip connections
                    outs = layer(outs)
            # relu always added after all skip connections
            current = F.relu(outs[-1], inplace=True)

        return current


class ETNas(nn.Module):
    """Main structure of ETNas.

    """
    def __init__(self, encoding, op_names=None, num_classes=1000):
        """Construct a ETNas network with given encoding, op_names and num_classes.

        :param encoding: model encoding
        :type encoding: str
        :param op_names: list of operation names, or None if default ops are used
        :type op_names: list str
        :param num_classes: number of classes
        :type num_classes: int
        """
        super(ETNas, self).__init__()
        if op_names is None:
            op_names = ["conv3", "conv1", "conv3_grp2", "conv3_grp4", "conv3_base1", "conv3_base32", "conv3_sep"]

        block_encoding, num_first_channel, macro_encoding = encoding.split('_')
        self.macro_encoding = macro_encoding
        curr_channel, index = int(num_first_channel), 0
        layers = [
            create_op('conv3', 3, curr_channel // 2, stride=2),
            nn.ReLU(inplace=True),
            create_op('conv3', curr_channel // 2, curr_channel // 2),
            nn.ReLU(inplace=True),
            create_op('conv3', curr_channel // 2, curr_channel, stride=2),
            nn.ReLU(inplace=True)
        ]

        while index < len(macro_encoding):
            stride = 1
            if macro_encoding[index] == '-':
                stride = 2
                index += 1

            channel_multiplier = int(macro_encoding[index])
            block = EncodedBlock(block_encoding, curr_channel, op_names, stride, channel_multiplier)
            layers.append(block)
            curr_channel *= channel_multiplier
            index += 1

        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=curr_channel, out_features=num_classes)

    def forward(self, x):
        """Calculate the forward propagation of the network.

        :param x: input tensor
        :type x: tensor
        :return: output tensor
        :rtype: tensor
        """
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

