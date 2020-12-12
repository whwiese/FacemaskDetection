import torch
import torch.nn as nn
"""
ARCHITECTURE FORMAT: 
(K,C,S,P) - ConvNet with Kernel Size=K, Output Filters=C, Stride=S, Padding=P
"M" - 2x2 Max Pooling Layer with Stride = 2
[(K,C,S,P),(K,C,S,P),N] - Tuples signify CovNets with same format as above,
N signifies number of times to repeat sequence of conv layers
"""

def create_darknet(architecture, in_channels=3):
    layers = []

    for layer in architecture:
        if type(layer) == tuple:
            layers += [CNNBlock(in_channels,layer[1],layer[0],layer[2],layer[3])]
            in_channels = layer[1]
        elif type(layer) == str:
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        elif type(layer) == list:
            for _ in range(layer[-1]):
                for conv in layer[:-1]:
                    layers += [CNNBlock(in_channels,conv[1],conv[0],conv[2],conv[3])]
                    in_channels = conv[1]

    return nn.Sequential(*layers)

#A CNN-->Batch Norm--> Leaky ReLU block that will make up most of the DarkNet structure
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1, padding=0):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size,  bias=False,
                stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.LeakyReLU(x)
        return x

class YOLOv1(nn.Module):
    def __init__(
            self, num_classes, in_channels=3, grid_size=7, num_boxes=2):
        super().__init__()

        yolov1_architecture = [
            (7, 64, 2, 3),
            "m",
            (3, 192, 1, 1),
            "m",
            (1, 128, 1, 0),
            (3, 256, 1, 1),
            (1, 256, 1, 0),
            (3, 512, 1, 1),
            "m",
            [(1, 256, 1, 0), (3, 512, 1, 1), 4],
            (1, 512, 1, 0),
            (3, 1024, 1, 1),
            "m",
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1024, 1, 1),
            (3, 1024, 2, 1),
            (3, 1024, 1, 1),
            (3, 1024, 1, 1),
        ]

        self.in_channels = in_channels
        self.darknet = create_darknet(yolov1_architecture)
        self.fcs = self.create_fcs(num_classes, grid_size, num_boxes)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        return x

    def create_fcs(self, num_classes, grid_size, num_boxes):
        S,B,C = grid_size, num_boxes, num_classes
        return  nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024*7*7, 1024), #Original paper uses output size of 4096
                nn.Dropout(0.0),
                nn.LeakyReLU(0.1),
                nn.Linear(1024,S*S*(C+B*5)), #rehspe later for SxSx(C+B*5) output
                )

class YOLOv2_lite(nn.Module):
    """
    I call this YOLOv2 lite because it implements the new darknet19
    architecture, but does not contain the passthrough layer which concatenates
    a reorganized version of the output of darknet pt1 to the output
    of darknet pt 2. The passthrough layer supposedly helps the
    model learn based on fine-grained features in the image.
    """
    def __init__(self, num_classes, in_channels=3, grid_size=7, num_boxes=2):
        super().__init__()
        
        yolov2_darknet_pt1 = [
            (3, 32, 1, 1),
            "M",
            (3, 64, 1, 1),
            "M",
            (3, 128, 1, 1),
            (1, 64, 1, 0),
            (3, 128, 1, 1),
            "M",
            (3, 256, 1, 1),
            (1, 128, 1, 0),
            (3, 256, 1, 1),
            "M",
            [(3, 512, 1, 1), (1, 256, 1, 0), 2],
            (3, 512, 1, 1),
        ]

        yolov2_darknet_pt2 = [
            "M",
            [(3, 1024, 1, 1), (1, 512, 1, 0), 2],
            [(3, 1024, 1, 1), 3],
        ]

        self.darknet_pt1 = create_darknet(yolov2_darknet_pt1)
        self.darknet_pt2 = create_darknet(yolov2_darknet_pt2, in_channels=512)
        self.last_conv= nn.Conv2d(
                1024, num_classes+(num_boxes*5), 1,  bias=False,
                stride=1, padding=0
        )

    def forward(self, x):
        x_512 = self.darknet_pt1(x)
        x_1024 = self.darknet_pt2(x_512) 
        x_out = self.last_conv(x_1024)
        return x_out

"""
def test_v1(grid_size = 7, num_boxes=2, num_classes=20):
    model = YOLOv1(num_classes,3,grid_size,num_boxes)
    x = torch.zeros((10,3,448,448))
    print(model(x).shape)

def test_v2_lite(grid_size = 13, num_boxes=2, num_classes=20):
    model = YOLOv2_lite(num_classes,3,grid_size,num_boxes)
    x = torch.zeros((10,3,416,416))
    print(model(x).shape)

test_v1()
test_v2_lite()
"""
