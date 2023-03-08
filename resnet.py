import torch
from torch import nn

class ResNetAutoEncoder(nn.Module):

    def __init__(self, img_shape):
        super(ResNetAutoEncoder, self).__init__()
        self.img_shape = img_shape
        self.encoder = ResNetEncoder()
        self.decoder = ResNetDecoder(img_shape)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def get_feature_map(self, img):
        return self.encoder(img)
    
    @torch.no_grad()
    def test_loss(self, test_load, loss_metric) -> float:
        self.train(False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss = 0.0
        for batch_im, _ in test_load:
            batch_im = batch_im.to(device)
            output = self(batch_im)
            loss = loss + loss_metric(batch_im, output).item()
        self.train(True)
        return loss / len(test_load)

    @torch.no_grad()
    def get_embedding(self, img, scaler, pca):
        """
        scaler is an already fitted sklearn.preprocessing.StandardScaler object
        pca is an already fitted sklearn.decomposition.PCA object
        """
        feature_map = self.encoder(img)
        flat_feature_map = torch.nn.Flatten(start_dim=0)(feature_map).numpy()
        scaled_feature_map = scaler.fit_transform([flat_feature_map])
        pca_embedding = pca.fit_transform(scaled_feature_map) 
        
        return pca_embedding  


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace = True),
        )

        self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=2, downsample_method="pool")
        self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=2, downsample_method="conv")
        self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=2, downsample_method="conv")
        self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=2, downsample_method="conv")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(self, img_shape):
        super(ResNetDecoder, self).__init__()

        self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=2)
        self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=2)
        self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=2)
        self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=2)

        # After applying conv4 the shape of the out is (n_batch, 64, 112, 112)


        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
            nn.Upsample(size = img_shape)
        )
        # Sigmoid because images are reprensented
        # with values in [0, 1]
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x


class EncoderResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":
            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):
    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):
            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x

class EncoderResidualLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x


class DecoderResidualLayer(nn.Module):
    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x):
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x
