import torch.nn as nn

class FFAutoencoder(nn.Module):
    def __init__(self, in_features, latent_dim, hidden_dimensions):
        super(FFAutoencoder, self).__init__()
        self.down , self.up = self.__prepare_layers(in_features, latent_dim, hidden_dimensions)

    def __prepare_layers(self, in_features, latent_dim, hidden_dimensions):
        # Encoder layers
        encoder_layers = [nn.Linear(in_features=in_features, out_features=hidden_dimensions[0])]
        for i in range(1, len(hidden_dimensions)):
            encoder_layers.append(nn.Linear(in_features=hidden_dimensions[i - 1], out_features=hidden_dimensions[i]))
        encoder_layers.append(nn.Linear(in_features=hidden_dimensions[-1], out_features=latent_dim))

        # Decoder layers
        decoder_layers = [nn.Linear(in_features=latent_dim, out_features=hidden_dimensions[-1])]
        for i in range(len(hidden_dimensions) - 1, 0, -1):
            decoder_layers.append(nn.Linear(in_features=hidden_dimensions[i], out_features=hidden_dimensions[i - 1]))
        decoder_layers.append(nn.Linear(in_features=hidden_dimensions[0], out_features=in_features))

        return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)



    def forward(self, x):
        latent = self.down(x)
        reconstructed = self.up(latent)
        return reconstructed