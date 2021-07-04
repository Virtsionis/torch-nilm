import torch.nn as nn

class SAED_transfer(nn.Module):

    def __init__(self, file_path):
        super().__init__()

        model = SAED_transfer.load_from_checkpoint(file_path)

        frozen_layers = list(model.children())[:-2]
        self.feature_extractor = nn.Sequential(*frozen_layers)
        # layers are frozen by using eval()
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        trainable_layers = list(model.children())[-1]
        self.classifier = nn.Sequential(*trainable_layers)

    # returns the feature tensor from the frozen_layers
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x[:, -1, :]
        out = self.classifier(x)
        return out
