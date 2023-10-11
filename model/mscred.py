import torch
from torch import nn
import torch.nn.functional as F
from conformer.model import Conformer


class ConformerMSCRED(nn.Module):
    def __init__(self, num_timesteps=5, attention=True, device="cpu"):
        super().__init__()
        self.Conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=32,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        self.Conformer1 = Conformer(
            input_dim=6,
            encoder_dim=6,
            num_encoder_layers=12,
            conv_kernel_size=5,
        ).to(device)
        self.Conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            device=device,
        )
        self.Conformer2 = Conformer(
            input_dim=3,
            encoder_dim=3,
            num_encoder_layers=12,
            conv_kernel_size=5,
        ).to(device)
        self.Conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.Conformer3 = Conformer(
            input_dim=2,
            encoder_dim=2,
            num_encoder_layers=12,
            conv_kernel_size=5,
        ).to(device)
        self.Conv4 = nn.Conv3d(
            in_channels=128, out_channels=256, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.Conformer4 = Conformer(
            input_dim=1,
            encoder_dim=1,
            num_encoder_layers=12,
            conv_kernel_size=5,
        ).to(device)
        self.Deconv4 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.Deconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=1
        )
        self.Deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.Deconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        input X with shape: (batch, seq_len, num_channels, height, width)

        """

        x_c1_seq = F.selu(self.Conv1(x))
        x_c1 = self.Conformer1(x_c1_seq.permute(0, 2, 1, 3, 4))

        x_c2_seq = F.selu(self.Conv2(x_c1_seq))
        x_c2 = self.Conformer2(x_c2_seq.permute(0, 2, 1, 3, 4))

        x_c3_seq = F.selu(self.Conv3(x_c2_seq))
        x_c3 = self.Conformer3(x_c3_seq.permute(0, 2, 1, 3, 4))

        x_c4_seq = F.selu(self.Conv4(x_c3_seq))
        x_c4 = self.Conformer4(x_c4_seq.permute(0, 2, 1, 3, 4))

        x_d4 = F.selu(
            self.Deconv4.forward(x_c4, output_size=[x_c3.shape[-1], x_c3.shape[-2]])
        )

        x_d3 = torch.cat((x_d4, x_c3), dim=1)
        x_d3 = F.selu(
            self.Deconv3.forward(x_d3, output_size=[x_c2.shape[-1], x_c2.shape[-2]])
        )

        x_d2 = torch.cat((x_d3, x_c2), dim=1)
        x_d2 = F.selu(
            self.Deconv2.forward(x_d2, output_size=[x_c1.shape[-1], x_c1.shape[-2]])
        )

        x_d1 = torch.cat((x_d2, x_c1), dim=1)
        x_rec = F.selu(
            self.Deconv1.forward(x_d1, output_size=[x_c1.shape[-1], x_c1.shape[-2]])
        )

        # X_rec - reconstructed signature matrix at last time step

        return x_rec
