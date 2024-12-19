import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBlock(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        return self.conv(x)


class MultiViewUNet(Model):
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super().__init__()

        self.n_channels = [init_features * 2 ** i for i in range(5)]  # [32, 64, 128, 256, 512]

        # Encoder blocks
        self.enc1 = ConvBlock(self.n_channels[0])
        self.enc2 = ConvBlock(self.n_channels[1])
        self.enc3 = ConvBlock(self.n_channels[2])
        self.enc4 = ConvBlock(self.n_channels[3])
        self.enc5 = ConvBlock(self.n_channels[4])

        # Decoder blocks
        self.dec4 = ConvBlock(self.n_channels[3])
        self.dec3 = ConvBlock(self.n_channels[2])
        self.dec2 = ConvBlock(self.n_channels[1])
        self.dec1 = ConvBlock(self.n_channels[0])

        # Full-scale skip connection blocks
        self.fs_enc = [ConvBlock(self.n_channels[0]) for _ in range(4)]
        self.fs_dec = [ConvBlock(self.n_channels[0]) for _ in range(4)]

        # Final output
        self.final = layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        # Encoder path with full-scale connections
        e1 = self.enc1(x)
        e1_fs = self.fs_enc[0](e1)

        e2 = self.enc2(layers.MaxPooling2D(2)(e1))
        e2_fs = self.fs_enc[1](e2)

        e3 = self.enc3(layers.MaxPooling2D(2)(e2))
        e3_fs = self.fs_enc[2](e3)

        e4 = self.enc4(layers.MaxPooling2D(2)(e3))
        e4_fs = self.fs_enc[3](e4)

        # Bottom
        e5 = self.enc5(layers.MaxPooling2D(2)(e4))

        # Decoder path with full-scale connections
        d4 = self.dec4(tf.concat([layers.UpSampling2D(2, interpolation='bilinear')(e5), e4], axis=-1))
        d4_fs = self.fs_dec[3](d4)

        d3 = self.dec3(tf.concat([layers.UpSampling2D(2, interpolation='bilinear')(d4), e3], axis=-1))
        d3_fs = self.fs_dec[2](d3)

        d2 = self.dec2(tf.concat([layers.UpSampling2D(2, interpolation='bilinear')(d3), e2], axis=-1))
        d2_fs = self.fs_dec[1](d2)

        d1 = self.dec1(tf.concat([layers.UpSampling2D(2, interpolation='bilinear')(d2), e1], axis=-1))
        d1_fs = self.fs_dec[0](d1)

        # Full-scale fusion
        fs_fusion1 = d1_fs + e1_fs
        fs_fusion2 = layers.UpSampling2D(2, interpolation='bilinear')(d2_fs + e2_fs)
        fs_fusion3 = layers.UpSampling2D(4, interpolation='bilinear')(d3_fs + e3_fs)
        fs_fusion4 = layers.UpSampling2D(8, interpolation='bilinear')(d4_fs + e4_fs)

        # Combine all full-scale features
        final_features = fs_fusion1 + fs_fusion2 + fs_fusion3 + fs_fusion4

        return self.final(final_features)


# Example usage
def test_network():
    model = MultiViewUNet()
    x = tf.random.normal((1, 256, 256, 3))  # Note: TensorFlow uses channels-last format
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    return model


if __name__ == "__main__":
    model = test_network()