"""
official code:
https://github.com/google/automl/tree/master/efficientnetv2
"""

import itertools
import math

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


class CBAM(layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7, name=None, **kwargs):
        super(CBAM, self).__init__(name=name, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channels = input_shape[-1]

        # 通道注意力部分
        self.channel_attention_gap = layers.GlobalAveragePooling2D()
        self.channel_attention_gmp = layers.GlobalMaxPooling2D()
        self.channel_attention_fc1 = layers.Dense(
            units=self.channels // self.reduction_ratio,
            activation='relu',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=l2(1e-4)
        )
        self.channel_attention_fc2 = layers.Dense(
            units=self.channels,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=l2(1e-4)
        )
        self.channel_attention_sigmoid = layers.Activation('sigmoid')

        # 空间注意力部分
        self.spatial_attention_conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=l2(1e-4)
        )

        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        # 通道注意力
        gap = self.channel_attention_gap(inputs)
        gmp = self.channel_attention_gmp(inputs)

        gap_fc = self.channel_attention_fc1(gap)
        gmp_fc = self.channel_attention_fc1(gmp)

        channel_weights = self.channel_attention_fc2(gap_fc) + self.channel_attention_fc2(gmp_fc)
        channel_weights = self.channel_attention_sigmoid(channel_weights)
        channel_weights = layers.Reshape((1, 1, self.channels))(channel_weights)

        # 应用通道注意力
        x = inputs * channel_weights

        # 空间注意力
        gap_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        gmp_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat = layers.Concatenate(axis=-1)([gap_spatial, gmp_spatial])

        spatial_weights = self.spatial_attention_conv(spatial_concat)

        # 应用空间注意力
        x = x * spatial_weights

        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config


class SE(layers.Layer):
    def __init__(self,
                 se_filters: int,
                 output_filters: int,
                 name: str = None):
        super(SE, self).__init__(name=name)

        self.se_reduce = layers.Conv2D(filters=se_filters,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       activation="swish",
                                       use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       name="conv2d")

        self.se_expand = layers.Conv2D(filters=output_filters,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       activation="sigmoid",
                                       use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       name="conv2d_1")

    def call(self, inputs, **kwargs):
        # Tensor: [N, H, W, C] -> [N, 1, 1, C]
        se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
        se_tensor = self.se_reduce(se_tensor)
        se_tensor = self.se_expand(se_tensor)
        return se_tensor * inputs


class MBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float = 0.25,
                 drop_rate: float = 0.,
                 use_cbam: bool = False,  # 新增CBAM参数
                 name: str = None):
        super(MBConv, self).__init__(name=name)

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)
        expanded_c = input_c * expand_ratio

        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = layers.Conv2D(
            filters=expanded_c,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm0 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())
        self.act0 = layers.Activation("swish")

        # Depth-wise convolution
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="depthwise_conv2d")
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())
        self.act1 = layers.Activation("swish")

        # SE
        num_reduced_filters = max(1, int(input_c * se_ratio))
        self.se = SE(num_reduced_filters, expanded_c, name="se")

        # 新增CBAM注意力
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(name=name + "_cbam" if name else "cbam")

        # Point-wise linear projection
        self.project_conv = layers.Conv2D(
            filters=out_c,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm2 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            # Stochastic Depth
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                            name="drop_path")

    def call(self, inputs, training=None):
        x = inputs

        x = self.expand_conv(x)
        x = self.norm0(x, training=training)
        x = self.act0(x)

        x = self.depthwise_conv(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        x = self.se(x)

        # 应用CBAM注意力
        if self.use_cbam:
            x = self.cbam(x)

        x = self.project_conv(x)
        x = self.norm2(x, training=training)

        if self.has_shortcut:
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)

            x = tf.add(x, inputs)

        return x


class FusedMBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float = 0.,
                 use_cbam: bool = False,  # 新增CBAM参数
                 name: str = None):
        super(FusedMBConv, self).__init__(name=name)
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        assert se_ratio == 0.

        self.has_shortcut = (stride == 1 and input_c == out_c)
        self.has_expansion = expand_ratio != 1
        expanded_c = input_c * expand_ratio

        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))

        if expand_ratio != 1:
            self.expand_conv = layers.Conv2D(
                filters=expanded_c,
                kernel_size=kernel_size,
                strides=stride,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                use_bias=False,
                name=get_conv_name())
            self.norm0 = layers.BatchNormalization(
                axis=-1,
                momentum=0.9,
                epsilon=1e-3,
                name=get_norm_name())
            self.act0 = layers.Activation("swish")

        # 新增CBAM注意力
        self.use_cbam = use_cbam
        if use_cbam and expand_ratio != 1:
            self.cbam = CBAM(name=name + "_cbam" if name else "cbam")

        self.project_conv = layers.Conv2D(
            filters=out_c,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else stride,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())

        if expand_ratio == 1:
            self.act1 = layers.Activation("swish")

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            # Stochastic Depth
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                            name="drop_path")

    def call(self, inputs, training=None):
        x = inputs
        if self.has_expansion:
            x = self.expand_conv(x)
            x = self.norm0(x, training=training)
            x = self.act0(x)

            # 应用CBAM注意力（仅在扩展阶段使用）
            if self.use_cbam:
                x = self.cbam(x)

        x = self.project_conv(x)
        x = self.norm1(x, training=training)
        if self.has_expansion is False:
            x = self.act1(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)

            x = tf.add(x, inputs)

        return x


class Stem(layers.Layer):
    def __init__(self, filters: int, name: str = None):
        super(Stem, self).__init__(name=name)
        self.conv_stem = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="conv2d")
        self.norm = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name="batch_normalization")
        self.act = layers.Activation("swish")

    def call(self, inputs, training=None):
        x = self.conv_stem(inputs)
        x = self.norm(x, training=training)
        x = self.act(x)

        return x


class Head(layers.Layer):
    def __init__(self,
                 filters: int = 1280,
                 num_classes: int = 1000,
                 drop_rate: float = 0.,
                 use_cbam: bool = False,  # 新增CBAM参数
                 name: str = None):
        super(Head, self).__init__(name=name)
        self.conv_head = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="conv2d")
        self.norm = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name="batch_normalization")
        self.act = layers.Activation("swish")

        # 在头部添加CBAM
        self.use_cbam = use_cbam
        if use_cbam:
            # 修复：处理name为None的情况
            cbam_name = name + "_cbam" if name else "head_cbam"
            self.cbam = CBAM(name=cbam_name)

        self.avg = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes,
                               kernel_initializer=DENSE_KERNEL_INITIALIZER)

        self.drop_rate = drop_rate
        if drop_rate > 0:
            self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training=None):
        x = self.conv_head(inputs)
        x = self.norm(x)
        x = self.act(x)

        # 应用CBAM注意力
        if self.use_cbam:
            x = self.cbam(x)

        x = self.avg(x)

        if hasattr(self, 'dropout') and self.drop_rate > 0:
            x = self.dropout(x, training=training)

        x = self.fc(x)
        return x


class EfficientNetV2(Model):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 use_cbam: bool = True,  # 新增CBAM参数
                 name: str = None):
        super(EfficientNetV2, self).__init__(name=name)

        for cnf in model_cnf:
            assert len(cnf) == 8

        stem_filter_num = model_cnf[0][4]
        self.stem = Stem(stem_filter_num)

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        self.blocks = []
        # Builds blocks.
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                # 在后几个阶段使用CBAM
                stage_cbam = use_cbam and block_id >= total_blocks * 0.7  # 在后50%的块中使用CBAM
                self.blocks.append(op(kernel_size=cnf[1],
                                      input_c=cnf[4] if i == 0 else cnf[5],
                                      out_c=cnf[5],
                                      expand_ratio=cnf[3],
                                      stride=cnf[2] if i == 0 else 1,
                                      se_ratio=cnf[-1],
                                      drop_rate=drop_connect_rate * block_id / total_blocks,
                                      use_cbam=stage_cbam,
                                      name="blocks_{}".format(block_id)))
                block_id += 1

        self.head = Head(num_features, num_classes, dropout_rate, use_cbam=use_cbam, name="head")

    def call(self, inputs, training=None):
        x = self.stem(inputs, training)

        # call for blocks.
        for _, block in enumerate(self.blocks):
            x = block(x, training=training)

        x = self.head(x, training=training)

        return x


def efficientnetv2_s(num_classes: int = 1000, use_cbam: bool = True):
    """
    EfficientNetV2 with CBAM attention
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model_name = "efficientnetv2-s_cbam" if use_cbam else "efficientnetv2-s"
    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2,
                           use_cbam=use_cbam,
                           name=model_name)
    return model


def efficientnetv2_m(num_classes: int = 1000, use_cbam: bool = True):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model_name = "efficientnetv2-m_cbam" if use_cbam else "efficientnetv2-m"
    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3,
                           use_cbam=use_cbam,
                           name=model_name)
    return model


