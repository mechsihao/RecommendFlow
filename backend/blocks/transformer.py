import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import MultiHeadAttention, Add, LayerNormalization, Flatten
from backend.blocks.mlp import create_mlp


def create_tab_transformer(num_transformer_blocks, num_heads, embedding_dims, dropout_rate, name):
    encoded_categorical_features = Input((None, ))
    for block_idx in range(num_transformer_blocks):
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"{name}_multi_head_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        x = Add(name=f"{name}_skip_connection1_{block_idx}")([attention_output, encoded_categorical_features])
        x = LayerNormalization(name=f"{name}_layer_norm1_{block_idx}", epsilon=1e-6)(x)
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims], dropout_rate=dropout_rate, activation=keras.activations.selu,
            normalization_layer=LayerNormalization(epsilon=1e-6), name=f"{name}_feedforward_{block_idx}",
        )(x)
        x = Add(name=f"{name}_skip_connection2_{block_idx}")([feedforward_output, x])
        encoded_categorical_features = LayerNormalization(name=f"{name}_layer_norm2_{block_idx}", epsilon=1e-6)(x)
    out_categorical_features = Flatten()(encoded_categorical_features)
    return tf.keras.models.Model(encoded_categorical_features, out_categorical_features, name=name)
