import os
from tensorflow.keras.callbacks import Callback


class ModelCheckpoint(Callback):

    def __init__(self, model_save_root, model_name):
        super().__init__()
        self.model_save_root = model_save_root
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        i = epoch + 1
        save_path = os.path.join(self.model_save_root, f"epoch{i}", self.model_name)
        self.model.save_weights(save_path)
