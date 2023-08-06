from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from model import create_model
from data import create_dataset

# Creating model
model = create_model()

# Compiling model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=BinaryCrossentropy(from_logits=False),
              metrics=['accuracy',
              Precision(),
              Recall(),
              AUC(from_logits=False)])

# Creating callbacks
logdir = '../tb_logs/model11'
tensorboard_callback = TensorBoard(log_dir=logdir)

checkpoint_filepath = '../model_checkpoint/model11'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

train_data, val_data, batch = create_dataset()

# Model training
istorija = model.fit(train_data,
                     epochs=50,
                     batch_size=batch,
                     validation_data=val_data,
                     steps_per_epoch=len(train_data),
                     validation_steps=len(val_data),
                     callbacks=[tensorboard_callback, model_checkpoint_callback])
