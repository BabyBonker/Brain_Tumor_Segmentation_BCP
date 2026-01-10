import os
import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


from google.colab import files
uploaded = files.upload()

!tar -xf aidataset.tar

files = os.listdir('.')
print(files)

def normalize_slice(x):
    x = x - np.min(x)
    x = x / (np.max(x) + 1e-8)
    return x
t1    = nib.load("BraTS2021_00495_t1.nii.gz").get_fdata()
t1ce  = nib.load("BraTS2021_00495_t1ce.nii.gz").get_fdata()
t2    = nib.load("BraTS2021_00495_t2.nii.gz").get_fdata()
flair = nib.load("BraTS2021_00495_flair.nii.gz").get_fdata()
seg   = nib.load("BraTS2021_00495_seg.nii.gz").get_fdata()

assert t1.shape == seg.shape
print("Volume shape:", t1.shape)

IMG_SIZE = 128
X = []
Y = []

for i in range(t1.shape[2]):
    mask_slice = seg[:, :, i]

    if np.sum(mask_slice) == 0:
        continue   # skip non-tumor slices

    # Stack modalities
    slice_stack = np.stack([
        normalize_slice(t1[:, :, i]),
        normalize_slice(t1ce[:, :, i]),
        normalize_slice(t2[:, :, i]),
        normalize_slice(flair[:, :, i])
    ], axis=-1)

    slice_stack = tf.image.resize(slice_stack, (IMG_SIZE, IMG_SIZE))
    mask_slice  = tf.image.resize(mask_slice[..., None],
                                  (IMG_SIZE, IMG_SIZE),
                                  method="nearest")

    X.append(slice_stack)
    Y.append(mask_slice)

X = np.array(X, dtype=np.float32)
Y = np.array(Y)


Y[Y == 4] = 3
NUM_CLASSES = 4

Y = tf.keras.utils.to_categorical(
    Y.squeeze(), num_classes=NUM_CLASSES
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


print("X shape:", X.shape)   # (N, 128, 128, 4)
print("Y shape:", Y.shape)   # (N, 128, 128, 4)

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(128,128,4)):
    inputs = layers.Input(input_shape)

    c1, p1 = encoder_block(inputs, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)

    b = conv_block(p3, 256)

    d1 = decoder_block(b, c3, 128)
    d2 = decoder_block(d1, c2, 64)
    d3 = decoder_block(d2, c1, 32)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(d3)
    return models.Model(inputs, outputs)

def multiclass_dice_loss(y_true, y_pred):
    smooth = 1e-6
    loss = 0.0

    for i in range(NUM_CLASSES):
        yt = y_true[..., i]
        yp = y_pred[..., i]

        intersection = tf.reduce_sum(yt * yp)
        dice = (2. * intersection + smooth) / (
            tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth
        )
        loss += 1 - dice

    return loss / NUM_CLASSES

def combined_loss(y_true, y_pred):
    dice = multiclass_dice_loss(y_true, y_pred)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice + ce

model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test)
)

pred = model.predict(X[:1])
print(np.unique(np.argmax(pred, axis=-1)))

idx = 5
pred = model.predict(X[idx:idx+1])
pred = np.squeeze(pred)              # (H, W, C)

pred_mask = np.argmax(pred, axis=-1) # (H, W)

np.unique(pred_mask)

idx = 5



pred = model.predict(X[idx:idx+1])
pred = np.squeeze(pred)
pred_mask = np.argmax(pred, axis=-1)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("T1ce")
plt.imshow(X[idx, :, :, 1], cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(np.argmax(Y[idx], axis=-1), cmap='tab10')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred_mask, cmap='tab10')
plt.axis('off')

plt.show()

