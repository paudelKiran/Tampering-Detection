import os
import pandas as pd
import tensorflow as tf

IMG_SIZE = (512, 512)
AUTOTUNE = tf.data.AUTOTUNE


def _fix_path(path, project_root):
    
    if not path:
        return path
    arranged_dir = os.path.join(project_root, "dataset_arranged")
  
    if arranged_dir in path:
        return path
    
    return path.replace(project_root, arranged_dir, 1)


def load_manifest(csv_path, project_root=None):

    df = pd.read_csv(csv_path)
    image_paths = df["image_path"].tolist()
    mask_paths = df["mask_path"].fillna("").tolist()
    labels = df["label"].astype(int).tolist() if "label" in df.columns else [0] * len(image_paths)
    if project_root:
        image_paths = [_fix_path(p, project_root) for p in image_paths]
        mask_paths = [_fix_path(p, project_root) for p in mask_paths]
    return image_paths, mask_paths, labels


def parse_image_mask(image_path, mask_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0


    def _load_mask():
        raw = tf.io.read_file(mask_path)
        m = tf.image.decode_png(raw, channels=1)
        m = tf.image.resize(m, IMG_SIZE, method="nearest")
        m = tf.cast(m > 127, tf.float32)
        return m

    def _zeros_mask():
        return tf.zeros((*IMG_SIZE, 1), dtype=tf.float32)

    has_mask = tf.not_equal(tf.strings.length(mask_path), 0)
    mask = tf.cond(has_mask, _load_mask, _zeros_mask)
    mask.set_shape([IMG_SIZE[0], IMG_SIZE[1], 1])

    return image, mask


def parse_image_mask_label(image_path, mask_path, label):
    image, mask = parse_image_mask(image_path, mask_path)
    label = tf.cast(tf.reshape(label, (1,)), tf.float32)
    return image, {'segmentation': mask, 'classification': label}


def build_dataset(image_paths, mask_paths, labels, batch_size, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.map(parse_image_mask_label, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_datasets(manifest_dir="manifests", batch_size=8, project_root=None):

    train_imgs, train_masks, train_labels = load_manifest(
        os.path.join(manifest_dir, "train_manifest.csv"), project_root
    )
    val_imgs, val_masks, val_labels = load_manifest(
        os.path.join(manifest_dir, "val_manifest.csv"), project_root
    )
    test_imgs, test_masks, test_labels = load_manifest(
        os.path.join(manifest_dir, "test_manifest.csv"), project_root
    )

    train_dataset = build_dataset(train_imgs, train_masks, train_labels, batch_size, shuffle=True)
    val_dataset = build_dataset(val_imgs, val_masks, val_labels, batch_size, shuffle=False)
    test_dataset = build_dataset(test_imgs, test_masks, test_labels, batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()

    for images, targets in train_ds.take(1):
        print(f"Image batch shape:  {images.shape}")
        print(f"Mask batch shape:   {targets['segmentation'].shape}")
        print(f"Label batch shape:  {targets['classification'].shape}")
        print(f"Image range:        [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
