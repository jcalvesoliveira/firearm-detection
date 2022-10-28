import os
import random
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from mrcnn.utils import Dataset

LABES_MAP = {'firearm': 1}


class AdvertisementDataset(Dataset):

    def image_path(self, dataset_dir: str, image_id: str) -> str:
        return dataset_dir + '/images/' + str(image_id) + '.jpg'

    def annotation_path(self, dataset_dir: str, image_id: str) -> str:
        return dataset_dir + '/annotations/' + str(image_id) + '.csv'

    def load_dataset(self, dataset_dir, is_train=True, train_size=0.8):
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        image_files = os.listdir(images_dir)
        # shuffle images to guarantte random order when splitting into train and test
        random.shuffle(image_files)
        images_count = len(image_files)
        train_threshold = int(images_count * train_size)
        logging.info(f"Total images: {images_count}")
        for label in LABES_MAP:
            self.add_class("dataset", LABES_MAP[label], label)
        for i, filename in enumerate(os.listdir(images_dir)):
            if is_train and i >= train_threshold:
                continue
            if not is_train and i < train_threshold:
                continue
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + str(image_id) + '.csv'
            self.add_image('dataset',
                           image_id=image_id,
                           path=img_path,
                           annotation=ann_path)

    def extract_boxes(self, dataset_dir: str,
                      image_id: str) -> Tuple[pd.DataFrame, int, int]:
        image_path = self.image_path(dataset_dir, image_id)
        annotation_path = self.annotation_path(dataset_dir, image_id)
        boxes = pd.read_csv(annotation_path)
        img = Image.open(image_path)
        width, height = img.size
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path, image_id)
        annotations_count = boxes.shape[0]
        masks = np.zeros([h, w, annotations_count], dtype='uint8')
        class_ids = list()
        for i, box in enumerate(boxes.values):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = LABES_MAP[box[4]]
            class_ids.append(self.class_names.index(box[4]))
        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


if __name__ == '__main__':
    train_dataset = AdvertisementDataset()
    train_dataset.load_dataset('data', is_train=True)
    train_dataset.prepare()
    print('Train: %d' % len(train_dataset.image_ids))
    test_dataset = AdvertisementDataset()
    test_dataset.load_dataset('data', is_train=False)
    test_dataset.prepare()
    print('Test: %d' % len(test_dataset.image_ids))