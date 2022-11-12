import typer
from mrcnn.config import Config
from mrcnn.utils import compute_ap
from numpy import expand_dims, mean
from mrcnn.model import MaskRCNN, load_image_gt, mold_image

from dataset.gun_dataset import GunDataset, GunConfig

app = typer.Typer()


class PredictionConfig(Config):
    NAME = "ads_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                 r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    mAP = mean(APs)
    return mAP


def get_datasets():
    # train set
    train_set = GunDataset()
    train_set.load_dataset('data', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # test/val set
    test_set = GunDataset()
    test_set.load_dataset('data', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    return train_set, test_set


def prepare_config():
    # prepare config
    config = GunConfig()
    config.display()
    return config


@app.command()
def model_mrcnn(epochs: int = 5):
    MODEL_NAME = f'mask_rcnn_gun_cfg_{epochs}.h5'
    train_set, test_set = get_datasets()
    config = prepare_config()
    # define the model
    model = MaskRCNN(mode='training', model_dir='models/', config=config)
    # train weights (output layers or 'heads')
    model.train(train_set,
                test_set,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='all')
    # save model
    model.keras_model.save_weights(f"models/{MODEL_NAME}")


@app.command()
def model_transfer_learning(epochs: int = 5):
    MODEL_NAME = f'mask_rcnn_coco_gun_transfer_learning_{epochs}.h5'
    train_set, test_set = get_datasets()
    config = prepare_config()
    # define the model
    model = MaskRCNN(mode='training', model_dir='models/', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('models/mask_rcnn_coco.h5',
                       by_name=True,
                       exclude=[
                           "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                           "mrcnn_mask"
                       ])
    # train weights (output layers or 'heads')
    model.train(train_set,
                test_set,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')
    # save model
    model.keras_model.save_weights(f"models/{MODEL_NAME}")


if __name__ == '__main__':
    app()