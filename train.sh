python train.py \
    --dataset ./data/emotion_aug_train.tfrecord \
    --val_dataset ./data/emotion_val.tfrecord \
    --classes ./data/emotion.names \
    --num_classes 5 \
    --tiny True --mode eager_fit --transfer darknet \
    --batch_size 8 --epochs 160 \
    --weights ./checkpoints/yolov3-tiny.tf \
    