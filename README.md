# Volume Classification for Rectal MR
## Train
python main_volume_wise.py --train

1. Select fusion type by --fusion\
&rarr; fr2d, fr3d, f2plus1d, fmc2, fmc3, fmc4, fmc5, frmc2, frmc3, frmc4, frmc5 

2. Select frame aggregation function by --aggregation-function\
&rarr; bilinear, gap, mxp, attention


## Test
python main_volume_wise.py

## ETC
1. Currently, utils/loss.py do support focal loss and center loss, however, triplet loss is included in the model. Therefore, running the code will automatically utilize triplet loss by default. We will decouple the triplet loss from the model class and make it as an option shortly.

2. Currently, though frame-wise encoder is available by model/video_resnet_triplet_frame_wise.py, we will shortly decouple the triplet loss as described above and make it available at the main code.