# Volume Classification for Rectal MR

![overall scheme](/img/overall_scheme.png)

## Topics:
1. 2D vs. 3D vs. Mixtures of 2D and 3D
2. Supplementary loss function
3. Depth aggregation function

## Common parser arguments:
1. *--fusion*\
&rarr; fr2d, fr3d, f2plus1d, fmc2, fmc3, fmc4, fmc5, frmc2, frmc3, frmc4, frmc5 

2. *--folder-name*\
&rarr; folder name to save results

3. *--node*\
&rarr; User must register his/her computer information to config.ini

## 1. 2D vs. 3D vs. Mixtures of 2D and 3D

main_backbone_fw.py

## 2. Supplementary loss function

### + triplet loss
main_triplet_fw.py
### + center loss
main_center_loss_sgd_fw.py

## 3. 2D vs. 3D vs. Mixtures of 2D and 3D

main_vw.py

* Select frame aggregation function by *--aggregation-function*\
&rarr; bilinear, gap, mxp, attention

## Sample images
![VCB](/img/easy_hard.png)
: Examples of T3-stage rectal cancer (a, b) and T2-stage rectal cancer (c, d). The difference between T2-stage and T3-stage rectal cancer is that T2-stage rectal cancer (red line) locates only inside of rectum (white line) whereas T3-stage rectal cancer (red line) invaded outside of rectum (white line). Samples at the right column (b, d) are ambiguous and thus mis-diagnosed samples whereas samples at the left column (a, c) are straightforward.