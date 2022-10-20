# Notes

- Batch Size greatly speeds up convergence
- Learning Rate has *some* effect on reducing jitter during training
- Adding more 3x3 2DConv seems to consistently improve the network
- Adding weight decay regularization for Adam reduces training overfitting
- Adding BatchNorm significantly improves performance
- ReLU performs significantly better than Sigmoid
- Having a tiny Fully Connected Layer at the end improves training stability

## Changes

- v11 uses a tiny FCN at the end
- v11 -> v12 uses a 2048 -> DropOut -> 1024 -> DropOut -> 2 instead of 2048 -> 2
- v11 -> v13 uses a AvgPool at the end (mimicking resnet)
- v11 -> v14 uses LeakyReLU (Not much difference)
- v11 -> v15 uses increase image size slightly, cropping out the middle
- v15 -> v16 move crop to end of transform pipeline
- v20 AvePoolModel
- v26 run without aug
- v27 run with aug