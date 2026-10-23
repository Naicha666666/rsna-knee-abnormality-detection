<!-- Cell 0 -->
# RSNA Knee: Raptor student blend

Public LB 0.944.

This is the public RSNA_Versia_5 pipeline (0.943) with one extra model blended in at the end.

The extra model is a CoAtNet-RMLP-2, the Raptor architecture, fine-tuned from the public Fine Spacing Raptor checkpoint. It reads 80 slices per study over 2 to 98 percent of each series, cropped to 140 mm at 336 px, and scores every window. On its own it gets 0.942.

The last two cells rank the pipeline output and the extra models output per finding and blend them 0.55 and 0.45.

## Credits

It builds on work by prvsiyan, renta0426, Mattia Angeli, maverick_ss_26, Jiwei Liu, Pilkwang, Sofia Anjenje, Antoine G., Marwan Mahmoud and Anvith Pothula. Dread Development made the Raptor weights, the Fine Spacing geometry and the soft labels.

<!-- Cell 1 -->
## Setup and asset lookup

<!-- Cell 3 -->
## Depth sampling

<!-- Cell 5 -->
## Startup checks

<!-- Cell 7 -->
## Paths and A5 applicability

<!-- Cell 9 -->
## DICOM headers

<!-- Cell 11 -->
## Series selection

<!-- Cell 13 -->
## Slice order and reading

<!-- Cell 15 -->
## Laterality

<!-- Cell 17 -->
## Image cache

<!-- Cell 19 -->
## SlotHead

<!-- Cell 21 -->
## DINOv2 model

<!-- Cell 23 -->
## Loading DINOv2

<!-- Cell 25 -->
## DINOv2 ensemble, 20 members

<!-- Cell 27 -->
## Writing and checking the submission

<!-- Cell 29 -->
## Run DINOv2

<!-- Cell 31 -->
## A5 setup

<!-- Cell 33 -->
## A5 study preparation

<!-- Cell 35 -->
## A5 model and folds

<!-- Cell 37 -->
## A5 inference

<!-- Cell 39 -->
## A5 blend with DINOv2

<!-- Cell 41 -->
## RadImageNet

<!-- Cell 43 -->
## Raptor and the CoAt family

<!-- Cell 45 -->
## Publish the pipeline submission

<!-- Cell 47 -->
## Extra model

Runs after the pipeline when its dataset is attached and there is time left. Otherwise the pipeline file stays as it is.

