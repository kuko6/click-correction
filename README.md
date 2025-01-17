# Method for Segmentation of Vestibular Schwannomas from Brain MRI Scans
The main idea behind this work was to develop a segmentation method which would require as little fully-annotated images as possible for training.

 Our approach is based on two networks, where the primary segmentation network generates initial segmentations, which we later improve by utilising user-defined clicks.

 <p align="center">
  <img src="./docs/imgs/correction_pipeline.png" alt="diagram" style="width:60%;"/>
  <br/>
  <i>Correction method diagram</i>
</p>

The correction is based on an auxiliary correction network, which refines the initial (imperfect) segmentations based on the provided clicks. In general, the correction is done on a local level, where the clicks denote areas of the initial segmentations that need to be improved. 

The architecture of the correction network is based on the U-Net architecture with separate encoders for each of the different modalities (cuts from the binary segmentation masks and from the MRI sequences).

 <p align="center">
  <img src="./docs/imgs/updated_multimodal_v2.png" alt="diagram" style="width:60%;"/>
  <br/>
  <i>Architecture of the correction network</i>
</p>

We also designed a custom loss function. The loss fuction is based on the Dice loss but adds an additional weighting factor which gives higher weights to the areas denoted by the clicks.

$$
L_{corr} = 1 - 2 \frac{\sum\limits_{i=1}^{W} \sum\limits_{j=1}^{H} (p_{i,j} \cdot y_{i,j} \cdot w_{i,j})}{\sum\limits_{i=1}^{W} \sum\limits_{j=1}^{H} ((p_{i,j} + y_{i,j}) \cdot w_{i,j})}
$$

<p align="center"><i>Correction loss</i></p>

With this method, we were able to achieve mean dice score of 0.883 on the testing set by utilising only 20% of the fully-annotated samples. In contrast, state-of-the-art fully supervised approaches achieve dice score around 0.920.

<p align="center">
  <img src="./docs/imgs/reconstructed_seg.png" alt="example" style="width:60%;"/>
  <br/>
  <i>Sample results</i>
</p>

## Setup
The training and evaluation scripts were built for the Azure ML services, so the workflow outside the Azure services might feel a little sluggish and was not properly tested. 

We however prepared a demo jupyter notebook, `demo.ipynb`, which demonstrates our pipeline.

### Try it locally
set up a virtual environment with:
```sh
python3 -m venv venv
```

activate the virtual environment:
```sh
source venv/bin/activate
```

install the requirements:
```sh
pip install -r requirements.txt
```
