# Correcting segmentations based on user clicks
Method for segmentation of Vestibular Schwannomas from brain MRI scans. The method utilises an auxilary correction network, which based on user-defined clicks, refines the generated initial segmentation. 

<!-- <img src="./docs/imgs/full_pipeline.png" alt="diagram" style="width:80%"/> -->
<!-- ![diagram](./docs/imgs/full_pipeline.png)

*method diagram*

$$
L_{corr} = 1 - 2 \frac{\sum_{i=1}^{W} \sum_{j=1}^{H} (p_{i,j} \cdot y_{i,j} \cdot w_{i,j})}{\sum_{i=1}^{W} \sum_{j=1}^{H} ((p_{i,j} + y_{i,j}) \cdot w_{i,j})}
$$ -->

<!-- <img src="./docs/imgs/reconstructed_seg.png" alt="example" style="width:80%"/> -->
![example](./docs/imgs/reconstructed_seg.png)

*sample results*

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