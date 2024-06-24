The training and evaluation scripts were built for the Azure ML services, so the workflow outside the Azure services might feel a little sluggish and was not properly tested. 

We however prepared a demo jupyter notebook, `demo.ipynb`, which demonstrates our pipeline.

## Setup
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