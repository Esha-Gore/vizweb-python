# vizweb-python
Vizweb implementation in python

It processes images based on their colorfulness, their quadtree decomposition and their spacetree decomposition. 
_____________


## Environment

- **Python**: 3.11 (tested with 3.11.13)
- **Virtual Environment**: `venv`
- **Dependencies**: pinned in `requirements.txt`

## Set-up

Clone the repo and create a virtual environment

Then install dependencies as follows:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


## Testing Files

There are three different testing files which each test a key metric:

### test_colorfulness_metrics.py

Tests the colorfulness metrics on a select few images in the test_images folder.

### test_quadtree.py

Tests Quadtree on nature, documentation, and webistes images from the images folder. 

### test_xy.py

Test spacetree decomposition on nature, documentation, and webistes images from the images folder. 



