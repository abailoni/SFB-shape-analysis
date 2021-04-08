# SFB-shape-analysis
### Installation
- Clone the repository: `git clone https://github.com/abailoni/SFB-shape-analysis.git`
- Move to the package directory: `cd SFB-shape-analysis`
- To install the dependencies, you will need [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- Once you have installed conda, run the command  `conda env create --name=shapeAnalysis --file=environment.yml`
- Before to run any of the scripts, activate your new environment with `conda activate shapeAnalysis`
- Install package with `python setup.py install`  


### How to use the scripts
Scripts can be found in the `scripts` folder. Most of the scripts require two arguments: 

- `DATA_DIR`: folder including original `.tif` images (can be organized in subfolders)
- `PROJECT_DIR`: results will be saved into subfolders of this directory (for example in `preprocessed`, `segmentations`, etc...)

The script to run the ilastik project will also require the arguments `ilastik_path` and `ilastik_project_path`. For example, the command should look something like this:

`python scripts/2_run_ilastik_segmentation.py --DATA_DIR=<your-path> --PROJECT_DIR=<your-path> --ilastik_path=<path-to-run_ilastik.sh> --ilastik_project_path=<path-to-ilastik-project.ilp>`


### Training ilastik classifier
Some useful [ilastik](https://www.ilastik.org/download.html) documentation material:

- [Train a pixel classifier with ilastik](https://www.ilastik.org/documentation/pixelclassification/pixelclassification):
  - Probably you want to create one ilastik project for each type of images / microscope / focal-length you want to process
  - When adding images to the ilastik project, remember to select `tyx` as input format, so the images are interpreted as videos and not 3D volumes
  - After you are done with training and painting the labels, in the training step, select the `Suggest Features` option to find the most informative features and possibly make the pipeline more efficient 
- [Use ilastik from command line](https://www.ilastik.org/documentation/basics/headless) (see example of usage in `scripts/2_run_ilastik_segmentation.py`)
