# dl_project

## Installation
1. Clone this repo **recursively** ```git clone --recursive https://github.com/rmulton/dl_project```
2. [Install Pytorch](http://pytorch.org/)
3. Run ```cd cocoapi/PythonAPI; python setup.py install; cd ../..``` to install the MSCOCO api
4. Download and unzip MSCOCO 2017 dataset [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [training images](http://images.cocodataset.org/zips/train2017.zip) and [validation images](http://images.cocodataset.org/zips/val2017.zip)
5. Run ```jupyter notebook mscoco_pose_estimation.ipynb```
6. Change the Data location section variables to give the program the path to the dataset
7. You're all set !
