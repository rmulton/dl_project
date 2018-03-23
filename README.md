# dl_project

## Installation
1. Clone this repo **recursively** ```git clone --recursive https://github.com/rmulton/dl_project```
2. [Install Pytorch](http://pytorch.org/)
3. Run ```cd ./dl_project/cocoapi/PythonAPI; python setup.py install; cd ../..``` to install the MSCOCO api
4. Download and unzip MSCOCO 2017 dataset [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [training images](http://images.cocodataset.org/zips/train2017.zip) and [validation images](http://images.cocodataset.org/zips/val2017.zip)

## Launch using jupyter notebook
1. Run ```jupyter notebook mscoco_pose_estimation.ipynb```
2. Change the Data location section variables to give the program the path to the dataset
3. You're all set !

## Launch using python files
Change the Data location section variables in the const.py file
- To launch the training of a new model : python main.py train False
- To launch the training over an existing model : python main.py train True <epochNumber>
- To launch the test over the dataset with a trained model : python main.py test <epochNumber>


## References
### Papers on pose estimation
- G. Papandreou et al., [*Towards Accurate Multi-person Pose Estimation in the Wild*](https://arxiv.org/pdf/1701.01779.pdf)
### Datasets
- [Common Objects in Context](http://cocodataset.org)
