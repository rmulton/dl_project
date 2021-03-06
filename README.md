# dl_project

## Installation
1. Clone this repo **recursively** ```git clone --recursive https://github.com/rmulton/dl_project```
2. [Install Pytorch](http://pytorch.org/)
3. Run ```cd ./dl_project/cocoapi/PythonAPI; python setup.py install; cd ../..``` to install the MSCOCO api
4. Download and unzip MSCOCO 2017 dataset [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [training images](http://images.cocodataset.org/zips/train2017.zip) and [validation images](http://images.cocodataset.org/zips/val2017.zip)

## Composition of the reposetory
- configuration.py : the parameters of the model
- const.py : data location constants
- dataset.py : construction of the dataset used for the training and testing of the model
- datasets_preparation : computer vision algorithm applied to the dataset
- heatmap.py : generation of heatmap from keypoints coordinates
- model.py : deeplearning model configuration
- run.py : the training and testing iterations
- main.py 

## Launch using jupyter notebook
1. Run ```jupyter notebook mscoco_pose_estimation.ipynb```
2. Change the Data location section variables to give the program the path to the dataset
3. You're all set !

## Launch using python files
Change the Data location section variables in the const.py file
- To launch the training of a new model : python main.py train False <inputType>
- To launch the training over an existing model : python main.py train True <inputType> <epochNumber>
- To launch the test over the dataset with a trained model : python main.py test <epochNumber> <inputType>
## InputType:
O : original image
1 : original image + skin filtered 
2 : original image + edge filter 
3 : original image + clustering filter 
4 : orignal image + skin filter + edge filter
5 : orignal image + skin filter + clustering filter


## References
### Papers on pose estimation
- G. Papandreou et al., [*Towards Accurate Multi-person Pose Estimation in the Wild*](https://arxiv.org/pdf/1701.01779.pdf)
### Datasets
- [Common Objects in Context](http://cocodataset.org)
