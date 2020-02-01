# 3DMM-fitting

This project is designed to fit a 3DMM to a frontal-face picture profile and a profile picture of the same person simultaneously. This is supposed to lead to a more reliable fitting result than the traditional way in which only one frontal face picture is used, since we acquired additional depth information from the extra profile image.

To add more "automation" flavour to the project, We also introduced landmark regression technique to generated landmarks used for 3DMM-fitting. It should be noticed that the frontal face landmark detection technique is quite mature, so we directly used Dlib-Python to realize the function. However, the profile landmark detection has not been introduced as frequently, and there is no available annotated profile database on Internet. After annotating a subset of profile faces of FERET by ourselves, we compared some techniques such as CNN and AAM, and found out AAM gave the best performance on the limited training set. So we eventually chose to use Dlib-Python to do the frontal face landmark regression, profile face bounding box location model generating and profile face detection; and to use AAM provided by menpo project to do the profile landmark regression.
<font color=red>However, as the training set is limited, this automatic annotation approach can only be used on profile pictures in FERET dataset. For other profile images, we provided manually marking tools to enable you to annotate them by hand.</font>

The 3D model fitting part is mainly based on [eos](https://github.com/patrikhuber/eos), a lightweight 3D Morphable Face Model fitting library in modern C++11/14. The 3D morphable model we used is also derived from their project, although we did some modification so that the model can be readily read by Python projects. Please note that if you prefer to do single-view 3D fitting with only one frontal image, comfortable with configurating C++ libraries and have no wish to play with source code, eos project might be a better choice since C++ have much higher performance in speed than Python.

## Library requirements

* Python3.5
* [OpenCV](http://opencv.org/)
* [Dlib](http://dlib.net/)
* [Numpy](http://www.numpy.org/)
* [toml](https://github.com/uiri/toml)
* [menpo](https://www.menpo.org/)
* [OpenGL](http://pyopengl.sourceforge.net/) (Optional)

## Face detection and landmark regression

For frontal face detection and landmark regression, please refer to Dlib. Usage example can be found at `assist\marker.py`.

The training of profile detector and profile landmark AAM model is based on [Menpo Project](https://www.menpo.org/). The installation of menpo lib can be found at their webpage. As the project has not been updated for a long time, some of it's library dependency is samewhat out-of-date and maybe conflict with current Python libraries. It is recommended to install their lib in a new conda environment with python 3.5, in case the already installed libs get messed up.

```
$ conda create -n menpo python=3.5
$ <conda> activate menpo

$ conda install -c menpo menpo
$ conda install -c menpo menpofit
$ conda install -c menpo menpodetect
$ conda install -c menpo menpowidgets
```
other menpo libs are not used in this project

After installing this, some minor updates and conflict solving are also need to be done to ensure all menpo function works properly. Specifically, `jupyter notebook` should be updated, and some dependencies of matplotlib such as ipywidgets must be downgraded to show widgets in jupyter notebook properly. If you encounter any problem, please consult Google or raise an issue at GitHub Repository linked at the end of this document.

The code for the training of profile detection model and AAM for profile landmark regression can be found at `\test\Menpo-Display.ipynb`.

## 3-D Morphable Model Fitting

You are recommended to go through the fitting procedure with `test\Display.ipynb`. In the folder `data\`, two sets of sample images are already given to test the code. These images are from [color FERET database](https://www.nist.gov/itl/iad/image-group/color-feret-database). The facial landmarks are saved as pts files with the same name as the pictures. Please note that the frontal-face landmarks are annotated according to the iBug 68 standard but the profile landmarks are annotated in a new way showed as below.

![the landmarks of a profile](https://i.imgur.com/ARFkW5F.jpg)

Not all the landmarks are used in the process of 3D-fitting.

The frontal face image is automatically annotated with Dlib library. You can call the `frontal_face_marker` funtion at `assist\marker.py` to get a pts file contains the landmarks of the frontal face image. The profile image can be marked automatically or manually according to the image source. You can call the `manual_marker` fuction at `assist\marker.py` to do manual mark, as shown in `test\Display.ipynb`.

## Search

`test\search.ipynb` is a trivial demo of searching a particular contend in all files with a particular suffix in all the subfolders of a particular path. This function is mainly to make my research procedure easier and has no special use for this project. But as I kept it I feel obliged to introduce its function. And so it is.

## `.py` in `test\` folder
They are legacies and not used for the demonstration for this project so I do not guarantee their functionality. I strongly suggest you not to waste your time on them.

## Demonstration

Run `test\fitting_test.py` with default imput images, you should get a picture discribes the accuracy of the fitting.

 <img src="https://github.com/Yinghao-Li/3DMM-fitting/blob/master/test/00029ba010_960521-outcome.jpg" width = "500" height = "400" alt="fitting result img" align=center />

This picture will be saved in the `test\` folder, along with the generated 3D model as ply file.

 <img src="https://github.com/Yinghao-Li/3DMM-fitting/blob/master/test/3D-captured.PNG" width = "500" height = "300" alt="fitting result 3D" align=center />

[Video Demo Link](https://youtu.be/U2EfZidSws8)


