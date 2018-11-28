# 3DMM-fitting

This project is designed to fit a 3DMM to a frontal-face picture profile and a profile picture of the same person simultaneously. This is supposed to lead to a more reliable fitting result than the traditional way in which only one frontal face picture is used, since we acquired additional depth information from the extra profile image.

The 3D model fitting part is mainly based on [eos](https://github.com/patrikhuber/eos), which is a lightweight 3D Morphable Face Model fitting library in modern C++11/14. The 3D morphable model we used is also derived from their project. Please note that if you prefer to do single-view 3D fitting with only one frontal image, and have no wish to play with source code, eos project may be a better choice since C++ have much higher performance in speed than Python.

## Library requirements

* Python3.6
* [OpenCV](http://opencv.org/)
* [Dlib](http://dlib.net/)
* [Numpy](http://www.numpy.org/)
* [toml](https://github.com/uiri/toml)

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

After installing this, some minor updates and conflict solving are also need to be done to ensure all menpo function works properly.

The code for the training of profile detection model and AAM for profile landmark regression can be found at `\test\Menpo-Display.ipynb`.

## Instructions

You can find sample code at `test\fitting_test.py`. In the folder `data\`, two sets of sample images are already given to test the code. These images are from [color FERET database](https://www.nist.gov/itl/iad/image-group/color-feret-database). The facial landmarks are saved as pts files with the same name as the pictures. Please note that the frontal-face landmarks are annotated according to the iBug  but the profile landmarks are annotated in a new way showed as below.

![the landmarks of a profile](https://i.imgur.com/ARFkW5F.jpg)

Not all the landmarks are used in the process of 3D-fitting.

The frontal face image is automatically annotated with Dlib library. You can call the `frontal_face_marker` funtion at `assist\marker.py` to get a pts file contains the landmarks of the frontal face image. The profile image is presently marked manually. You can call the `manual_marker` fuction at `assist\marker.py` to do it.

## Presentation

Run `test\fitting_test.py` with default imput images, you should get a picture discribes the accuracy of the fitting.

![fitting result img](https://github.com/Yinghao-Li/3DMM-fitting/blob/master/test/00029ba010_960521-outcome.jpg
)

This picture will be saved in the `test\` folder, along with the generated 3D model as ply file.

![fitting result 3D](https://github.com/Yinghao-Li/3DMM-fitting/blob/master/test/3D-captured.PNG)

