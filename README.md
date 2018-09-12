# 3DMM-fitting

This project is designed to fit a 3DMM to a frontal-face picture profile and a profile picture of the same person simultaneously. This is supposed to lead to a more reliable fitting result than the traditional way in which only one 2D picture is used, since we acquired additional depth information from the extra profile image.

This program is mainly based on [eos](https://github.com/patrikhuber/eos), which is a lightweight 3D Morphable Face Model fitting library in modern C++11/14. The 3D morphable model we used is also derived from their project. Please note that if you prefer to do single-view 3D fitting with only one frontal image, and have no wish to play with source code, eos project may be a better choice since C++ have much higher performance in speed than Python.

## Library requirements

* Python3.6
* [OpenCV](http://opencv.org/)
* [Dlib](http://dlib.net/)
* [Numpy](http://www.numpy.org/)
* [toml](https://github.com/uiri/toml)

The code has only been tested on Windows10 with Anaconda Python.

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

