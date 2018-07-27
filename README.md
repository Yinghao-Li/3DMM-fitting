# 3DMM-fitting

This project is designed to fit a 3DMM to a frontal-face picture profile and a profile picture of the same person simultaneously. This is supposed to lead to a more reliable fitting result than the traditional way in which only one 2D picture is used, since we acquired additional depth information from the extra profile image.

## Library requirements

* Python3.6
* [OpenCV](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [toml](https://github.com/uiri/toml)

The code has only been tested on Windows10 with Anaconda Python.

## Sample Code

You can find sample code at ```test\fitting_test.py```. In the folder ```data\```, two sets of sample images are already given to test the code. These images are from [color FERET database](https://www.nist.gov/itl/iad/image-group/color-feret-database). The facial landmarks are saved as pts files with the same name as the pictures. Please note that the frontal-face landmarks are annotated according to the iBug  but the profile landmarks are annotated with a new order showed as below. 
