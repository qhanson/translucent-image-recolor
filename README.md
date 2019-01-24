## Usage

This repository contains the code in paper "Translucent Image Recoloring through Homography Estimation".

***The core idea is to apply the Homography Estimation to estimate the transformation between the dominant colors and the target colors.***

The example directory shows several recoloring results.
```
refine_code/GUI_hm_dark_style.py is the main file.

steps:
$ conda create -n recolor python=3.5
$ conda  activate recolor
$ pip install -r requirements.txt
$ cd xx/refine_code
$ python GUI_hm_dark_style.py


```
Then, 
File->Open : first open one image that you wish to recolor.
Image-> Fine Dom Colors  : find the dominant colors of the image.
Image->Change Color by Hue: Specify the target colors. The Image view will be automatically updated.


Notice: we are still working on organizing the full code. We will update the codes in this repository soon.
We are also working on a CNN-based method to achieve color transfer. 



## Example results.


![Alt text](https://github.com/qhanson/translucent-image-recolor/blob/master/example-images/examples.png?raw=true "例子")