# opencv_particles

An OpenCV program to count and analyze particles or blobs in an image.

## Description
This program was made with the intention of counting the number of small waterdrops on a piece of 
Water-Sensitive Paper (感水試験紙). The color of the paper is originally yellow, but turns dark-blue
when it comes in contact with water/moisture. <br>

It can be applied to other uses of image-based particle analysis by adjusting the parameters accordingly. 

## Features
- Counts the total number of particles within the field of interest.
- Outputs two histograms that categorize particles based on their area and equivalent diameter, respectively.

## Requirements
Latest version of Python and OpenCV

## Usage
1. Create a directory named "images" containing the image files (.jpg) to be analyzed.
2. Execute program: `python3 particle.py [file-name]`
