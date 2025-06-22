# vggt-blender
Blender addon for vggt 3D reconstruction

Input an image folder which contains single or multiple images, then you will get point could geometry nodes with material.

This blender addon is based on [vggt](https://github.com/facebookresearch/vggt).

## Usage
1. Download vggt model from operation panel.
2. select an image folder.
3. Generate.

https://github.com/user-attachments/assets/7c98bdb6-5f0f-4fb2-a9c5-7a4f7e9e553e

## Installation (only the first time)
1. Download Zip from this github repo.
2. Toggle System Console for installation logs tracking.
3. Install addon in blender preference with "Install from Disk" and select downloaded zip.
4. Wait for python dependencies installation.
5. After addon activated, download vggt model from operation panel.

https://github.com/user-attachments/assets/aa48f2ad-8b68-4186-9d05-7df35d2eb5e8

## Tested on
- Win11
- Blender 4.2
- cuda 12.6

## Notes
The GPU memory usage increases when more images used to generate. Please check official [vggt](https://github.com/facebookresearch/vggt) readme for more information. You can start with sample_images in this repository. "ramen" folder contains one single image, and "dog" folder contains four images.
