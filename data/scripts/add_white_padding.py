#!/usr/bin/env python

from PIL import Image
import math
import os


def resize_canvas(old_image_path, new_image_path,
                  canvas_width=500, canvas_height=500):
    """
    Place one image on another image.

    Resize the canvas of old_image_path and store the new image in
    new_image_path. Center the image on the new canvas.
    """
    im = Image.open(old_image_path)
    old_width, old_height = im.size
    if old_width == old_height:
        return
    canvas_width = max(old_width, old_height)
    canvas_height = max(old_width, old_height)

    # Center the image
    x1 = int(math.floor((canvas_width - old_width) / 2))
    y1 = int(math.floor((canvas_height - old_height) / 2))

    mode = im.mode
    if len(mode) == 1:  # L, 1
        new_background = (255)
    if len(mode) == 3:  # RGB
        new_background = (255, 255, 255)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (255, 255, 255, 255)

    newImage = Image.new(mode, (canvas_width, canvas_height), new_background)
    newImage.paste(im, (x1, y1, x1 + old_width, y1 + old_height))
    newImage.save(new_image_path)
    im.close()
if __name__ == '__main__':
    l = ['demo', 'demo2']
    for i in l:
        files = os.listdir(i)
        for j in files:
            if j.endswith(".jpg"):
                inpath = os.path.join(i, j)
                outpath = os.path.join(i, j)
                resize_canvas(old_image_path=inpath, new_image_path=outpath)
        print '%s Finished' % i
