from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

input_dir = './images'
heights = []
widths = []
for filename in os.listdir(input_dir):
    try:
        im = Image.open(os.path.join(input_dir, filename))
        width, height = im.size
        heights += [height]
        widths += [width]
    except OSError as e:
        continue

plt.scatter(widths, heights)
plt.title(f'Logo maker image sizes, measured in pixels.\nMean width: {np.mean(widths):.2f}. Median width: {np.median(widths):.2f}.\nMean height: {np.mean(heights):.2f}. Median height: {np.median(heights):.2f}.')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()
