# Simple file to show basics of optics

import matplotlib.pyplot as plt
from helpfulFunctions import * 

path = 'peppers.png'
img = loadImage(path)

plt.imshow(img)
plt.show()



