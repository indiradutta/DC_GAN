from dcgan import DCGAN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML

# Initializing the DCGAN module with celeba dataset
dc_gan = DCGAN(data = '/content/dcgan/celeba')

# Training for num_epoch times
img_list, G_losses, D_losses = dc_gan.train('model.pth')

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
