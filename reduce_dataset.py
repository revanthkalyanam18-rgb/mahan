import os
import random

limit = 1000

folders = ["cell_images/Parasitized", "cell_images/Uninfected"]

for folder in folders:
    images = os.listdir(folder)

    if len(images) > limit:
        remove = random.sample(images, len(images) - limit)

        for img in remove:
            os.remove(os.path.join(folder, img))

print("Dataset reduced successfully!")