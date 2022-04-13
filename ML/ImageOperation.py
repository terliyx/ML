import numpy as np
import matplotlib.pyplot as plt
class ImageOperation(object):

    def plot(self, data):
        for k in range(data.shape[0]):
            img = []
            for i in range(32):
                for j in range(32):
                    pix = []
                    pix.append(data[k, i*32+j])
                    pix.append(data[k, 1024+i*32+j])
                    pix.append(data[k, 2048+i*32+j])
                    img.append(pix)
            plt.imshow(np.uint8(img))
            plt.show()
    def plot2(self, data, labels):
        X = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        Y = np.array(labels)

        fig, axes1 = plt.subplots(3, 3, figsize=(3, 3))
        for j in range(3):
            for k in range(3):

                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[k + 3 * j: k + 3 * j + 1][0])
        plt.show()