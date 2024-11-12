"""
Helper functions.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import math
import time
import random

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

RANDOM_SEED = 0


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)

def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data[0].transpose(1, 0).to(device)
        target = data[1].transpose(1, 0).to(device)

        if model.__class__.__name__ == 'FullTransformerTranslator':
            translation = model(source, target)
        else:
            translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].transpose(1, 0).to(device)
            target = data[1].transpose(1, 0).to(device)

            if model.__class__.__name__ == 'FullTransformerTranslator':
                translation = model(source, target)
            else:
                translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def plot_curves(train_perplexity_history, valid_perplexity_history, filename):
    '''
    Plot learning curves with matplotlib. Training perplexity and validation perplexity are plot in the same figure
    :param train_perplexity_history: training perplexity history of epochs
    :param valid_perplexity_history: validation perplexity history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    epochs = range(len(train_perplexity_history))
    plt.plot(epochs, train_perplexity_history, label='train')
    plt.plot(epochs, valid_perplexity_history, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Perplexity Curve - '+filename)
    plt.savefig(filename+'.png')
    plt.show()


def unit_test_values(testcase):
    if testcase == 'rnn':
        return torch.FloatTensor([[-0.9080, -0.5639, -3.5862],
                                  [-1.2683, -0.4294, -2.6910],
                                  [-1.7300, -0.3964, -1.8972],
                                  [-2.3217, -0.4933, -1.2334]]), torch.FloatTensor([[0.9629,  0.9805, -0.5052,  0.8956],
                                                                                    [0.7796,  0.9508, -
                                                                                        0.2961,  0.6516],
                                                                                    [0.1039,  0.8786, -
                                                                                        0.0543,  0.1066],
                                                                                    [-0.6836,  0.7156,  0.1941, -0.5110]])

    if testcase == 'lstm':
        ht = torch.FloatTensor([[-0.0452,  0.7843, -0.0061,  0.0965],
                                [-0.0206,  0.5646, -0.0246,  0.7761],
                                [-0.0116,  0.3177, -0.0452,  0.9305],
                                [-0.0077,  0.1003,  0.2622,  0.9760]])
        ct = torch.FloatTensor([[-0.2033,  1.2566, -0.0807,  0.1649],
                                [-0.1563,  0.8707, -0.1521,  1.7421],
                                [-0.1158,  0.5195, -0.1344,  2.6109],
                                [-0.0922,  0.1944,  0.4836,  2.8909]])
        return ht, ct

    if testcase == 'encoder':
        expected_out = torch.FloatTensor([[[-0.7773, -0.2031],
                                         [-0.6186, -0.2321]],

                                        [[ 0.0599, -0.0151],
                                         [-0.9237,  0.2675]],

                                        [[ 0.6161,  0.5412],
                                         [ 0.7036,  0.1150]],

                                        [[ 0.6161,  0.5412],
                                         [-0.5587,  0.7384]],

                                        [[-0.9062,  0.2514],
                                         [-0.8684,  0.7312]]])
        expected_hidden = torch.FloatTensor([[[ 0.4912, -0.6078],
                                         [ 0.4932, -0.6244],
                                         [ 0.5109, -0.7493],
                                         [ 0.5116, -0.7534],
                                         [ 0.5072, -0.7265]]])
        return expected_out, expected_hidden

    if testcase == 'decoder':
        expected_out = torch.FloatTensor(
        [[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
         -2.6142, -3.1621],
        [-1.9727, -2.1730, -3.3104, -3.1552, -2.4158, -1.7287, -2.1686, -1.7175,
         -2.6946, -3.2259],
        [-2.1952, -1.7092, -3.1261, -2.9943, -2.5070, -2.1580, -1.9062, -1.9384,
         -2.4951, -3.1813],
        [-2.1961, -1.7070, -3.1257, -2.9950, -2.5085, -2.1600, -1.9053, -1.9388,
         -2.4950, -3.1811],
        [-2.7090, -1.1256, -3.0272, -2.9924, -2.8914, -3.0171, -1.6696, -2.4206,
         -2.3964, -3.2794]])
        expected_hidden = torch.FloatTensor([[
                                            [-0.1854,  0.5561],
                                            [-0.6016,  0.0276],
                                            [ 0.0255,  0.3106],
                                            [ 0.0270,  0.3131],
                                            [ 0.9470,  0.8482]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor(
        [[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
          -2.1898],
         [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
          -2.3045]],

        [[-1.8506, -2.3783, -2.1297, -1.9083, -2.5922, -2.3552, -1.5708,
          -2.2505],
         [-2.0939, -2.1570, -2.0352, -2.2691, -2.1251, -1.8906, -1.8156,
          -2.3654]]]
        )
        return expected_out

    if testcase == 'attention':

        hidden = torch.FloatTensor(
            [[[-0.7232, -0.6048],
              [0.9299, 0.7423],
              [-0.4391, -0.7967],
              [-0.0012, -0.2803],
              [-0.3248, -0.3771]]]
        )

        enc_out = torch.FloatTensor(
            [[[-0.7773, -0.2031],
              [-0.6186, -0.2321]],

             [[0.0599, -0.0151],
              [-0.9237, 0.2675]],

             [[0.6161, 0.5412],
              [0.7036, 0.1150]],

             [[0.6161, 0.5412],
              [-0.5587, 0.7384]],

             [[-0.9062, 0.2514],
              [-0.8684, 0.7312]]]
        )

        expected_attention = torch.FloatTensor(
            [[[0.4902, 0.5098]],

             [[0.7654, 0.2346]],

             [[0.4199, 0.5801]],

             [[0.5329, 0.4671]],

             [[0.6023, 0.3977]]]
        )
        return hidden, enc_out, expected_attention

    if testcase == 'seq2seq_attention':
        expected_out = torch.FloatTensor(
            [[[-2.1755, -2.3331, -2.0947, -1.9011, -1.8674, -2.0639, -2.0731,
               -2.2103],
              [-2.3303, -2.3597, -2.1384, -1.7453, -1.8144, -2.1857, -2.0089,
               -2.2408]],

             [[-2.1715, -2.3639, -2.2964, -1.7742, -1.8789, -1.8759, -2.0575,
               -2.4308],
              [-2.0750, -2.1343, -2.2685, -1.8661, -1.9877, -2.0210, -2.0843,
               -2.2634]]]
        )
        return expected_out

    if testcase == 'full_trans_fwd':
        expected_out = torch.FloatTensor(
            [[[-5.2991e-01, 3.3825e-01, 7.0744e-02, 4.3333e-01, 3.8669e-01],
              [3.8602e-01, 3.1367e-01, 6.7051e-01, 1.1833e+00, 8.6196e-01],
              [-3.1397e-01, 2.0557e-01, -3.0206e-01, 1.1859e-01, 1.4784e+00],
              [-5.5275e-01, 7.3156e-01, -3.5635e-01, -1.0740e+00, 2.4372e-03],
              [-1.2044e-01, 3.0255e-01, -1.5296e-02, -2.2043e-01, 8.8205e-01],
              [-4.6795e-01, 2.8503e-01, -6.0042e-01, -4.7617e-01, 5.5953e-01],
              [3.7261e-01, -6.7359e-02, -8.0002e-01, -4.7465e-01, 3.9449e-01],
              [6.1375e-02, 7.4081e-02, -4.5705e-01, -3.0300e-01, 4.9081e-01],
              [2.5136e-01, -6.7792e-02, 2.5467e-01, -8.8787e-02, 9.7407e-01],
              [-7.1644e-01, 6.8178e-01, 1.2843e-01, 6.9867e-01, 3.9464e-01],
              [-7.4003e-02, -5.3314e-01, -7.7635e-01, -5.4385e-01, 7.3274e-01],
              [-2.2329e-01, -3.8742e-01, -7.2155e-01, -7.0207e-02, 9.3097e-01],
              [2.3047e-01, 4.6325e-01, -5.9637e-01, -7.9165e-01, 6.1274e-01],
              [1.5648e-01, 1.7464e-01, -3.9331e-01, -1.2669e-01, 9.3861e-01],
              [2.7520e-01, 5.8480e-01, -6.4889e-02, -1.5276e-02, 1.3385e+00],
              [2.3830e-01, 1.8516e-01, -7.6924e-01, 1.7116e-01, 5.4277e-01],
              [3.0086e-01, -3.2560e-01, -3.6847e-01, -1.1735e-01, 5.8129e-01],
              [-6.6563e-02, -4.5034e-02, -8.2084e-01, -2.4144e-01, 6.9282e-01],
              [-5.4360e-02, -4.5348e-02, -4.1394e-01, 2.4146e-01, 6.7673e-01],
              [-1.5074e-01, 1.2748e-01, -1.1450e+00, -1.7167e-01, 1.0018e+00],
              [2.8245e-01, 4.8498e-01, -8.9630e-01, -1.0152e-01, 7.8358e-01],
              [-2.1487e-01, -1.5675e-02, -8.3157e-01, -3.6155e-01, 5.2352e-01],
              [7.9770e-02, 9.4337e-02, -7.9253e-01, -4.2432e-01, 9.2304e-01],
              [-1.8966e-02, -5.6669e-01, -8.4327e-01, 9.9881e-02, 6.7961e-01],
              [2.3585e-01, 4.2400e-01, -9.1258e-01, -6.7772e-01, 6.5820e-01],
              [2.7151e-01, 7.2623e-02, -7.7338e-01, -1.0351e+00, 3.7994e-01],
              [-4.9488e-02, 4.0697e-02, -3.7868e-01, 6.4493e-02, 1.4494e+00],
              [-2.9432e-01, 8.3586e-03, -1.1938e-01, -7.0197e-01, 8.1998e-01],
              [9.1792e-02, 1.4808e-01, -1.3459e-02, 6.3131e-01, 4.5028e-01],
              [2.1210e-01, -3.1364e-01, -5.8913e-01, -4.1880e-01, 6.9112e-01],
              [4.2657e-01, 3.4690e-01, -2.8165e-01, -1.2676e-01, 5.0175e-01],
              [-2.2646e-01, -1.6020e-02, -5.6139e-01, 2.6324e-01, 2.9498e-01],
              [-2.2734e-01, 2.3139e-01, -3.6050e-01, -1.6385e-01, 2.1542e-01],
              [1.5429e-01, 1.0473e-01, 2.0145e-01, -5.7639e-01, 7.6994e-01],
              [1.8709e-01, 6.8240e-01, -5.0919e-01, -1.4007e-01, 1.0858e+00],
              [-3.4349e-01, 1.9623e-01, -7.1642e-01, -5.3031e-01, 1.0954e+00],
              [7.8733e-01, -2.2667e-01, -2.1819e-01, -5.0662e-01, 9.4222e-01],
              [3.2126e-01, 9.1827e-01, -5.6426e-01, 2.9004e-01, 7.4178e-01],
              [1.1605e+00, 4.7243e-01, 5.4141e-01, -5.0620e-01, 2.0791e+00],
              [4.7360e-01, -2.3379e-01, -5.2648e-01, -1.8986e-01, 9.2000e-01],
              [-8.2785e-02, 1.8957e-01, 1.1725e-01, 3.8225e-02, 8.6835e-01],
              [-9.8601e-02, 4.0623e-01, -3.0397e-01, 4.0375e-01, 1.5617e+00],
              [6.7749e-01, -3.8384e-02, -8.1424e-02, -6.5261e-01, 2.7836e-01]],

             [[-7.8050e-01, 3.6182e-01, 4.3845e-01, -4.5841e-02, 5.8881e-01],
              [-1.1423e-01, -2.9555e-01, 1.1679e-01, 9.3083e-01, 1.2177e+00],
              [-1.7291e-01, -7.6887e-01, -2.4065e-01, 9.2786e-01, 7.0940e-01],
              [-4.2711e-01, 3.1271e-02, -1.8515e-02, 6.1029e-02, 5.8663e-01],
              [1.7222e-01, -7.0689e-01, -6.2833e-01, 2.5792e-01, 4.2172e-01],
              [1.2222e-01, -3.7569e-01, -1.0024e+00, -4.4555e-01, 5.9147e-01],
              [6.4407e-01, -1.2317e-01, -8.5631e-01, -3.3366e-01, 3.7897e-01],
              [-5.0865e-03, -4.5680e-01, -1.1891e-01, -3.7479e-01, 7.9898e-01],
              [-2.9869e-01, -5.9030e-01, 6.1357e-01, 8.5549e-02, 7.9053e-01],
              [-9.2328e-01, 2.4251e-01, 1.4259e-03, 3.4660e-01, 9.1165e-01],
              [-3.7262e-01, -8.2624e-02, -4.9170e-01, -1.8179e-01, 4.0201e-01],
              [3.9768e-01, -8.6913e-02, 1.2927e-01, -3.7466e-01, 1.0149e+00],
              [-1.2414e-01, 3.9538e-01, -4.7217e-01, -2.6205e-01, 8.3489e-01],
              [3.7820e-02, 2.1549e-01, -1.5087e-02, 3.9300e-01, 5.9774e-01],
              [5.6200e-01, -9.6165e-02, -8.0610e-02, 3.6496e-01, 1.3356e+00],
              [1.2040e-01, -1.1543e-01, -5.7474e-01, -3.9755e-01, 3.4776e-01],
              [4.6390e-01, -4.0411e-01, -7.7353e-01, -1.4525e-01, 5.5663e-01],
              [-4.8031e-01, -3.2004e-01, -1.1133e+00, -7.2538e-01, 3.9708e-01],
              [-9.6895e-03, -5.1319e-01, -2.0693e-01, -6.7379e-02, 7.3828e-01],
              [3.4177e-01, 9.0015e-02, -8.9737e-01, 5.4613e-01, 3.3081e-01],
              [1.0400e-01, -1.7148e-01, -3.2772e-01, 4.2404e-01, 6.3782e-01],
              [-1.5938e-01, -4.1549e-01, -6.5641e-01, 5.4672e-02, 9.8981e-01],
              [2.0610e-01, -7.4291e-01, -3.9612e-01, -2.8068e-01, 5.7160e-01],
              [1.8803e-02, -5.8097e-01, -5.2280e-01, 1.8156e-01, 1.3034e+00],
              [-5.3330e-01, 1.8899e-02, 1.5330e-01, 2.4269e-01, 9.4857e-01],
              [3.3362e-01, -7.8935e-02, -4.7596e-01, -2.5661e-01, 8.3963e-01],
              [1.0205e-01, -5.9235e-01, -9.7127e-02, -1.0917e-01, 8.7554e-01],
              [1.0117e-01, -2.4916e-01, -8.3501e-01, -3.2555e-01, 7.2592e-01],
              [3.9172e-01, -4.0024e-01, 2.3254e-02, 3.3773e-01, 5.8294e-01],
              [-2.2390e-01, -9.3875e-01, -4.8680e-01, 2.0308e-01, 4.5998e-01],
              [-6.9155e-02, 5.4530e-01, -1.4520e-01, 3.5854e-01, 5.2096e-02],
              [5.6979e-02, -2.2058e-01, 9.0090e-02, -1.3128e-01, 4.9931e-01],
              [-5.5760e-01, -4.0400e-01, -9.8166e-01, -1.1645e-01, 2.7747e-01],
              [-4.5377e-01, 3.6731e-01, -1.5656e-01, -3.2344e-01, 1.3428e-01],
              [1.3331e-01, -1.1398e-01, -6.8221e-01, -3.2472e-01, 1.0549e+00],
              [-4.1211e-01, 4.1481e-02, -5.1634e-01, -1.3624e-01, 1.0305e+00],
              [1.0240e-01, -1.0246e-01, -6.3285e-01, 1.1912e-01, 6.2854e-01],
              [1.8645e-01, 1.3393e-01, -4.9750e-01, 8.4400e-02, 5.6298e-01],
              [-3.0264e-01, 3.8315e-02, 4.5638e-02, -3.9952e-01, 1.1149e+00],
              [2.9656e-01, -3.7196e-01, -6.6134e-01, 1.2188e-01, 7.3655e-01],
              [8.1650e-02, 3.7421e-01, -7.1753e-01, -6.1052e-01, 4.9217e-01],
              [-2.6778e-01, 3.5444e-03, -6.9056e-01, 2.7402e-01, 8.1634e-01],
              [1.9497e-01, -5.4937e-01, -3.9333e-01, -1.3944e-01, 1.1215e+00]],

             [[-2.8976e-01, 5.8456e-02, 4.9311e-01, 5.1019e-01, 2.3758e-02],
              [-2.0020e-01, -3.4765e-01, 2.4228e-01, 1.3289e+00, 1.0518e+00],
              [1.3852e-01, -3.0792e-01, -4.4354e-01, 6.0820e-01, 7.6708e-01],
              [2.4469e-01, -1.5907e-01, -6.5828e-01, -3.0262e-01, 2.2309e-01],
              [1.0062e-01, 6.4918e-01, 6.6005e-02, -4.3164e-02, 9.4044e-01],
              [-6.8953e-01, -4.5338e-01, -3.1631e-01, -4.6953e-01, 7.0162e-01],
              [5.5410e-01, 5.7975e-01, -7.6676e-01, -4.1268e-01, 3.8643e-01],
              [1.2324e-01, -2.2109e-01, -2.8438e-01, -6.7228e-01, 1.0095e+00],
              [-5.1889e-02, 1.1363e-01, 8.4185e-01, -2.3604e-01, 7.7119e-01],
              [-1.1096e+00, -5.5100e-02, 3.2807e-01, 3.7856e-02, 2.8809e-01],
              [-2.2305e-01, -1.8871e-02, -2.4411e-01, -2.1099e-01, 5.7992e-01],
              [-1.9480e-01, -4.2006e-01, -3.3324e-01, 1.8335e-02, 9.9111e-01],
              [-2.9440e-01, 2.2125e-01, -5.5792e-01, -5.0080e-01, 1.0431e+00],
              [-2.8706e-01, 1.6704e-01, -8.2584e-01, 6.3433e-02, -5.3505e-02],
              [3.0819e-01, -1.1755e-01, -7.1435e-02, 4.4786e-01, 1.6096e+00],
              [1.0249e-01, 4.5841e-01, -3.6804e-01, 1.4341e-01, 1.2592e+00],
              [4.5437e-02, -5.4352e-01, -6.3579e-01, 2.5254e-01, 7.0773e-02],
              [-8.9082e-01, -3.1512e-01, -6.9316e-01, -6.1477e-01, 5.2851e-01],
              [7.3867e-02, 3.4988e-01, -3.5626e-01, 3.6166e-02, 2.4879e-01],
              [5.8255e-01, -7.2015e-02, -2.7624e-01, -2.1316e-01, 3.3690e-01],
              [-5.6236e-01, 7.4583e-01, -3.9414e-01, 2.6790e-01, 2.5266e-01],
              [2.5862e-01, 3.7057e-01, -6.3182e-01, -6.2321e-01, 7.3884e-01],
              [2.6302e-01, -5.1417e-01, -7.5431e-01, -2.0465e-01, 1.0071e+00],
              [-2.5723e-01, -5.2893e-01, -6.2789e-01, -2.2892e-01, 7.8125e-02],
              [-3.1199e-01, 5.0781e-01, -2.3071e-01, -3.3404e-01, 9.4945e-01],
              [2.3473e-01, 1.0405e-02, -3.3699e-01, -6.4706e-01, 9.3063e-01],
              [-2.2759e-01, 2.5437e-01, -1.8306e-01, -6.2688e-01, 1.1281e+00],
              [5.5681e-03, -3.2277e-01, -5.3483e-01, -9.4188e-02, 9.0112e-01],
              [-4.3332e-01, 1.2618e-01, -1.1077e-01, 1.3832e-01, 4.9683e-01],
              [-1.3415e-01, -3.4990e-01, -2.6093e-01, -5.1943e-01, 4.2823e-01],
              [-4.7770e-01, 1.8815e-01, -1.4777e-01, 1.6222e-01, 4.0436e-01],
              [2.6427e-01, -3.3251e-01, -7.2267e-01, -7.9440e-01, 6.8283e-01],
              [-1.2618e+00, 2.6556e-02, -2.2770e-01, -6.0066e-01, 1.4155e-01],
              [-3.4240e-01, 1.3039e-01, -1.4545e-01, -5.1247e-02, 2.6328e-01],
              [5.9561e-01, -1.1360e-02, -7.1298e-01, 5.4985e-01, 1.2458e+00],
              [2.0271e-01, 4.9044e-01, -8.6612e-01, -9.2489e-02, 6.7482e-01],
              [1.6696e-01, -4.0083e-01, -6.9336e-01, -8.8870e-02, 2.3304e-01],
              [-2.8892e-01, 4.6363e-01, -1.0781e-01, -1.8353e-02, 6.6530e-01],
              [3.2228e-01, 8.7046e-01, -2.3595e-01, -5.2147e-01, 1.1932e+00],
              [4.8088e-01, -6.3980e-01, -4.8888e-01, -9.8864e-01, 1.2860e+00],
              [-1.3685e-01, -7.4998e-02, -7.7250e-01, -1.2224e-02, 4.7859e-01],
              [-3.2992e-02, -1.7741e-01, -4.5731e-01, 4.1837e-01, 1.5627e+00],
              [1.8450e-01, -4.9320e-01, -3.2620e-01, -4.3196e-01, 5.0752e-01]]]
         )
        return expected_out

    if testcase == 'full_trans_translate':
        expected_out = torch.FloatTensor(
            [[[-4.8176e-01, 2.7616e-01, 7.5049e-02, 4.5038e-01, 4.3136e-01],
              [-2.8781e-01, 1.3011e-01, 5.8005e-01, 1.1108e+00, 1.0721e+00],
              [-4.5948e-01, -6.8124e-01, -4.7446e-01, 8.8926e-02, 6.0755e-01],
              [-4.1190e-01, -7.1199e-01, 2.6963e-01, -2.4628e-01, 3.2017e-01],
              [7.2032e-02, -8.2675e-02, -1.0712e-01, 2.1111e-01, 9.9936e-01],
              [-4.2059e-01, 1.0115e-01, -1.1997e+00, -6.6616e-01, 1.3516e+00],
              [6.9926e-01, -2.0343e-01, -3.9382e-01, 3.1893e-01, 1.2905e+00],
              [-3.0755e-02, -7.1621e-01, -4.4173e-01, 2.7911e-01, 1.0559e+00],
              [-1.9206e-02, 9.7531e-02, 2.6851e-01, 2.2080e-01, 1.0282e+00],
              [-5.0380e-01, 4.7574e-01, -1.5425e-02, 3.7646e-01, 1.5708e+00],
              [-2.3567e-01, -1.5005e-02, -2.8170e-01, 4.8440e-01, 4.1641e-01],
              [1.6641e-01, -2.4440e-01, -5.7517e-01, -2.2512e-04, 1.4894e+00],
              [-6.8532e-01, -7.1789e-01, -6.7791e-01, 2.4855e-01, 7.5474e-01],
              [4.4367e-02, 6.4579e-01, -3.1426e-01, 1.5662e-01, 7.8239e-01],
              [7.5766e-02, 2.5095e-01, -1.4071e-01, 3.8743e-01, 1.7059e+00],
              [5.3794e-01, -3.2226e-01, -2.1623e-01, -2.6081e-01, 4.8722e-01],
              [-1.6508e-01, -3.0050e-01, -5.3420e-01, 6.3852e-01, 6.5335e-01],
              [-7.4240e-01, -1.6011e-01, -2.1274e-01, 4.9020e-01, 5.8871e-01],
              [5.3071e-01, 3.2373e-01, -2.0309e-01, -1.6928e-03, 5.9947e-01],
              [2.4328e-02, 3.9749e-01, -5.9064e-01, -3.3461e-02, 1.4434e+00],
              [3.4501e-01, -1.2736e-01, -1.5620e-01, 5.4970e-01, 5.6459e-01],
              [-2.5000e-01, -4.0311e-01, -5.5295e-01, -4.6094e-02, 7.2723e-01],
              [-1.0132e-01, 1.3449e-01, -5.5035e-01, 1.6613e-01, 1.0758e+00],
              [2.3085e-01, -2.8803e-01, -5.4546e-01, 2.5137e-01, 9.8926e-01],
              [-4.0273e-02, -7.0060e-03, -5.2273e-01, 8.3295e-02, 6.3984e-01],
              [2.7938e-01, -3.3418e-01, -8.4237e-01, 8.3115e-02, 3.0538e-01],
              [8.4939e-02, 3.5330e-02, -3.9973e-01, 1.0346e+00, 1.3729e+00],
              [9.2419e-02, 2.1753e-01, -1.5808e-01, 3.5524e-01, 1.4314e+00],
              [-2.9364e-01, -2.0394e-01, -1.1217e+00, 5.2710e-01, 6.5911e-01],
              [-3.1618e-02, -4.9085e-02, -8.8993e-01, 2.5398e-01, 8.5891e-01],
              [5.9449e-02, 3.0380e-01, 4.3595e-01, 6.2014e-01, 1.2387e+00],
              [4.3053e-01, -7.8616e-01, 6.5557e-02, 9.1982e-01, 7.1598e-01],
              [-8.9925e-02, -5.4608e-01, 2.2107e-01, 8.9566e-01, 7.5547e-01],
              [-4.7897e-01, -8.7798e-01, 7.4200e-01, 5.1963e-01, 2.0175e-01],
              [-1.1534e-01, -5.2626e-01, -7.4478e-01, 2.6747e-01, 7.7049e-01],
              [-2.6900e-01, 6.2999e-01, -4.7930e-01, 3.8568e-01, 1.2215e+00],
              [2.9447e-01, 1.3660e-01, -6.3694e-01, 3.3110e-01, 1.2649e+00],
              [3.6381e-01, 3.3251e-01, 6.1826e-02, 5.8534e-01, 1.2078e+00],
              [3.2955e-01, 7.7334e-01, 3.5571e-01, 4.4052e-02, 1.5277e+00],
              [5.1388e-01, -3.8004e-01, -3.7028e-01, 1.5861e-01, 1.3737e+00],
              [5.2996e-01, -2.2585e-01, -4.4312e-01, 6.6543e-01, 7.6086e-01],
              [1.6152e-01, -6.3352e-01, 1.7117e-02, 2.6735e-01, 1.5853e+00],
              [1.8423e-01, 9.7822e-02, -6.3467e-01, 4.1051e-01, 8.0278e-01]],

             [[-7.9326e-01, 2.4356e-01, 5.5482e-01, -1.1575e-01, 5.5490e-01],
              [-1.5251e-01, -4.4968e-02, 1.2040e-01, 5.2247e-01, 8.8957e-01],
              [-2.3002e-01, -4.7249e-01, -5.5829e-01, -6.9434e-02, 1.4522e+00],
              [-3.3325e-01, 1.9734e-01, -7.2688e-02, -4.3592e-02, 1.0595e+00],
              [4.8507e-02, -6.4723e-01, 2.9163e-02, 7.2788e-01, 1.0168e+00],
              [-6.1085e-01, -4.1194e-01, -1.0035e+00, -2.1013e-01, 1.0504e+00],
              [2.0676e-01, -3.6578e-01, -6.7763e-01, -2.2711e-01, 5.8203e-01],
              [3.6396e-01, -1.9399e-01, -4.0362e-01, -4.5673e-02, 1.1495e+00],
              [-1.1672e-01, -5.8319e-01, 4.2141e-01, 6.2506e-02, 1.3323e+00],
              [-6.8038e-01, 9.3163e-02, 1.4229e-01, 5.0435e-01, 1.2380e+00],
              [-7.5352e-01, -2.1771e-01, 7.8138e-03, 8.9520e-02, 1.8207e-01],
              [1.9153e-02, -2.9349e-01, 5.0870e-01, 6.9850e-02, 1.6195e+00],
              [-4.4971e-01, 1.7168e-01, 2.5828e-02, -2.1037e-01, 7.9407e-01],
              [-3.5256e-01, 3.9035e-01, -1.7077e-01, 3.6666e-01, 4.1278e-01],
              [-3.4157e-02, 4.9714e-02, -1.0588e-01, 6.5082e-01, 1.6700e+00],
              [-6.0730e-02, -5.6676e-04, -4.1458e-01, -1.1666e-02, 1.2956e+00],
              [-2.5106e-01, 3.2698e-02, 2.1663e-01, 6.4221e-02, 8.5694e-01],
              [-1.6239e-01, -2.0582e-02, -2.2931e-01, 4.7310e-01, 8.5316e-01],
              [8.5144e-02, -9.9867e-03, 1.5274e-01, 3.8616e-01, 6.9804e-01],
              [-4.1882e-02, 6.0164e-01, 8.6178e-02, 2.4634e-01, 1.2967e+00],
              [-3.7608e-01, -6.4579e-01, -5.2446e-01, 6.7989e-01, 8.4290e-01],
              [-3.1589e-01, 1.8718e-01, -3.5554e-01, -4.8686e-01, 1.5436e+00],
              [2.2011e-01, -3.7342e-02, -1.0278e+00, -6.4004e-02, 1.0790e+00],
              [-2.2091e-01, 2.3036e-01, -1.8409e-01, 2.5250e-01, 1.1999e+00],
              [-2.6768e-01, 1.9146e-01, -3.3950e-01, 8.7540e-04, 1.2861e+00],
              [2.7971e-01, -4.8545e-01, -9.2327e-02, 1.8435e-01, 8.1781e-01],
              [-3.0292e-01, 6.3334e-02, -1.1617e-01, 7.9806e-01, 1.1045e+00],
              [1.1245e-01, -1.5470e-01, 1.8751e-02, 8.0555e-01, 1.4192e+00],
              [-2.3493e-01, 1.2126e-01, -1.1860e-01, 9.7264e-01, 9.4792e-01],
              [-1.4040e-01, -1.3905e-01, -3.8495e-01, -6.6514e-02, 1.0898e+00],
              [-2.9393e-01, -2.8232e-01, 3.7097e-01, 1.9641e-03, 5.9399e-01],
              [-2.4921e-01, -7.2325e-01, -1.2389e-01, -1.0400e-01, 1.0390e+00],
              [-1.6282e-01, -4.1089e-01, -4.2076e-01, -3.2805e-03, 6.7523e-01],
              [-5.1485e-01, 8.5265e-01, 8.5109e-01, 5.3612e-01, 1.3325e+00],
              [-3.2950e-02, 2.1676e-01, -3.7511e-01, 4.5445e-01, 1.8543e+00],
              [-7.1849e-02, -2.3559e-01, -3.5329e-02, 3.3816e-01, 1.4587e+00],
              [-1.4479e-01, -1.4658e-01, -7.2203e-01, 2.8436e-01, 7.7346e-01],
              [3.6226e-01, 6.0488e-01, 2.6634e-01, 6.3428e-01, 1.2273e+00],
              [-3.3479e-01, 3.5737e-01, 3.5491e-01, 2.1683e-01, 1.6383e+00],
              [2.4232e-01, -5.2012e-01, -2.7085e-01, 3.3098e-01, 9.2595e-01],
              [-2.9237e-01, 2.7623e-01, -2.5897e-01, 4.6753e-01, 8.9860e-01],
              [-4.3194e-01, -2.9920e-01, 2.0123e-01, 3.9163e-01, 1.7467e+00],
              [6.6839e-01, -2.1980e-01, 3.1417e-02, -1.2329e-01, 1.0655e+00]],

             [[-1.0257e-01, -9.5984e-02, 3.9792e-01, 4.6262e-01, 6.3653e-02],
              [8.5408e-02, 1.2615e-01, -6.5660e-02, 9.0802e-01, 8.9439e-01],
              [-5.1049e-01, -9.6448e-01, -4.3326e-01, -2.4569e-01, 6.3654e-01],
              [3.9953e-01, -9.8523e-01, -3.0819e-01, -5.7089e-01, 7.4551e-01],
              [-3.0852e-01, -5.0384e-01, -4.1714e-01, 5.9197e-01, 1.1823e+00],
              [-5.1340e-01, -1.1283e+00, -9.8856e-01, -1.3617e-01, 9.9172e-01],
              [4.9104e-01, 1.6731e-01, -9.0477e-01, -3.0818e-01, 1.3937e+00],
              [4.1693e-01, -5.2988e-01, -7.1511e-01, -2.0499e-01, 8.7263e-01],
              [1.4023e-01, 5.8402e-02, 1.2109e-01, 1.2008e-01, 1.2996e+00],
              [-1.6160e-01, -1.9478e-01, 1.7385e-01, 5.2288e-01, 9.0534e-01],
              [-7.6864e-02, -7.2856e-01, -9.1734e-01, 4.9143e-01, 1.0031e+00],
              [3.4084e-01, -5.0936e-01, -5.5195e-02, 1.8893e-01, 1.1378e+00],
              [-3.8993e-01, -1.0674e-01, -3.1585e-01, 4.9136e-01, 1.3354e+00],
              [5.1204e-01, 2.5601e-01, -4.2808e-01, -8.2625e-02, 7.9132e-01],
              [7.6865e-02, -3.6190e-01, -2.6874e-01, 1.2427e+00, 1.2963e+00],
              [4.1334e-01, 2.6175e-01, 6.9800e-03, -2.3850e-01, 1.6004e+00],
              [7.6667e-01, -6.8881e-01, -3.6885e-01, 3.2802e-01, 8.7825e-01],
              [-2.8231e-01, -8.5661e-02, -4.2475e-01, 6.4673e-01, 7.5187e-01],
              [5.5166e-01, -2.9921e-01, -4.6923e-01, -2.7244e-01, 1.0580e+00],
              [6.6336e-02, 2.1916e-02, -3.8259e-02, -7.7930e-02, 1.3046e+00],
              [4.3098e-01, -8.8060e-02, -3.2040e-01, 6.0336e-01, 9.7113e-01],
              [-4.1528e-01, -4.6588e-01, -5.4050e-01, -1.4505e-01, 1.0445e+00],
              [2.9039e-01, -1.2079e-01, -5.9341e-01, 1.7426e-02, 1.5117e+00],
              [-3.3176e-01, -2.4981e-01, -4.0661e-01, 7.7122e-01, 1.5683e+00],
              [3.0306e-01, -8.9118e-02, -5.2497e-01, 4.5576e-01, 1.2725e+00],
              [1.8583e-01, 1.3522e-01, -5.5937e-01, 4.4386e-01, 8.6130e-01],
              [-7.6841e-02, 3.4473e-01, 1.3622e-01, 3.7955e-01, 1.6055e+00],
              [1.4049e-01, -5.3844e-01, -2.7895e-01, 9.4561e-02, 8.5332e-01],
              [-1.5524e-01, -5.8744e-01, -5.0222e-01, 3.8306e-01, 1.1222e+00],
              [1.5741e-01, -1.4954e-01, -4.6173e-01, 1.6295e-01, 7.4931e-01],
              [1.6128e-01, 2.9332e-01, 1.5433e-01, 2.2147e-01, 6.4891e-01],
              [-2.3380e-01, -2.4658e-01, 8.8958e-02, -2.3271e-01, 1.3517e+00],
              [-9.9264e-02, 2.1367e-01, -1.7571e-01, 5.2580e-01, 7.2081e-01],
              [2.3539e-02, -9.0086e-02, 3.0333e-01, 3.4116e-01, 1.7753e+00],
              [-1.1620e-01, 1.6229e-01, -5.0215e-01, 2.8078e-01, 1.5002e+00],
              [-1.9819e-01, -2.7795e-01, -5.8840e-01, 1.8987e-01, 1.5904e+00],
              [3.3362e-01, -1.7521e-01, -4.2755e-01, 5.5693e-02, 1.0397e+00],
              [4.3966e-01, 8.0232e-01, -4.2084e-01, 5.5216e-02, 1.6535e+00],
              [1.6840e-01, -1.7426e-01, 2.9581e-01, -2.7033e-01, 2.2219e+00],
              [5.3958e-01, -7.5812e-01, -4.4123e-01, -3.7893e-01, 1.2719e+00],
              [6.1689e-02, 9.5145e-02, 6.6215e-03, 2.5107e-01, 1.7015e+00],
              [9.5639e-02, 4.0796e-01, -3.1851e-01, 2.2392e-01, 1.8748e+00],
              [2.9118e-01, 1.3730e-02, -1.9782e-01, -2.3815e-01, 1.2603e+00]]]
        )
        return expected_out



