#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from datageneration import sparse_data
from learning import multilayer_backprop, create_model

data = sparse_data()

encoder, decoder, error = multilayer_backprop(data, data, 3, epochs=1000)

encoder = create_model(encoder)
decoder = create_model(decoder)

test_data = data[1, :]
print('Input vector:  ', test_data)
encoding = encoder(test_data)
print('Encoded vector:', encoding)
decoding = decoder(encoding)
print('Decoded vector:', decoding)
print('error:         ', test_data - decoding)

plt.plot(error)
plt.show()

# weights, hidden_weights, error = trainMultiLayer(
#     data, data, 0.001, 3, nOutput=8
# )
