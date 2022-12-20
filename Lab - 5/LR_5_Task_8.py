import numpy as np
import numpy.random as rand
import neurolab as nl
import pylab as pl

skv = .04
center = np.array([[0.2, 0.3], [0.4, 0.4], [0.7, 0.3], [0.1, 0.5], [0.4, 0.5]])
random_norm = skv * rand.randn(100, 5, 2)
inp = np.array([center + r for r in random_norm])
inp = inp.reshape(100 * 5, 2)
rand.shuffle(inp)

net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 5)
error = net.train(inp, epochs=200, show=20)

pl.title('Classification problem')
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')
w = net.layers[0].np['w']

pl.subplot(212)
pl.plot(inp[:, 0], inp[:, 1], '.', center[:, 0], center[:, 1], 'yv', w[:, 0], w[:, 1], 'p')
pl.legend(['train samples', 'centers', 'train centers'])
pl.show()
