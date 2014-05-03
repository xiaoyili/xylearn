__author__ = 'eric'

from sklearn.datasets import load_digits

digits = load_digits()

print digits.data.shape
print max(digits.target) - min(digits.target) + 1

from sklearn.datasets import load_iris

iris = load_iris()

print iris.data.shape
print iris.target.shape

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

print mnist.data.shape
print mnist.target.shape

# shuffle data
import numpy
data = numpy.random.randint(5, size=(5,5))
label = numpy.random.randint(2, size=(5,))

from sklearn.utils import shuffle
x,y = shuffle(data, label)

print x, y