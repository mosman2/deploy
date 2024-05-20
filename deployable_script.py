# Auto-generated deployable script

# Dependencies
import os
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Models and Model Parameters
tfd.Poisson(rate=1., name='One Poisson Scalar Batch')
tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons')
tfd.Poisson(rate=[[1., 10., 100.,], [2., 20., 200.]], name='Two-by-Three Poissons')
tfd.Poisson(rate=[1.], name='One Poisson Vector Batch')
tfd.Poisson(rate=[[1.]], name='One Poisson Expanded Batch')
tfd.Multinomial(total_count=1000., probs=[.3, .25, .2, .15, .08, .02])
tfd.TransformedDistribution(distribution=six_way_multinomial, bijector=tfb.Reshape(event_shape_out=[2, 3]))
tfd.Bernoulli(probs=[[.05, .1, .15, .2, .25], [.3, .35, .4, .45, .5]])
tfd.Independent(distribution=two_by_five_bernoulli, reinterpreted_batch_ndims=1)

# Functions
def initialize_environment():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enable_eager_execution():
    tf.contrib.eager.enable_eager_execution()

def print_versions():
    print('TensorFlow version {}, TF Probability version {}.'.format(tf.__version__, tfp.__version__))

def create_poisson_distributions():
    poisson_distributions = [
        tfd.Poisson(rate=1., name='One Poisson Scalar Batch'),
        tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons'),
        tfd.Poisson(rate=[[1., 10., 100.,], [2., 20., 200.]], name='Two-by-Three Poissons'),
        tfd.Poisson(rate=[1.], name='One Poisson Vector Batch'),
        tfd.Poisson(rate=[[1.]], name='One Poisson Expanded Batch')
    ]
    return poisson_distributions

def print_poisson_distributions(distributions):
    print('\n'.join([str(d) for d in distributions]))

def create_three_poissons():
    return tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons')

def create_six_way_multinomial():
    return tfd.Multinomial(total_count=1000., probs=[.3, .25, .2, .15, .08, .02])

def create_transformed_multinomial(six_way_multinomial):
    return tfd.TransformedDistribution(
        distribution=six_way_multinomial,
        bijector=tfb.Reshape(event_shape_out=[2, 3]))

def create_two_by_five_bernoulli():
    return tfd.Bernoulli(
        probs=[[.05, .1, .15, .2, .25], [.3, .35, .4, .45, .5]])

def create_two_sets_of_five(two_by_five_bernoulli):
    return tfd.Independent(
        distribution=two_by_five_bernoulli,
        reinterpreted_batch_ndims=1)

# Inference Code
three_poissons.log_prob([10.])
three_poissons.log_prob([[[1.], [10.]], [[100.], [1000.]]])
three_poissons.log_prob([[1., 10.], [100., 1000.]])
assert six_way_multinomial.log_prob(event).numpy() == transformed_multinomial.log_prob(event_).numpy()
two_by_five_bernoulli.log_prob(event)
two_sets_of_five.log_prob(event)

