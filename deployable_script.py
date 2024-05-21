import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
tf.contrib.eager.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

poisson_distributions = [
    tfd.Poisson(rate=1., name='One Poisson Scalar Batch'),
    tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons'),
    tfd.Poisson(rate=[[1., 10., 100.,], [2., 20., 200.]],
                name='Two-by-Three Poissons'),
    tfd.Poisson(rate=[1.], name='One Poisson Vector Batch'),
    tfd.Poisson(rate=[[1.]], name='One Poisson Expanded Batch')
]

three_poissons = tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons')
three_poissons.log_prob([10.])
three_poissons.log_prob([[[1.], [10.]], [[100.], [1000.]]])
three_poissons.log_prob([[1., 10.], [100., 1000.]])
six_way_multinomial = tfd.Multinomial(total_count=1000., probs=[.3, .25, .2, .15, .08, .02])
transformed_multinomial = tfd.TransformedDistribution(
    distribution=six_way_multinomial,
    bijector=tfb.Reshape(event_shape_out=[2, 3]))
event = [500., 100., 100., 150., 100., 50.]
event_ = [[500., 100., 100.], [150., 100., 50.]]
assert six_way_multinomial.log_prob(event).numpy() == \
    transformed_multinomial.log_prob(event_).numpy()
two_by_five_bernoulli = tfd.Bernoulli(
    probs=[[.05, .1, .15, .2, .25], [.3, .35, .4, .45, .5]])
two_sets_of_five = tfd.Independent(
    distribution=two_by_five_bernoulli,
    reinterpreted_batch_ndims=1)
event = [[1., 0., 0., 1., 0.], [0., 0., 1., 1., 1.]]
print(two_by_five_bernoulli.log_prob(event))
print(two_sets_of_five.log_prob(event))