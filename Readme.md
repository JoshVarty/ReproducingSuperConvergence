## Reproducing Super Convergence


This is an attempt to reproduce a subset of the results found in [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://openreview.net/forum?id=H1A5ztj3b).

Super-Convergence is described as *"a phenomenon... where residual networks can be trained using an order of magnitude fewer iterations
than is used with standard training methods"*.

Figure 1A demonstrates the phenomenon below:

<center><img src="https://i.imgur.com/JQ8lHHA.png" width="400" /></center>
<center>Cyclical Learning Rate (CLR) allows for competitive training in just 10,000 training steps.</center>

Independent evidence of super-convergence is demonstrated below:

<center>
    <img src="https://i.imgur.com/e9RXHl1.png" width="350" />
    <img src="https://i.imgur.com/PGZ9nlI.png" width="350" />
    <p><strong>Left: </strong>Test accuracy after 10,000 steps with CLR.</p>
    <p><strong>Right: </strong>Test accuracy after 80,000 steps with multistep.</p>
</center>

In the above images:
 - A Cyclical Learning Rate allows for a test accuracy of ~85% after 10,000 training steps.
 - A multistep learning rate allows for a test accuracy ofr ~80% after 80,000 training steps.


### Architecture

The Tensorflow implementation in based on the ResNet-56 architecture described in Appendix A of [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://openreview.net/pdf?id=H1A5ztj3b) with the following changes:

#### Corrections
- The 3x3 Conv Layer at the start of the network has `stride=1`, not  `stride=2` as mentioned in the paper.

#### Undocumented Elements
 - While training, images are flipped left-to-right with 50% probability
 - All weights before ReLUs are initialized according to [Delving Deep into Rectifiers](https://arxiv.org/pdf/1502.01852v1.pdf). See: [`variance_scaling_initializer`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer)
 - All weights before softmax are initialized according to [Understanding the difficulty of training deep feedforward neural networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf). See: [`xavier_initializer`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer)
