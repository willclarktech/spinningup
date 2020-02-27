# Spinning Up Exercises Postmortem

Exercises from https://spinningup.openai.com/en/latest/spinningup/exercises.html

Completed using pytorch.

## Problem Set 1: Basics of Implementation

### Exercise 1.1: Gaussian Log-Likelihood

-   Successfully completed exercise with respect to the test.
-   The solution involved creating a tensor which detached from the graph, so that had to be rewritten when the function was reused in later exercises.
-   My first implementation did not make use of numpy magic, and was tightly coupled to the shape of the function inputs, so it had to be rewritten when a later exercise used the function with `x` and `mu` tensors with different shapes.

### Exercise 1.2: Policy for PPO

-   Successfully implemented `mlp` function.
-   Successfully implemented `MLPGaussianActor.__init__` method.
-   Incorrectly implemented `DiagonalGaussianDistribution.sample` method: I mistakenly assumed I should use the provided `entropy` method to generate the "vector z of noise from a spherical Gaussian (`z ~ N(0, I)`)", when in fact I should have used a normal random number generator.

### Exercise 1.3: Computation Graph for TD3

-   Successfully implemented `compute_loss_q` function.
-   Incorrectly implemented `compute_loss_pi` function: I had conflated the `pi` and `act` methods on the actor-critic. I also used the wrong sign on the result of the function: I was focusing too myopically on the formula (given in the documentation) for which value to maximise and did not think about how this related to the _loss_ function required in the code.

## Problem Set 2: Algorithm Failure Modes

### Exercise 2.1: Value Function Fitting in TRPO

n/a

### Exercise 2.2: Silent Bug in DDPG

There was a big clue in the doc string for the `ddpg` function, describing the output of the `q` module of the `actor_critic` argument:

> Tensor containing the current estimate of Q\* for the provided observations and actions. (Critical: make sure to flatten this!)

Checking the shape of the Q function's output confirmed it had rank 2, not rank 1, so I added a `.flatten()` and the bug was fixed. Checking against the provided implementation, they use `torch.squeeze(q, -1)` instead of `flatten`, which is equivalent in this case.

As the solution page explains,

> The line that produces the Bellman backup was written with the assumption that it would add together tensors with the same shape. However, this line can also add together tensors with different shapes, as long as they’re broadcast-compatible.

From pytorch's docs on broadcasting:

> If two tensors x, y are “broadcastable”, the resulting tensor size is calculated as follows:
>
> -   If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
> -   Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

So when the experiment is run with a batch size of 64, and the output of the buggy Q function is multiplied by the `d` mask, we have a tensor of shape `(64, 1)` multiplied by a tensor of shape `(64,)`, which is prepended with ones to become `(1, 64)` due to broadcasting, resulting in a tensor with shape `(64, 64)`, which results in further broadcasting errors. With the correct code we multiply two tensors of shape `(64,)`, resulting in another tensor of shape `(64,)`.

I'm left wondering why people don't pepper their code with assertions against the shape of tensors everywhere, at least in development. I'm also even more scared than I was before of numpy/broadcasting magic.
