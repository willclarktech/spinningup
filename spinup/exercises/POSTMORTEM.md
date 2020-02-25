# Spinning Up Exercises Postmortem

Exercises from https://spinningup.openai.com/en/latest/spinningup/exercises.html

Completed using `pytorch`.

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

### Exercise 2.2: Silent Bug in DDPG
