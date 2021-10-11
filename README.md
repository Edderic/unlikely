## TODO
- Make weighting scheme handle a mixture of variable types
   (ordinal vs. non-ordinal)

  - Implement priors
       TODO: Problem: When having discrete and continuous
       distributions, The PDF of continous distributions can be
       larger than the PMF of discrete distributions, the latter
       upper-bounded by 1, but the former possibly being greater
       than 1.

       Possible solution: Can use the PMF of continuous
       distributions. For a continuous distribution, have x - width
       and x + width. Subtract the cdf of x-width from cdf of
       x+width.

       ContinuousUnsampled
       - pmf(x, width)
         CDF(x+width) - CDF(x-width)

       ContinuousSampled
       - pmf(x, width)
         Count the number of items within (x-width, x+width)

       DiscreteOrdinalSampled
       - pmf(x, width)
         Disregard width

       DiscreteNonordinalSampled
       - pmf(x, width)
         Disregard width

- Serialize data.
- Deserialize data.
- Prevent overflow by computing in log space

- Add tests
  - Test the case of Bayesian updating.
    - E.g. one-by-one updating of a Bernoulli distributed data vs. updating the
      prior with the Binomial full data all at once
    - They should be the same


DONE:
- Implement model comparison
- Implement Normal distribution
- Implement HalfCauchy distribution

## Notes

The ABC-SMC scheme of Toni et al. involves creating intermediary distributions,
that get closer and closer to the true posterior distribution as the distance
gets closer and closer to 0. In each step of creating intermediary
distributions, we are collecting a finite amount of samples. Since we're not
collecting an infinite amount, there's a chance that the tail ends of the
distribution won't get sampled. Not including an acceptable particle in one
population means that the next population would be missing the tails. The more
populations there are, the more likely that we would be undersampling tails.
Having a perturbation kernel lets the algorithm sample more from the tails.
Larger perturbations is beneficial because samples close to the extremes (which
have a low probability of being explored) become more likely to be explored.
However, if it's set too large, then we might be putting too much weight to
these low probability particles, making the variance of the posterior larger
than what it should be.

## Issues:

Intermediate distribution sometimes assigns a lot of weight to highly
improbable areas, probably because the weighting scheme gets really close to 0
in the denominator.

The weighting scheme is:
  Prior(x) / sum of each i-th particle's previous weight times the kernel pdf of (perturbed particle given particle i)

Using a Gaussian kernel, if a particle is proposed that is really far (i.e. low
pdf values) from the accepted particles of the previous generation, then
the denominator would be tiny, which would blow up the weight for the perturbed
particle.

Possible solution:

Use a Uniform kernel instead. If something is close enough, then we use the PDF
of it, multiplied by some weight, then add all those up. Less likely to be
really tiny?

We could also add some positive value to the denominator, so that if a
proposal is made, and all the accepted particles from the previous generation
are too far from the perturbed proposal, we won't blow up the weight too much.

What if the value to add was 1? In that case, we're essentially sampling from the prior.

If we encounter a perturbed particle that is close enough, then the denominator
would be greater than 1, which would downweight said particle a little bit. We
know this is a little bit because weights from the previous round are
normalized (which are essentially probabilities.) We're multiplying a
probability times the kernel value, which is probably going to be tiny. Not
necessarily? The kernel value could be pretty high if the distribution is
narrow (e.g. Normal distribution with standard deviation 0.1). What if we
normalize the densities?
