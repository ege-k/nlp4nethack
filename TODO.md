## Actionables

### Runtime optimization
* parallelize actor model
   * convert last argument to tensor
   * if that does not work, open Torch issue
   * if this succeeds, don't forget to find batch_size, num_actors, etc. to optimize runtime
* use DDP instead of DP

### Algorithmic optimization
* use embeddings in the CNN layer
* use embeddings in the action layer

## Things to think about

### Runtime
* What exactly is the interplay between num_actors, actor_model and learner_model?
   * how are CPUs utilized?

### Algorithmic
* Is one attention layer "enough"?
* How to best partition the corpus?
