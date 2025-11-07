# Variational Path Flows

`vpf.py for the goods`

I've been trying to share more, since I tend to just solo hack and don't end up showing people stuff, and I am never going be the guy who writes papers.
This is still a work in progress, but directionally it is what I was thinking about on my walk this morning.

This was an attempt to formulated infinite resolution / continous latent sde vector fields without the need to solve an ODE/SDE at inference, which makes rollouts a lot nicer. And without dual stage training or having to compute a model wide JVP like mean flows or collapsing the trajectory entropy like other distilation/consistency methods.  I also wanted an approach that would be suitable for inference time continous/latent space search. Having a nice partition function / known density in the places that tend to drift off manifold with gradient based latent optimisation tends to help here. 

Again, this is a first pass. I'll be thinking about this a bit more.

The current varient is an exact path-likelihood latent sequence model with conditional observation flow:
    z_t = f_t(y_t),  y_t follows a diagonal Ornstein–Uhlenbeck process, conditioned on y_0 
    p(z_0) is a learned diagonal Gaussian in z-space.
    x_t = g(u_t ; z_t),  u_t ~ N(0, I)  (conditional normalizing flow observation)
 Loss:
   E_q[ sum_t log p(x_t|z_t) + log p(z0) + log p(z_{1:T}|z0) - log q(z_{0:T}|x) ]

Not really thinking about the rest of the archecture, just the objective formulation. You can replace any of it with task specific archectures.

- **`prior_flow = NF_SDE_Model(...)`**  
  Time-conditioned RealNVP mappin; base dynamics is diagonal OU SDE in y-space; exact path log-likelihood via change-of-variables.

- **`z0_prior = PriorInitDistribution(...)`**  
  Learned diagonal Gaussian over  z_0 (trainable mean/scale).

- **`p_obs_flow = ObservationFlow(...)`**  
  Conditional RealNVP likelihood p(x_t | z_t): invertible given z_t with tractable log-det.

- **`q_enc = PosteriorEncoder(...)`**  
  Sequence encoder producing global + per-step context. 

- **`q_affine = PosteriorAffine(...)`**  
  Time-adaptive affine Gaussian head: mu_t, sigma_t for  q(z_t | x, t) from encoder context + scalar time.

Grigory Bartosh and his team did some very relevant work, but doesn't attempt to do single pass inference. But its great alternative if you don't need this. 
`https://arxiv.org/abs/2502.02472`
