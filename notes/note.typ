// #import "@preview/thmbox:0.1.1": *
#import "@preview/physica:0.9.3": *
// #import "@preview/noteworthy:0.1.0": *
#import "./lib.typ": *

// #show: thmbox-init()
#show: mynoteinfo.with(
  paper-size: "a4",
  font: "New Computer Modern",
  language: "EN",
  title: [Free-Energy Machine for Combinatorial Optimization],
  author: "Xiwei Pan",
  // reference: "Bart De Schutter, Bart De Moor"
//   contact-details: "https://example.com", // Optional: Maybe a link to your website, or phone number
  // toc-title: "Table of Contents",
)

= Methods

== Using the Boltzmann distribution to solve COPs
Any Combinatorial Optimization Problem (COP) can be characterized by a #underline([cost funtion]) $E(bold(sigma))$, where $bold(sigma) = (sigma_1, sigma_2, ..., sigma_N)$ is the configuration comprising $N$ variables.

In this work, the authors proposed 4 type of COPs.
+ Ising model (or QUBO problem): $ E(bold(sigma)) = -sum_(i < j) W_(i j) sigma_i sigma_j, "where" sigma_i = {+1,-1}. $
+ Potts model: $ E(bold(sigma)) = -sum_(i < j) W_(i j) delta_(sigma_i, sigma_j), "where" sigma_i in {1,2,...,q}. $
+ $p$-spin glass model (or PUBO problem): $ E(bold(sigma)) = -sum_(i_1 < i_2 < ... < i_p) W_(i_1 i_2 ... i_p)  sigma_(i_1) sigma_(i_2) ... sigma_(i_p), "where" sigma_i = {+1,-1}. $
+ General model: $ E(bold(sigma)) = -sum_(i_1 < i_2 < ... < i_p) W_(i_1 i_2 ... i_p)  sigma_(i_1) sigma_(i_2) ... sigma_(i_p), "where" sigma_i in {1,2,...,q}. $

The authors proposed one variational framework for all these problems.
$ P_B (bold(sigma), beta) = (1)/(Z) e^(-beta E(bold(sigma))), "where" beta = 1/T , "and" Z = sum_(bold(sigma)) e^(-beta E(bold(sigma))). $
For $i$-th spin $sigma_i$, the probability of being in state $j$ is $P_B (sigma_i = j, beta)$.

The ground state configuration $bold(sigma)^("GS")$ is the configuration that minimizes the cost function $E(bold(sigma))$, i.e., $ bold(sigma)^("GS") = arg min_(bold(sigma)) E(bold(sigma)) = arg max_(bold(sigma)) lim_(beta -> infinity) P_B (bold(sigma), beta) $.

== Using variational mean-field distribution to approximate the Boltzmann distribution

In this work, $P_"MF" (bold(sigma)) = product_i P_i (sigma_i)$ is used to approximate $P_B (bold(sigma), beta)$.
Minimizing the Kullback-Leibler divergence between $P_B (bold(sigma), beta)$ and $P_"MF" (bold(sigma))$ is equivalent to minimizing the variational free energy $ F_"MF" = sum_bold(sigma) P_"MF" (bold(sigma)) E(bold(sigma)) + 1/beta sum_bold(sigma) P_"MF" (bold(sigma)) ln P_"MF" (bold(sigma)). $

Then the authors devide the problem into two parts:
+ Energy term: $ U_("MF") = sum_bold(sigma) P_"MF" (bold(sigma)) E(bold(sigma)). $
+ Entropy term: $ S_("MF") = - sum_bold(sigma) P_"MF" (bold(sigma)) ln P_"MF" (bold(sigma)). $

So we have $ F_"MF" = U_("MF") - 1/beta S_("MF"). $

Here, the authors assign local fields $h_i (sigma_i)$ to each spin $sigma_i$, or each marginal probability $P_i (sigma_i)$, and use the following ansatz: $ P_i (sigma_i) = "softmax" (h_i (sigma_i)) = e^(h_i (sigma_i))/(sum_(sigma_i=1)^(q_i) e^(h_i (sigma_i))). $

= Applications

== Max-Cut problem

$ E(bold(sigma)) = - sum_((i,j) in cal(E)) W_(i,j) (1-delta(sigma_i, sigma_j)) $

