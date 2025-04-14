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

Solver these problems is equivalent to finding the ground state configuration $bold(sigma)^("GS")$ ,
$ bold(sigma)^("GS") = arg min_(bold(sigma)) E(bold(sigma)). $

The authors proposed one variational framework for all these problems. Consider the #underline([Boltzmann distribution]) at a specific temperature $T$, 
$ P_B (bold(sigma), beta) = (1)/(Z) e^(-beta E(bold(sigma))), "where" beta = 1/T , "and" Z = sum_(bold(sigma)) e^(-beta E(bold(sigma))). $
(For $i$-th spin $sigma_i$, the marginal probability of being in state $j$ is $P_B (sigma_i = j, beta)$.)

#figure(
  image("assets/intro.png")
)
The ground state configuration $bold(sigma)^("GS")$ that minimizes the energy can be achieved at the #underline([zero-temperature limit]), i.e., $ bold(sigma)^("GS") = arg min_(bold(sigma)) E(bold(sigma)) = arg max_(bold(sigma)) lim_(beta -> infinity) P_B (bold(sigma), beta). $

#figure(
  image("assets/opt.png")
)


As the temperature $T -> 0 (beta -> infinity)$, this distribution becomes sharply peaked at the lowest energy states (the ground states).

In many complex systems, the #underline([*energy landscape*]) is rugged — imagine hills and valleys:
	-	There are many local minima.
	-	These valleys are separated by high-energy barriers.
	-	At zero temperature, the system can get stuck in a local minimum and can’t escape.


This phenomenon is often described using #underline([*replica symmetry breaking*]) in statistical physics: the state space splits into many disconnected basins, each with their own low-energy configurations, and it’s hard to move between them.

To avoid getting trapped in local minima, we use #underline([*annealing*]), which simulates the system at finite temperature and gradually cools it down.



== Using variational mean-field distribution to approximate the Boltzmann distribution

Exactly computing the Boltzmann distribution belongs to the computational class of \#P, so we need to approximate it efficiently.

In this work, $P_"MF" (bold(sigma)) = product_i P_i (sigma_i)$ is used to approximate $P_B (bold(sigma), beta)$.

#note[
  Here $P_i (sigma_i)$ is the marginal probability of the $i$-th spin $sigma_i$, which can be intepreted as a vector like $[P_i (sigma_i = 1), P_i (sigma_i = 2), ..., P_i (sigma_i = q_i)]$. $P_"MF" (bold(sigma)) = product_i P_i (sigma_i)$ is the joint probability of all spins, which can be intepreted as a $N$-dimensional tensor.
]

#theorem[Minimizing the Kullback-Leibler divergence between $P_B (bold(sigma), beta)$ and $P_"MF" (bold(sigma))$ is equivalent to minimizing the variational free energy $ F_"MF" = sum_bold(sigma) P_"MF" (bold(sigma)) E(bold(sigma)) + 1/beta sum_bold(sigma) P_"MF" (bold(sigma)) ln P_"MF" (bold(sigma)). $]
#proof[Noticed that $P_B (bold(sigma), beta) = 1 / Z  e^(-beta E(bold(sigma))), $ then we have, 
  $ D_"KL" (P||P_B) &= sum_bold(sigma) P (bold(sigma)) ln (P(bold(sigma))/(P_B (bold(sigma), beta))) = sum_bold(sigma) P (bold(sigma)) ln P (bold(sigma)) - sum_bold(sigma) P (bold(sigma)) ln P_B (bold(sigma), beta)\ &= sum_bold(sigma) P (bold(sigma)) (ln P (bold(sigma)) + beta E(bold(sigma)) + ln Z)\ &= beta (sum_bold(sigma) P (bold(sigma)) E(bold(sigma)) + 1/beta sum_bold(sigma) P (bold(sigma)) ln P (bold(sigma)) + 1/beta ln Z) $
]

Then the authors devide the free energy into two parts:
+ Energy term: $U_("MF") = sum_bold(sigma) P_"MF" (bold(sigma)) E(bold(sigma)). $
+ Entropy term: $S_("MF") = - sum_bold(sigma) P_"MF" (bold(sigma)) ln P_"MF" (bold(sigma)). $

So we have $ F_"MF" = U_("MF") - 1/beta S_("MF"). $

#figure(
  image("assets/methods.png")
)

In conclusion, for a specific $beta$, 

$ min F_"MF" => "Adjust" P_"MF" => "Adjust" P_i (sigma_i) => ... $

Here, the authors assign local fields $h_i (sigma_i)$ to each spin $sigma_i$, i.e. each marginal probability $P_i (sigma_i)$, with `softmax` function: $ P_i (sigma_i) = "softmax" (h_i (sigma_i)) = e^(h_i (sigma_i))/(sum_(sigma_i=1)^(q_i) e^(h_i (sigma_i))). $

To minimize the free energy $F_"MF"$, we need to calculate the gradient of $F_"MF"$, ${g_i^h (sigma_i)}$.
Here the authors use the #underline([*automatic differentiation*]) to calculate the gradient. Also, Utilizing #underline([explicit gradient formulations]) can halve the computational time and enhance the numerical stability.  

At a high temperature, there could be just one mean-field
distribution that minimizes the variational free energy. However, at a low temperature, there could be many mean-field distributions, each of which has a local free energy minimum, corresponding to a set of marginal probabilities. So the authors use a set of marginal distributions, which they term as #underline([$m$ replicas of mean-field solutions with parameters]), each of which is updated to minimize the corresponding mean-field free energy.

Then choose the configuration with the minimum energy
from all replicas.

= Applications

== Max-Cut problem

$ E(bold(sigma)) = - sum_((i,j) in cal(E)) W_(i,j) (1-delta(sigma_i, sigma_j)) $

