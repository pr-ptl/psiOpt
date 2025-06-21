# psiOpt
An adjoint-based streamfunction optimiser

#---------------------------------------------------------------------

INTRODUCTION

A fundamental theorem of vector calculus known as Helmholtz decomposition, describing a sufficiently smooth vector field as its divergence free and curl free component.

If seen from a perspective of fluid mechanics, this is described as a motion of a volume element of continuous fluid media in $\R^3$ consisting of
- An expansion or contraction in three orthogonal directions
- Rotation about an axis
- Translation (could either be represented by curl free or divergence free component)

and in 2D, an Inverse problem (otherwise known as helmholtz decomposition) would look like

$$
\mathbf{\xi} = \nabla D + \nabla X \mathbf{R}
$$

were, \nabla D is curl free and \nabla X \mathbf{R} is divergence free component respectively. These elements corresponds to

$$
\mathbf{F} = \nabla^\perp \psi + \nabla \phi
$$

Inturn defining a vector in term of streamfunctions $\psi$ and potential $\phi$ (if its a velocity vector). This theoretical picture which technically doesn't exist forms the basis of a vector with added errors.

#---------------------------------------------------------------------

STREAMFUNCTION OPTIMIZATION FROM HELMHOLTZ DECOMPOSITION

OVERVIEW

#---------------------------------------------------------------------

This project extends the classical Helmholtz–Hodge decomposition framework to solve an inverse problem in computational fluid dynamics: reconstructing the streamfunction $\psi$ from a known target velocity field (u, v) using optimization.

Instead of simply decomposing a vector field into its divergence-free and curl-free
components, the problem has been posed as minimizing a loss function:

$$
\mathcal{J}(\psi) = \frac{1}{2} \sum_{i,j} \left[
\left( \frac{\partial \psi}{\partial y} - u_{\text{target}} \right)^2 +
\left( -\frac{\partial \psi}{\partial x} - v_{\text{target}} \right)^2
\right]
$$

This yields an optimized $\psi$ such that the inferred velocity field closely matches
the given target field — suitable for inverse design, flow reconstruction, and data
assimilation.

PROJECT SCOPE

#---------------------------------------------------------------------

- Takes in a 2D velocity field (u_{target}, v_{target})
- Initializes a streamfunction $\psi$ (e.g., from Helmholtz decomposition or zeros)
- Iteratively updates $\psi$ to minimize the velocity mismatch loss
- Supports different optimizers (RMSprop, Adam, Stochastic gradient descent) - extends the optimsation based on chosen method
- Provides visualizations for:
    - Reconstructed (optimized) field
    - Error field
    - Convergence plots

FORMULATONS

#---------------------------------------------------------------------

We assume:

$$
u = \frac{\partial \psi}{\partial y}
\quad,\quad
v = -\frac{\partial \psi}{\partial x}
$$


The loss is minimized by updating ψ in the direction of the gradient:

$$
\psi \leftarrow \psi - \alpha \cdot \nabla \mathcal{J}(\psi)
$$


using the adjoint method:

$$
\nabla \mathcal{J}(\psi) = - \frac{\partial \lambda_u}{\partial y} + \frac{\partial \lambda_v}{\partial x}
\quad \text{where} \quad
\lambda_u = u - u_{\text{target}}, \quad \lambda_v = v - v_{\text{target}}
$$


EXTENSION

#---------------------------------------------------------------------

- Add support for unstructured grid
- Extend to vector potential formulation
- An independent flow solver based on HHD

LICENSE

#---------------------------------------------------------------------

MIT License. See LICENSE file.

AUTHOR

#---------------------------------------------------------------------

Parth Patel

rp.parth15@gmail.com
  
