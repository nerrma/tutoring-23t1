---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{pgfplots}
	\usepackage{algpseudocode}
	\usepackage{graphicx}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
title: "Kernel Methods"
author: "COMP9417, 23T1"
theme: "Frankfurt"
colortheme: "beaver"
fonttheme: "professionalfonts"
---

# Kernel Methods


# Primal vs. Dual Algorithms

The *dual* view of a problem is simply just another way to view a problem mathematically.

\pause

Instead of pure parameter based learning (i.e minimising a loss function etc.), dual algorithms introduce **instance-based** learning.

\pause

This is where we 'remember' mistakes in our data and adjust the corresponding weights accordingly.

We then use a *similarity function* or **kernel** in our predictions to weight the influence of the training data on the prediction.

---

:::: columns
::: column
In the primal problem, we typically learn parameters:

\begin{align*}
  \mathbf{w} \in \mathbb{R}^{p}
\end{align*}

meaning we learn parameters for each of the $p$ features in our dataset.
:::
\pause
::: column
In the dual problem, we typically learn parameters:
\begin{align*}
  \alpha_{i} && \text{ for } i \in [1, n]
\end{align*}
meaning we learn parameters for each of the $n$ **data-points**.

\pause

\vspace{0.5cm}

$\alpha_{i}$ represents the *importance* of a data point $(x_{i}, y_{i})$.
:::
::::

---

**What do we mean by importance?**

\pause

\centering
\includegraphics[scale=0.175]{tut3_linearly_separable.png}

## The Dual/Kernel Perceptron

Recall the *primal* perceptron:

:::: columns
::: column
\begin{algorithmic}
  \State $converged \gets 0$
  \While{not $converged$}
  \State $converged \gets 1$
  \For{$x_{i} \in X, y_{i} \in y$}
  \If{$y_{i} w \cdot x_{i} \leq 0$}
  \State $w \gets w + \eta y_{i} x_{i}$
  \State $converged \gets 0$
  \EndIf
  \EndFor
  \EndWhile
\end{algorithmic}
:::
\pause
::: column
If we define the number of iterations the perceptron makes as $K \in \mathbb{N}^{+}$ and assume $\eta = 1$. We can derive an expression for the final weight vector $w^{(K)}$:

\pause

\begin{align*}
  w^{(K)} = \sum_{i=1}^{N} \sum_{j=1}^{K} \mathbf{1}\{y_{i}w^{(j)}x_{i} \leq 0\} y_{i} x_{i}
\end{align*}
:::
::::

---

We can simply our expression and take out the indicator variable:

\begin{align*}
  w^{(K)} &= \sum_{i=1}^{N} \sum_{j=1}^{K} \mathbf{1}\{y_{i}w^{(j)}x_{i} \leq 0\} y_{i} x_{i} \\
   &= \sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}
\end{align*}

where $\alpha_{i}$ is the number of times the perceptron makes a mistake on a data point $(x_{i}, y_{i})$.

---

If we sub in $w^{(K)} = \sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}$. We get the algorithm for the **dual** perceptron.

\vspace{0.5cm}
\pause

\begin{algorithmic}
  \State $converged \gets 0$
  \While{not $converged$}
  \State $converged \gets 1$
  \For{$x_{i} \in X, y_{i} \in y$}
  \If{$y_{i} \sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x_{i} \leq 0$}
  \State $\alpha_{i} \gets \alpha_{i} + 1$
  \State $converged \gets 0$
  \EndIf
  \EndFor
  \EndWhile
\end{algorithmic}

---

### Gram Matrix

The Gram matrix represents the *inner product* of two vectors.

For a dataset $X$ we define $G = X^{T} X$. That is:

\pause

\begin{align*}
  G &= \begin{bmatrix}
        \langle x_{1}, x_{1} \rangle & \langle x_{1}, x_{2}\rangle &  \cdots & \langle x_{1}, x_{n} \rangle \\
        \langle x_{2}, x_{1} \rangle & \langle x_{2}, x_{2}\rangle & \cdots & \langle x_{2}, x_{n} \rangle \\
        \vdots & \vdots & \ddots & \vdots  \\
        \langle x_{n}, x_{1} \rangle & \langle x_{n}, x_{2}\rangle &  \cdots & \langle x_{n}, x_{n} \rangle \\
      \end{bmatrix} \\
  G_{i,j} &= \langle x_{i}, x_{j} \rangle
\end{align*}

# Transformations

How do we go about solving **non-linearly separable** datasets with linear classifiers?

\pause

Project them to higher dimensional spaces through a transformation $\phi : \mathbb{R}^{p} \rightarrow \mathbb{R}^{k}$.

\pause

\vspace{-0.5cm}
\centering
\includegraphics[scale=0.33]{tut5_projection.png}

---

Let's revisit the XOR.

\centering
\includegraphics[scale=0.6]{tut5_xor.png}

---

A solution:

\pause

For our input vectors in the form $\textbf{x} = [x_{1}, x_{2}]^{T}$, use a transformation:
\begin{align*}
  \phi(\mathbf{x}) = \begin{bmatrix} 1 \\ \sqrt{2} x_{1} \\ \sqrt{2} x_{2} \\ x_{1}^{2} \\ x_{2}^{2} \\ \sqrt{2} x_{1}x_{2} \end{bmatrix}
\end{align*}

---

For our dataset,

\pause
\begin{align*}
  \phi\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} 1 \\ \sqrt{2} \\ \sqrt{2} \\ 1 \\ 1 \\ \sqrt{2} \end{bmatrix} &
  \phi\left(\begin{bmatrix} -1 \\ -1 \end{bmatrix}\right) = \begin{bmatrix} 1 \\ -\sqrt{2} \\ -\sqrt{2} \\ 1 \\ 1 \\ \sqrt{2} \end{bmatrix} &&
  \phi\left(\begin{bmatrix} -1 \\ 1\end{bmatrix}\right) = \begin{bmatrix} 1 \\ -\sqrt{2} \\ \sqrt{2} \\ 1 \\ 1 \\ -\sqrt{2} \end{bmatrix} &
  \phi\left(\begin{bmatrix} 1 \\ -1\end{bmatrix}\right) = \begin{bmatrix} 1 \\ \sqrt{2} \\ -\sqrt{2} \\ 1 \\ 1 \\ -\sqrt{2} \end{bmatrix} &
\end{align*}

---

:::: columns
::: column
For the negative class:
\begin{align*}
  \phi\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} \sqrt{2} \\ \sqrt{2} \end{bmatrix} \\
  \phi\left(\begin{bmatrix} -1 \\ -1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} -\sqrt{2} \\ \sqrt{2} \end{bmatrix} &
\end{align*}
:::
::: column
For the positive class:
\begin{align*}
  \phi\left(\begin{bmatrix} -1 \\ 1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} -\sqrt{2} \\ -\sqrt{2} \end{bmatrix} \\
  \phi\left(\begin{bmatrix} 1 \\ -1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} \sqrt{2} \\ -\sqrt{2} \end{bmatrix}
\end{align*}
:::
::::

\pause
\centering
\includegraphics[scale=0.35]{tut5_xor_projected.png}

---

We may have a problem, recall the **dual perceptron**.

\begin{algorithmic}
  \State $converged \gets 0$
  \While{not $converged$}
  \State $converged \gets 1$
  \For{$x_{i} \in X, y_{i} \in y$}
  \only<1>{\If{$y_{i} \sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x_{i} \leq 0$}}
  \only<2>{\If{$y_{i} \sum_{j=1}^{N} \alpha_{j} y_{j} \phi(x_{j}) \cdot \phi(x_{i}) \leq 0$}}
  \State $\alpha_{i} \gets \alpha_{i} + 1$
  \State $converged \gets 0$
  \EndIf
  \EndFor
  \EndWhile
\end{algorithmic}

---

Recall the transformation $\phi : \mathbb{R}^{p} \rightarrow \mathbb{R}^{k}$. \pause For an arbitrarily large $k$,
\begin{align*}
  G &= \begin{bmatrix}
        \langle \phi(x_{1}), \phi(x_{1}) \rangle & \langle \phi(x_{1}), \phi(x_{2})\rangle &  \cdots & \langle \phi(x_{1}), x_{n} \rangle \\
        \langle \phi(x_{2}), \phi(x_{1}) \rangle & \langle \phi(x_{2}), \phi(x_{2})\rangle & \cdots & \langle \phi(x_{2}), \phi(x_{n}) \rangle \\
        \vdots & \vdots & \ddots & \vdots  \\
        \langle \phi(x_{n}), \phi(x_{1}) \rangle & \langle \phi(x_{n}), \phi(x_{2})\rangle & \cdots & \langle \phi(x_{n}), \phi(x_{n}) \rangle \\
      \end{bmatrix} \\
\end{align*}

the Gram matrix becomes far too complex to compute.

# The Kernel Trick

An absolute mathemagical idea which allows us to calculate the values of the Gram matrix for cheap.

Recall the transformation to the XOR data:
\begin{align*}
  \only<1>{\phi(\mathbf{x}) = \begin{bmatrix} 1 \\ \sqrt{2} x_{1} \\ \sqrt{2} x_{2} \\ x_{1}^{2} \\ x_{2}^{2} \\ \sqrt{2} x_{1}x_{2} \end{bmatrix}}
  \only<2>{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) = \begin{bmatrix} 1 \\ \sqrt{2} x_{1} \\ \sqrt{2} x_{2} \\ x_{1}^{2} \\ x_{2}^{2} \\ \sqrt{2} x_{1}x_{2} \end{bmatrix} \begin{bmatrix} 1 \\ \sqrt{2} y_{1} \\ \sqrt{2} y_{2} \\ y_{1}^{2} \\ y_{2}^{2} \\ \sqrt{2} y_{1}y_{2} \end{bmatrix}}
\end{align*}

---

\begin{align*}
  \only<1->{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) &= 1 + 2x_{1} y_{1} + 2 x_{2} y_{2} + x_{1}^{2} y_{1}^{2} + x_{2}^{2} y_{2}^{2} + 2 x_{1} x_{2} y_{1} y_{2}} \\
  \only<2->{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) &= 1 + 2(x_{1} y_{1} + x_{2} y_{2}) + (x_{1} y_{1} + x_{2} y_{2})^{2} }\\
  \only<3->{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) &= (1 + \mathbf{x} \cdot \mathbf{y})^{2}}
\end{align*}

\pause
\pause
\pause

Say we define a *kernel*: $k(\mathbf{x}, \mathbf{y}) = (1 + \mathbf{x} \cdot \mathbf{y})^{2}$

\pause

So our Gram matrix is:

\vspace{-0.5cm}
\begin{align*}
  G &= \begin{bmatrix}
        k(x_{1}, x_{1}) &  k(x_{1}, x_{2})&  \cdots &  k(x_{1}, x_{n}) \\
        k(x_{2}, x_{1}) &  k(x_{2}, x_{2})&  \cdots &  k(x_{2}, x_{n}) \\
        \vdots & \vdots & \ddots & \vdots  \\
        k(x_{n}, x_{1}) &  k(x_{n}, x_{2}) &  \cdots &  k(x_{n}, x_{n}) \\
      \end{bmatrix} \\
\end{align*}

\pause
\vspace{-0.7cm}
**Why is this useful?**

# Support Vector Machines

\centering
\includegraphics[scale=0.5]{tut5_svm_plot.png}

---

The basic SVM is a linear classifier defined by:

\begin{align*}
  \argmin_{w, t} \frac{1}{2} \norm{w}^{2} && \text{subject to } y_{i} (\langle x_{i}, w \rangle - t) \geq m
\end{align*}

where $t$ is the line's intercept, and we a consider a margin $m$. Typically, we'll see $m=1$ for a standardised dataset.

\pause

This formulation means that we find the **maximal margin** classifier for the dataset.

## Aside: Lagrangian Dual Problem

Say we have a problem as follows:

\begin{align*}
  \max_{x, y} xy && \text{subject to } x + y = 4
\end{align*}

we can also consider the constraint as $x + y - 4 = 0$.

\pause

We can set up the Lagrangian dual and *move* the constraint into the function itself:
\begin{align*}
  \Lambda(x, y, \lambda) = xy + \lambda(x+y-4)
\end{align*}

\pause

To solve this, we can calculate $\frac{\partial L}{\partial x}$, $\frac{\partial L}{\partial y}$ and $\frac{\partial L}{\partial \lambda}$ and solve the remaining system of equations.

## The General Form of a Dual Problem

If we have a problem:
\begin{align*}
  \argmin_{x} f(x) \\
  \text{subject to } &g_{i}(x) \leq 0, && i \in \{1, \ldots, n\} \\
\end{align*}

\pause

The general *dual* problem is:
\begin{align*}
  \Lambda(\mathbf{x}, \mathbf{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^{n} \lambda_{i} g_{i}(x_{i})
\end{align*}

## The Dual Problem for SVM

If we take the general SVM problem ($m = 1$):
\begin{align*}
  \only<1>{ \argmin_{w, t} \frac{1}{2} \norm{w}^{2} && \text{subject to } y_{i} (\langle x_{i}, w \rangle - t) \geq 1}
  \only<2->{ \argmin_{w, t} \frac{1}{2} \norm{w}^{2} && \text{subject to } y_{i} (\langle x_{i}, w \rangle - t) - 1 \geq 0}
\end{align*}

\pause
\pause

From the general form, we can take the vector $\alpha$ to form the dual problem:
\begin{align*}
  \Lambda(w, t, \alpha) = \frac{1}{2} \norm{w}^{2} + \left(-\sum_{i=1}^{n} \alpha_{i} y_{i} (\langle x_{i}, w \rangle - t)- 1) \right)
\end{align*}

---

\begin{align*}
  \only<1->{\Lambda(w, t, \alpha) &= \frac{1}{2} \norm{w}^{2} + \left(-\sum_{i=1}^{n} \alpha_{i} y_{i} (\langle x_{i}, w \rangle - t)- 1) \right) } \\
  \only<2->{\Lambda(w, t, \alpha) &= \frac{1}{2} \norm{w}^{2} -\sum_{i=1}^{n} \alpha_{i} y_{i} (w \cdot x_{i}) + t \sum_{i=1}^{n} \alpha_{i} y_{i}+ \sum_{i=1}^{n} \alpha_{i}} \\
  \only<3->{\Lambda(w, t, \alpha) &= \frac{1}{2} \norm{w}^{2} - w \cdot \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i} + t \sum_{i=1}^{n}\alpha_{i} y_{i}+ \sum_{i=1}^{n} \alpha_{i} } \\
\end{align*}

---

Let's try and optimise the Lagrangian $\Lambda$ w.r.t $w$,

\begin{align*}
  \only<1->{\Lambda(w, t, \alpha) &= \frac{1}{2} \norm{w}^{2} - w \cdot \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i} + t \sum_{i=1}^{n}\alpha_{i} y_{i}+ \sum_{i=1}^{n} \alpha_{i} } \\
  \only<2->{\frac{\partial \Lambda}{\partial w} &= \frac{1}{2} 2 w - \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}} \\
  \only<3->{&= w - \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}} \\
  \only<4->{\text{We can see that at } \frac{\partial \Lambda}{\partial w} = 0 \\
  w &= \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}} \\
\end{align*}

---

Repeating a similar process for $t$,

\begin{align*}
  \only<1->{\Lambda(w, t, \alpha) &= \frac{1}{2} \norm{w}^{2} - w \cdot \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i} + t \sum_{i=1}^{n}\alpha_{i} y_{i}+ \sum_{i=1}^{n} \alpha_{i} } \\
  \only<2->{\frac{\partial \Lambda}{\partial t} &=  \sum_{i=1}^{n} \alpha_{i} y_{i}} \\
  \only<3->{\text{We can see that at } \frac{\partial \Lambda}{\partial t} = 0 \\
  \sum_{i=1}^{n} \alpha_{i} y_{i} &= 0} \\
\end{align*}

## The Dual Problem for SVM

We've derived that for an optimal solution, $\sum_{i=1}^{n} \alpha_{i} y_{i} = 0$ and $w = \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}$

\begin{align*}
  \only<1->{\Lambda(w, t, \alpha) &= \frac{1}{2} \norm{w}^{2} - w \cdot \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i} + t \sum_{i=1}^{n}\alpha_{i} y_{i}+ \sum_{i=1}^{n} \alpha_{i} } \\
  \only<2->{\Lambda(w, \alpha) &= \frac{1}{2} w^{T} w - w^{T} w + \sum_{i=1}^{n} \alpha_{i} } \\
  \only<3->{\Lambda(w, \alpha) &= -\frac{1}{2} w^{T} w + \sum_{i=1}^{n} \alpha_{i} } \\
  \only<4->{\Lambda(\mathbf{\alpha}) &= -\frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} (x_{i} \cdot x_{j}) + \sum_{i=1}^{n} \alpha_{i} } \\
\end{align*}

---

Our final problem now has relaxed constraints:

\begin{align*}
  \Lambda(\mathbf{\alpha}) &= -\frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} (x_{i} \cdot x_{j}) + \sum_{i=1}^{n} \alpha_{i} \\
  \text{subject to } &\sum_{i=1}^{n} \alpha_{i}y_{i} = 0 \\
   &\alpha_{i} \geq 0 \text{ for } i = 1, \ldots, n\\
\end{align*}

# Question 7

Given data $\mathbf{X}$ and targets $\mathbf{y}$, with transformed data $\mathbf{X'}$.

\begin{align*}
  \mathbf{X} &= \begin{bmatrix} 1 & 3 \\ 2 & 1 \\ 0 & 1 \end{bmatrix} \quad \quad
  \mathbf{X'} = \begin{bmatrix} 1 & 3 \\ 2 & 1 \\ 0 & -1 \end{bmatrix} \\
  \mathbf{y} &= \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix} \\
\end{align*}

solve the SVM problem by hand.

---

The steps given are:

1. Set up the Gram matrix for labelled data
2. Set up the expression to be minimised
3. Take partial derivatives
4. Set to zero and solve for each multiplier
5. Solve for $w$
6. Solve for $t$
7. Solve for $m$

---

1. Set up the Gram matrix for labelled data

The Gram matrix is just the product $\mathbf{X'} (\mathbf{X'})^T$.

\begin{align*}
  \mathbf{X'} (\mathbf{X'})^T &= \begin{bmatrix} 1 & 3 \\ 2 & 1 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 0 \\ 3 & 1 & -1 \end{bmatrix} \\
  \only<2->{&= \begin{bmatrix} 10 & 5 & -1 \\ 5 & 5 & -1 \\ -3 & -1 & 1 \end{bmatrix}} \\
\end{align*}

---

2. Set up the expression to be minimised

Recall the dual problem for the SVM:

\begin{align*}
  \only<1>{\argmin_{\alpha_1, \ldots, \alpha_n} -\frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} (x_{i} \cdot x_{j}) + \sum_{i=1}^{n} \alpha_{i}} \\
  \only<2->{\argmin_{\alpha_1, \alpha_2, \alpha_3} -\frac{1}{2} \sum_{i=1}^{3}\sum_{j=1}^{3} \alpha_{i} \alpha_{j} y_{i} y_{j} \mathbf{G}[i,j] + \sum_{i=1}^{3} \alpha_{i}} \\
  \only<1>{\text{subject to } &\sum_{i=1}^{n} \alpha_{i}y_{i} = 0 \\
   &\alpha_{i} \geq 0 \text{ for } i = 1, \ldots, n\\}
  \only<2->{\text{subject to } &\sum_{i=1}^{3} \alpha_{i}y_{i} = 0 \\
   &\alpha_{i} \geq 0 \text{ for } i = 1, \ldots, 3\\}
\end{align*}

---

Recall the gram matrix:
\begin{align*}
  \mathbf{G} &= \begin{bmatrix} 10 & 5 & -1 \\ 5 & 5 & -1 \\ -3 & -1 & 1 \end{bmatrix}
\end{align*}

\begin{align*}
  &\argmin_{\alpha_1, \alpha_2, \alpha_3} -\frac{1}{2} \sum_{i=1}^{3}\sum_{j=1}^{3} \alpha_{i} \alpha_{j} y_{i} y_{j} \mathbf{G}[i,j] + \sum_{i=1}^{3} \alpha_{i} \\
  \only<2->{&\argmin_{\alpha_1, \alpha_2, \alpha_3} -\frac{1}{2} \left(10\alpha_1^2 + 10\alpha_1\alpha_2 - 6\alpha_1\alpha_3 + 5 \alpha_2^2 - 2\alpha_2\alpha_3 + \alpha_3^2\right) + \alpha_1 + \alpha_2 + \alpha_3}
\end{align*}

---

If we look at the constraints ($\sum_i \alpha_i y_i = 0$),

\begin{align*}
  \sum_{i=1}^3 \alpha_i y_i &= 0 \\
  \only<2->{\alpha_1 + \alpha_2 - \alpha_3 &= 0 \\}
  \only<2->{\alpha_3 &= \alpha_1 + \alpha_2  \\}
\end{align*}


Therefore if we substitute in $\alpha_3 = \alpha_1 + \alpha_2$, out final maximisation problem becomes:
\begin{align*}
  \only<2>{&\argmin_{\alpha_1, \alpha_2} -\frac{1}{2} \left(10\alpha_1^2 + 10\alpha_1\alpha_2 - 6\alpha_1(\alpha_1 + \alpha_2) + 5 \alpha_2^2 - 2\alpha_2(\alpha_1 + \alpha_2) + (\alpha_1 + \alpha_2)^2\right) + \\
  &\quad \alpha_1 + \alpha_2 + (\alpha_1 + \alpha_2) \\}
  \only<3->{&\argmin_{\alpha_1, \alpha_2} -\frac{1}{2} \left(5\alpha_1^2 + 4\alpha_1\alpha_2 + 3 \alpha_2^2\right) + 2\alpha_1 + 2\alpha_2}
\end{align*}

--- 

3. Take partial derivatives

\begin{align*}
 &\argmin_{\alpha_1, \alpha_2} -\frac{1}{2} \left(5\alpha_1^2 + 4\alpha_1\alpha_2 + 3 \alpha_2^2\right) + 2\alpha_1 + 2\alpha_2
\end{align*}

\pause

\begin{align*}
  \frac{\partial}{\partial \alpha_1} &= -5\alpha_1 - 2\alpha_2 + 2 \\
  \frac{\partial}{\partial \alpha_2} &= -2\alpha_1 - 4\alpha_2 + 2 \\
\end{align*}

---

4. Set to zero and solve for each multiplier

For $\alpha_1$,
\begin{align*}
  -5\alpha_1 - 2\alpha_2 + 2 &= 0\\
  \alpha_1 &= -\frac{(2\alpha_2 - 2)}{5}\\
\end{align*}

\pause

For $\alpha_2$,
\begin{align*}
  -2\alpha_1 - 4\alpha_2 + 2 &= 0\\
  \frac{2}{5}(2\alpha_2 - 2) - 4\alpha_2 + 2 &= 0\\
  2\alpha_2 - 2 - 10\alpha_2 + 5 &= 0\\
  \alpha_2 = \frac{3}{8} \quad\quad \alpha_1 = \frac{1}{4} \quad\quad \alpha_3 = \frac{5}{8}\\
\end{align*}


---

5. Solve for $w$

What did we define $w$ as for the dual problem?

\pause
\begin{align*}
  w = \sum_{i=1}^n \alpha_i y_i \mathbf{x_i}
\end{align*}

\pause

So, in this case:
\begin{align*}
  w &= \frac{1}{4} x_1 + \frac{3}{8} x_2 + \frac{5}{8} x_3 \\
  &= \frac{1}{4} \begin{bmatrix} 1 \\ 3 \end{bmatrix} + \frac{3}{8} \begin{bmatrix} 2 \\ 1 \end{bmatrix} + \frac{5}{8} \begin{bmatrix} 0 \\ 1 \end{bmatrix} \\
  &= \begin{bmatrix} 1 \\ \frac{1}{2} \end{bmatrix}
\end{align*}


---

6. Solve for $t$

The constraint $y_i (\langle w, x_i \rangle - t) = 1$ for all support vectors. We can use the 3rd data point:

\begin{align*}
  y_3(\langle w, x_3\rangle - t) &= 1 \\
  -\left(\frac{1}{2} - t\right) &= 1 \\
  t &= \frac{3}{2} \\
\end{align*}

7. Solve for $m$

\begin{align*}
  m &= \frac{1}{\norm{w}} = \frac{2}{\sqrt{5}} \\
\end{align*}


# Extension: The RBF Kernel

A popular Kernel is the Radial Basis Function kernel, defined below:

\begin{align*}
  K(x, y) = \exp\left(-\frac{\norm{x - y}^{2}}{2 \sigma^{2}}\right) \\
\end{align*}

for scalar values:

\begin{align*}
  K(x, y) = \exp\left(-\frac{(x - y)^{2}}{2 \sigma^{2}}\right) \\
\end{align*}

---

\begin{align*}
  \only<1->{K(x, y) &= \exp\left(\frac{(x - y)^{2}}{2 \sigma^{2}}\right)} \\
  \only<2->{&= \exp\left(\frac{-x^{2} + 2xy - y^{2}}{2 \sigma^{2}}\right)} \\
  \only<3->{&= \exp\left(\frac{-x^{2}}{2 \sigma^{2}}\right)\exp\left(\frac{-y^{2}}{2 \sigma^{2}}\right)\exp\left(\frac{xy}{\sigma^{2}}\right)} \\
  \only<4->{&= \exp\left(\frac{-x^{2}}{2 \sigma^{2}}\right)\exp\left(\frac{-y^{2}}{2 \sigma^{2}}\right)\sum_{i=1}^{\infty}\frac{(xy)^{k}}{\sigma^{2k}k!}} \\
\end{align*}

----

By definition
\begin{align*}
  \langle \phi(x), \phi(y) \rangle&= \exp\left(\frac{-x^{2}}{2 \sigma^{2}}\right)\exp\left(\frac{-y^{2}}{2 \sigma^{2}}\right)\sum_{i=1}^{\infty}\frac{(xy)^{k}}{\sigma^{2k}k!} \\
\end{align*}

So, our basis transformation is:
\begin{align*}
  \phi(x) &= \exp\left(\frac{-x^{2}}{2 \sigma^{2}}\right)\sum_{i=1}^{\infty}\frac{x^{k}}{\sigma^{k}\sqrt{k!}} \\
\end{align*}

*What does this represent?*
\pause
A projection to infinite dimensions!
