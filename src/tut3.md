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
title: "Classification I"
author: "COMP9417, 23T2"
theme: "Frankfurt"
colortheme: "beaver"
fonttheme: "professionalfonts"
---

# Classification

Recall the standard form of a machine learning problem:

- We have 'input' data $X$ and targets/outputs $y$
- Our data can be modelled as $y = f(X)$
- Goal is to find the best approximation for $f$ as $\hat{f}$

Here, $f(x)$ outputs *classes* rather than numeric values.

### Note:

We call a two-class problem a binary classification problem.

## Linearly Separable Datasets

We define a **linearly separable** dataset as one which can be classified in a binary fashion using a hyperplane.

::: columns
:::: column
\vspace{1.25cm}
More simply, if you can classify it by drawing a line through it. Your dataset is linearly separable.
::::
:::: column
\begin{figure}
	\centering
	\includegraphics[scale=0.1]{tut3_linearly_separable.png}
\end{figure}
::::
:::

# Perceptron

Learns weights $w$ for a decision boundary $w^{T} \mathbf{x} = 0$, where $\mathbf{x}$ represents points on the Cartesian plane, not our dataset.

### Key Properties

- The classic perceptron solves only binary classification
- **Always converges** to a solution if the dataset in linearly separable
- Solutions can differ depending on starting weights and learning rate

## The algorithm

For weights $w$ and a learning rate $\eta$.

\centering
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

# Logistic Regression

Often called *logit* **model**. A way for us to use a linear combination $w^{T} x$ to predict probabilities of a binary classification problem.

For a data point $(x_i, y_i)$ the model will predict:
\begin{align*}
  P(y_{i}=1 | x_{i})
\end{align*}
Simply, the probability that the target belongs to class 1 given the datapoint at index $i$.

---

:::: columns
::: column
\vspace{.25cm}

The logistic regression is defined as the following function:
\begin{align*}
  \sigma(w^{T}x_{i}) = \frac{1}{1 + e^{-w^{T}x_{i}}}
\end{align*}

\vspace{1.25cm}
In the basic case where we only have one feature:
\begin{align*}
  \sigma(w^{T}x_{i}) = \frac{1}{1 + e^{-w_{0} - w_{1} x_{i}}}
\end{align*}
:::
::: column
\includegraphics[scale=0.2]{tut3_logit_curve.png}
:::
::::

# 3 (a, b, c)

If we define the binary prediction problem as a probability:
\begin{align*}
  P(y=1 | x) = p(x) \\
\end{align*}

We write the logistic regression prediction as:
\begin{align*}
  \hat{p}(x) = \sigma(\hat{w}^{T}x) \\
  \text{where } \sigma(z) = \frac{1}{1 + e^{-z}} \\
\end{align*}

where we predict the class of an input $x$ to be $1$ if $\hat{p}(x) \geq 0.5$.

## 3a

**What is the role of the sigmoid function here?**

\pause
In a linear model, we can't simply predict probabilities or classes with the classic equation $\hat{p}(x) = \hat{w}^{T} x$.
\pause
The sigmoid $\sigma(z)$ us model probabilities in a valid interval ($[0, 1]$).

## 3b

Consider the statistical view of the binary classification problem $y_{i} | x_{i} \sim \text{Bernoulli}(p_{i}^{*} )$ where $p_{i}^{*}  = \sigma(x_{i}^{T} w)$ is our logistic regression model.

By definition of the Bernoulli:
\begin{align*}
  P(y|x) = p^{y} (1-p)^{1-y}
\end{align*}

So, we can estimate $p$ using MLE:
\pause
\begin{center}
  $\displaystyle \ln L(w) = \ln \left( \prod_{i=1}^{n} P(y_{i} | x_{i}) \right)$
\end{center}
  \pause
\begin{center}
  $\displaystyle = \sum_{i=1}^{n}\ln  P(y_{i} | x_{i})$
\end{center}

---

\begin{center}
  $\displaystyle = \sum_{i=1}^{n}\ln \left( p_{i}^{y_{i}} (1-p_{i})^{1-y_{i}} \right)$
\end{center}
  \pause
\begin{center}
  $\displaystyle = \sum_{i=1}^{n} \left[ y_{i} \ln p_{i} + (1-y_{i}) \ln (1-p_{i}) \right]$
\end{center}
  \pause
\begin{center}
  $\displaystyle = \sum_{i=1}^{n}\left[  y_{i} \ln \left( \sigma(w^{T}x_{i}) \right) + (1-y_{i}) \ln (1-\sigma(w^{T}x_{i})) \right]$
\end{center}
\pause
\begin{center}
  $\displaystyle = \sum_{i=1}^{n}\left[  y_{i} \ln \left( \frac{\sigma(w^{T}x_{i})}{1-\sigma(w^{T}x_{i})} \right) + \ln (1-\sigma(w^{T}x_{i})) \right]$
\end{center}

---

We then solve,
\begin{align*}
	\hat{w}_{\text{MLE}} = \argmax_w \sum_{i=1}^{n}\left[  y_{i} \ln \left( \frac{\sigma(w^{T}x_{i})}{1-\sigma(w^{T}x_{i})} \right) + \ln (1-\sigma(w^{T}x_{i})) \right]
\end{align*}

# 4 (a, b, c)

We have the following optimisation problem,
\begin{align*}
	\hat{w} &= \argmin_w L(w) \\
	&= \argmin_w -\sum_{i=1}^{n}\left[  y_{i} \ln \left( p_i \right) + (1-y_i)\ln (1-p_i) \right]
\end{align*}

where $p_i = \sigma(w^{T}x_{i})$.

## 4a

**Show that $\sigma(z)^\prime = \sigma(z) (1 - \sigma(z))$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$**

\pause

\begin{align*}
	\sigma(z) &= \frac{1}{1 + e^{-z}} \\
	\only<2->{\sigma^\prime(z) &= \frac{-e^{-z}}{(1 + e^{-z})^2} \\}
	\only<3->{&= \frac{-e^{-z} - 1 + 1}{(1 + e^{-z})^2}} \\
	\only<4->{&= \frac{1}{(1 + e^{-z})^2} -\frac{1}{1 + e^{-z}} \\}
	\only<4->{&= \sigma(z) (1 - \sigma(z))}
\end{align*}

---

**Therefore, show that $\frac{dp_i}{dw} = p_i (1-p_i) w_i$**

\pause
\begin{align*}
	\frac{dp_i}{dw} &= \frac{dp_i}{d(w^T x_i)} \frac{d(w^T x_i)}{dw} \\
	&= p_i(1-p_i) \frac{d(w^T x_i)}{dw} \\
	&= p_i(1-p_i)x_i
\end{align*}

## 4b

**Use the previous result to show that $\frac{d \ln p_i}{dw} = (1-p_i)x_i$**

\pause
\begin{align*}
	\frac{d \ln p_i}{dw} &= \frac{d \ln p_i}{d p_i} \frac{dp_i}{dw} \\
	&= \frac{1}{p_i} p_i (1-p_i) x_i \\
	&= (1-p_i) x_i
\end{align*}

## 4c

**Using the previous results compute the gradient $\frac{d L(w)}{dw}$**

\begin{align*}
	\frac{dL(w)}{dw} &= \frac{d}{dw} \left[- \sum_{i=1}^n y_i \ln p_i + (1 - y_i) \ln (1-p_i)\right] \\
	\only<2->{&= - \sum_{i=1}^n \left[y_i \frac{d}{dw} \ln p_i + (1 - y_i) \frac{d}{dw} \ln (1-p_i)\right] \\}
	\only<3->{&= -\sum_{i=1}^n \left[y_i (1-p_i)x_i + (1 - y_i) \left( - \frac{1}{1-p_i} p_i (1-p_i) x_i\right)\right] \\}
	\only<4->{&= -\sum_{i=1}^n( y_i (1-p_i)x_i - (1 - y_i)p_ix_i) \\}
	\only<5->{&= -\sum_{i=1}^n( y_i  - p_i)x_i \\}
\end{align*}
