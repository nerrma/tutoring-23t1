---
header-includes: |
	\usepackage{amsmath}
	 \usepackage{fancyhdr}
	 \usepackage{physics}
	 \usepackage{hyperref}
	 \usepackage{graphicx}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
title: "Regression I"
author: "COMP9417, 23T1"
theme: "Frankfurt"
colortheme: "beaver"
fonttheme: "professionalfonts"
---

# Intro

Who am I? \pause
Who are you? \pause

What you'll get from this course:

- Understand the basis of machine learning
- ML algorithms and the math behind them
- Ability to implement these ideas in Python

How to do well:

- Fully understand tut questions from week to week (they pile up)
- Don't be afraid of math or notation, break it all down
- Keep researching


# Linear Regression

Say we're given a task to explain the relationship of the prices of homes based on their size in square meters.

\begin{center}
	\only<2>{\includegraphics[scale=0.4]{tut1_data.png}}
\end{center}

---

Let's try fitting a line of best fit:

\begin{center}
	\includegraphics[scale=0.4]{tut1_fit.png}
\end{center}

\pause

How do we know that this is the line of *best* fit?

---

Let's define our error as

\begin{align*}
	E &= e_1 + e_2 + e_3 + \cdots + e_n \\
	&= \sum_{i=1}^n e_i
\end{align*}

\pause

We can generalise this to a function in nicer form:
\begin{align*}
	L(\hat{y}) &= \sum_{i=1}^n (y_i - \hat{y_i})
\end{align*}

\pause
Something is wrong here.


---

Formally, we define our error/loss function as:

\begin{align*}
	L(\hat{y}) &= \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2 & \text{ a.k.a MSE } \\
	L(w_0, w_1) &= \frac{1}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i)^2 & \text{ by definition }
\end{align*}

The minimum of our loss function w.r.t $w_0$ and $w_1$ will be their optimal values respectively.

# Question 1 (a $\to$ c)

## 1a

Derive the least-squares estimates for the univariate linear regression model.

i.e Solve:
\begin{align*}
	\argmin_{w_0, w_1}& \quad L(w_0, w_1) \\
	\argmin_{w_0, w_1}& \quad \frac{1}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i)^2 \\
\end{align*}

---

First we differentiate $L(w_0, w_1)$ with respect to $w_0$, 

\pause

\begin{align*}
	\frac{\partial L(w_0, w_1)}{\partial w_0} &= -\frac{2}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i) \\
	&= -\frac{2}{n} \left( \sum_{i=1}^n y_i - n w_0 - w_1 \sum_{i=1}^n x_i \right) \\
\end{align*}

\pause

For the minimum, $\frac{\partial L(w_0, w_1)}{\partial w_0} = 0$,
\begin{align*}
	-\frac{2}{n} \left( \sum_{i=1}^n y_i - n w_0 - w_1 \sum_{i=1}^n x_i \right) &= 0\\
\end{align*}

---

\begin{equation}
\begin{aligned}
	\frac{1}{n} \sum_{i=1}^n y_i - w_0 - w_1 \frac{1}{n} \sum_{i=1}^n x_i &= 0\\
	\bar{y} - w_0 - w_1 \bar{x} &= 0\\
	w_0 = \bar{y} - w_1 \bar{x}\\
\end{aligned}
\end{equation}

To find $w_1$, we follow a similar process and use simple simultaneous equations to solve for the final solution.

---

So,
\pause
\begin{align*}
	\frac{\partial L(w_0, w_1)}{\partial w_1} &= -\frac{2}{n} \sum_{i=1}^n x_i (y_i - w_0 - w_1 x_i) \\
	&= -\frac{2}{n} \left( \sum_{i=1}^n x_i y_i - w_0 \sum_{i=1}^n x_i - w_1 \sum_{i=1}^n x_i^2  \right) \\
\end{align*}

\pause
$\frac{\partial L(w_0, w_1)}{\partial w_1} = 0$,
\begin{align*}
	\frac{1}{n} \left( \sum_{i=1}^n x_i y_i -  w_0\sum_{i=1}^n x_i - w_1\sum_{i=1}^n x_i^2  \right) = 0 \\
	\overline{xy} -  w_0 \bar{x} - w_1 \overline{x^2} = 0 \\
\end{align*}

---

\begin{equation}
\begin{aligned}
	\overline{xy} -  w_0 \bar{x} - w_1 \overline{x^2} &= 0 \\
	w_1 &= \frac{\overline{xy} -  w_0 \bar{x}}{\overline{x^2}}\\
\end{aligned}
\end{equation}

\pause

Sub (1) into (2):

\pause
\begin{align*}
	w_1 &= \frac{\overline{xy} -  (\bar{y} - w_1 \bar{x}) \bar{x}}{\overline{x^2}}\\
	w_1 &= \frac{\overline{xy} -  \bar{x}\bar{y} +  w_1 \bar{x}^2}{\overline{x^2}}\\
	w_1 (\frac{\overline{x^2} - \bar{x}^2}{\bar{x}^2}) &= \frac{\overline{xy} -  \bar{x}\bar{y} +  w_1 \bar{x}^2}{\overline{x^2}}\\
	w_1 &= \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2}
\end{align*}

---

Finally, we have

\begin{align*}
	w_1 &= \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} \text{ and } w_0 = \bar{y} - w_1 \bar{x}
\end{align*}


## 1b

**Problem**: Prove $(\bar{x}, \bar{y})$ is on the line.

From 1(a), the equation of our line ($\hat{y} = w_0 + w_1 x$) becomes:

\begin{align*}
	\hat{y} &= \bar{y} - \bar{x} \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} + \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} x \\
\end{align*}

Sub $x = \bar{x}$, 
\begin{align*}
	\hat{y} &= \bar{y} - \bar{x} \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} + \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} \bar{x} \\
	\hat{y} &= \bar{y} & \therefore (\bar{x}, \bar{y}) \text{ is on the line }
\end{align*}

## 1c

Similar to 1a, though take care with the partial derivatives:

\begin{align*}
	\frac{\partial L(w_0, w_1)}{\partial w_0} = -\frac{2}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i) \\
	\frac{\partial L(w_0, w_1)}{\partial w_1} = -\frac{2}{n} \sum_{i=1}^n x_i (y_i - w_0 - w_1 x_i) + 2 \lambda w_1
\end{align*}

---

Final result is:

\begin{align*}
	w_0 = \bar{y} - w_1 \bar{x} \\
	w_1 = \frac{\overline{xy} - \bar{x} \bar{y}}{\overline{x^2} - \bar{x}^2 + \lambda}
\end{align*}

Notice how the coefficients have an inverse relationship with $\lambda$.

# Multiple Linear Regression

Recall the previous problem where we were tasked with finding price patterns of homes using the size of the home. \pause Say we're now given the number of bedrooms in the house, how do we account for this in the model?

\pause
Simple, just add another parameter:

\begin{align*}
	\hat{y} = w_0 + w_1 x_1 + w_2 x_2 
\end{align*}

\pause
What if we're given the year the house was built and the coordinates? Let's say $d$ more features?

---

Let's vectorise our model, say:

$x_i = \begin{bmatrix} 1 \\ x_{i1} \end{bmatrix}$ to represent our input \& the bias ($w_0$)

$y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$ to represent the target variable

$w = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$ to represent the parameters

---

Then, let's define our entire feature set $X$ as:

\begin{equation*}
\begin{aligned}
	X = \begin{bmatrix} 1 & x_{11} \\ 1 & x_{21} \\ \vdots & \vdots \\1 & x_{n1} \end{bmatrix}
\end{aligned}
\end{equation*}

\pause

So, 
\begin{equation*}
\begin{aligned}
	Xw &= \begin{bmatrix} w_0 & w_1 x_{11} \\ w_0 & w_1 x_{21} \\ \vdots & \vdots \\ w_0 & w_1 x_{n1} \end{bmatrix} \\
	\hat{y} &= Xw
\end{aligned}
\end{equation*}

---

Then, what does our error become?

\pause

\begin{align*}
	\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^n (y_i - [Xw]_i)^2
\end{align*}

\pause

Formally,
\begin{align*}
	\mathcal{L}(w) = \frac{1}{n} \norm{y - Xw}^2_2
\end{align*}

\pause

### Squared 2-Norm Identity

For a vector $v$,

\begin{align*}
	\norm{v}_2^2 &= v^T v\\
\end{align*}


--- 

### Vector Calculus

Say we have our weight vector $w$ and a constant vector $c$,
\begin{align*}
	\frac{\partial (c w)}{\partial w} &= c^{T} \\
	\frac{\partial (w^{T} c w)}{\partial w} &= 2cw \\
	\frac{\partial (c w^{2})}{\partial w} &= 2cw \\
\end{align*}

# Question 2 (a $\to$ h)

## 2a

**Problem**: Show that $\mathcal{L}(w) = \frac{1}{n} \norm{y - Xw}^2_2$ has critical point $\hat{w} = (X^T X)^{-1} X^T y$.

To find optimal $w$, solve $\displaystyle \frac{\partial\mathcal{L}(w)}{\partial w} = 0$

\pause

\begin{align*}
	\mathcal{L}(w) &= \frac{1}{n} (y - Xw)^T (y - Xw)\\
	&= \frac{1}{n} \left( y^T y - y^T X w  - w^T X^T y + w^T X^T X w \right) \\
	&= \frac{1}{n} \left( y^T y - 2y^T X w + w^T X^T X w \right) \\
\end{align*}

---

Let's find the derivative w.r.t $w$,
\pause

\begin{align*}
	\frac{\partial\mathcal{L}(w)}{\partial w} &= -\frac{1}{n} (-2 X^T y + 2 X^T X w) \\
\end{align*}

\pause

To solve for $\hat{w}$,

\begin{align*}
	- 2 X^T y + 2 X^T X \hat{w} = 0\\
	\hat{w} = (X^T X)^{-1} X^T y\\
\end{align*}

## 2b
**Problem**: Prove $\hat{w} = (X^T X)^{-1} X^T y$ is a global minimum.

\pause

\begin{align*}
	\nabla_w^2 \mathcal{L}(w) &= \nabla_w (\nabla_w \mathcal{L}(w)) \\
	&= \nabla_w (-2X^T y + 2X^T X w) \\
	&= 2 X^T X
\end{align*}

\pause

So, for a vector $u \in \mathbb{R}^p$, 
\begin{align*}
	u^T (2 X^T X) u &= 2(u^TX^T)(Xu) \\
	&= 2(Xu)^T(Xu) \\
	&= 2\norm{Xu}^2_2 \geq 0 \\
\end{align*}

---

Therefore, $\mathcal{L}$ is convex and $\hat{w}$ is the unique global minimum.

## 2c

$x_i = \begin{bmatrix} 1 \\ x_{i1} \end{bmatrix}$ to represent our input \& the bias ($w_0$)

$y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$ to represent the target variable

$w = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$ to represent the parameters

---

::: columns

:::: column
\begin{equation*}
\begin{aligned}
	X &= \begin{bmatrix} 1 & x_{11} \\ 1 & x_{21} \\ \vdots & \vdots \\1 & x_{n1} \end{bmatrix} \\
	X^T y &= \begin{bmatrix} 1 & 1 &\cdots& 1 \\ x_{11} &x_{21} &\cdots &x_{n1}  \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}\\
	\pause
	&= \begin{bmatrix} n \bar{y} \\ n \overline{xy} \end{bmatrix}
\end{aligned}
\end{equation*}
::::

\pause

:::: column
\begin{equation*}
\begin{aligned}
	X^T X &= \begin{bmatrix} 1 & 1 &\cdots& 1 \\ x_{11} &x_{21} &\cdots &x_{n1}  \end{bmatrix} \begin{bmatrix} 1 & x_{11} \\ 1 & x_{11} \\ \vdots & \vdots \\1 & x_{n1} \end{bmatrix} \\
	\pause
	&= \begin{bmatrix} n & \sum_{i=1}^n x_i \\ \sum_{i=1}^n x_i & \sum_{i=1}^n x_i^2 \end{bmatrix} \\
	\pause
	&= \begin{bmatrix} n & n \bar{x} \\ n \bar{x} & n \overline{x^2} \end{bmatrix}
\end{aligned}
\end{equation*}
::::

:::

---

We have:
\begin{equation*}
\begin{aligned}
	X^T X &= \begin{bmatrix} n & n \bar{x} \\ n \bar{x} & n\overline{x^2} \end{bmatrix} \\
\end{aligned}
\end{equation*}

\pause

Recall the inverse of a matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ is $A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$.

\pause

\begin{equation*}
\begin{aligned}
	(X^T X)^{-1} &= \frac{1}{n^2\overline{x^2} - n^2 \bar{x}^2}\begin{bmatrix} n \overline{x^2} & -n \bar{x} \\ -n \bar{x} & n \end{bmatrix} \\
	&= \frac{1}{n(\overline{x^2} - \bar{x}^2)}\begin{bmatrix} \overline{x^2} & -\bar{x} \\ -\bar{x} & 1 \end{bmatrix} \\
\end{aligned}
\end{equation*}

## 2d

\begin{equation*}
\begin{aligned}
	(X^T X)^{-1} X^T y &= \frac{1}{n(\overline{x^2} - \bar{x}^2)}\begin{bmatrix} \overline{x^2} & -\bar{x} \\ -\bar{x} & 1 \end{bmatrix} \begin{bmatrix} n \bar{y} \\ n \overline{xy} \end{bmatrix} \\
	\pause
	&= \frac{1}{\overline{x^2} - \bar{x}^2}\begin{bmatrix} \overline{x^2}\bar{y} - \bar{x} \overline{xy} \\ \overline{xy} - \bar{x} \bar{y} \end{bmatrix} \\
	\pause
	&= \begin{bmatrix} \bar{y} - \hat{w}_1 \bar{x} \\ \frac{\overline{xy} - \bar{x} \bar{y}}{\overline{x^2} - \bar{x}^2} \end{bmatrix}
\end{aligned}
\end{equation*}

## 2e - Lab

\begin{center}
  Onto Jupyter.
\end{center}


## 2g

MSE$\displaystyle (w) = \argmin_{w} \frac{1}{n} \norm{y - X w}_{2}^{2}$ and SSE$\displaystyle (w) = \argmin_{w} \norm{y - X w}_{2}^{2}$


**i)** Is the minimiser of MSE and SSE the same? 

**ii)** Is the minimum value of MSE and SSE the same?
