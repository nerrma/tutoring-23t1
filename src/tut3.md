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
author: "COMP9417, 23T1"
theme: "Frankfurt"
colortheme: "beaver"
fonttheme: "professionalfonts"
---

# Bayes Rule

The basic Bayes rule lets us express conditional probabilities in a different way:

\begin{align*}
	P(A | B) = \frac{P(A) \cdot P(B | A)}{P(B)}
\end{align*}

Also, recall the law of total probability, if we have an event $A$ which is conditionally dependent on a sample space $B$, we can calculate the marginal probability $P(A)$:

\begin{align*}
	P(A) = \sum_{i} P(B_i) \cdot P(A | B_i)
\end{align*}

# 1 (a, b, c)

Assume that the probability of a certain disease is 0.01. The probability of testing positive given that a person is infected with the disease is 0.95, and the probability of testing positive given that the person is not infected with the disease is 0.05.

\pause


**a) Calculate the probability of testing positive.**

---

If we define $D$ to represent the disease being present, $T$ to represent a positive test.

From the problem we have the probabilities:

- $P(D) = 0.01$
- $P(T | D) = 0.95$
- $P(T | \lnot D) = 0.05$

we need to calculate $P(T)$.

\pause

We can simply apply the law of total probability:

\begin{align*}
	P(T) &= P(D) \cdot P(T | D) + P(\lnot D) \cdot P(T | \lnot D) \\
	&= 0.01 \cdot 0.95 + (1 - 0.01) \cdot 0.05 \\
	&= 0.059
\end{align*}

---

**b) Calculate the probability of begin infected with the disease, given that the test is positive.**

We need to find $P(D | T)$.

\pause

We can use what we know and apply Bayes rule:

\begin{align*}
	P(D | T) &= \frac{P(T | D) P(D)}{P(T)} \\
	&= \frac{0.95 \times 0.01}{0.059} \\
	&= 0.16
\end{align*}

---

**c)** Now assume that you test the individual a second time, and the test comes back positive (so two tests, two positives). Assume that conditional on having the disease, the outcomes of the two tests are independent, what is the probability that the individual has the disease? (note, conditional independence in this case means that $P(TT|D) =P(T|D)P(T|D)$, and not $P(TT) =P(T)P(T)$.) You may also assume that the test outcomes are conditionally independent given not having the disease.

\pause

We are trying to find $P(D | TT)$.

---

Let's get this into a nicer form,

\pause

\begin{align*}
	P(D | TT) &= \frac{P(TT | D) P(D)}{P(TT)} \\
	&= \frac{P(T | D)^2 P(D)}{P(TT)} \\
\end{align*}

we see that we don't have the value $P(TT)$.

\pause

Apply the law of total probability:

\begin{align*}
	P(TT) &= P(D) \cdot P(TT | D) + P(\lnot D) \cdot P(TT | \lnot D) \\
	&= 0.01 \cdot (0.95)^2 + (1 - 0.01) \cdot (0.05)^2 \\
	&= 0.0115
\end{align*}

---

Now, we can sub our new values in for the answer:

\begin{align*}
	P(D | TT) &= \frac{P(T | D)^2 P(D)}{P(TT)} \\
	&= \frac{(0.95)^2 \times 0.01}{0.0115} \\
	&= 0.7848
\end{align*}

# Naive Bayes Classification

The naive Bayes classifier solves the problem:

\begin{align*}
	\hat{y_i} &= \argmax_{k \in \{1, \ldots, K\}} p(C_k | x_i) \\
	&= \argmax_{k \in \{1, \ldots, K\}} p(C_k) \prod_{i=1}^p p(x_i | C_k)
\end{align*}

we are essentially trying to estimate the class of a data point based on the prior and posterior probabilities estimated from our data.

# 2 (a, b, c, d, e)

a) What is probabilistic classification? How does it differ from non-probabalistic classification methods?

\pause

In probabilistic methods, we estimate probabilities from our dataset and use these to learn a possible distribution for our data. When using probabilities, the most optimal choice of parameters are the parameters which occur with the highest probability/likelihood. In contrast, non-parametric methods mean that we define some empirical 'loss' function to estimate the error of our estimate in comparison to the assumed true pattern of the data. Then, our solution is the one which minimises this loss.

\pause

b) What is the Naive Bayes assumption and why do we need it?

\pause

The naive Bayes assumption is the assumption that our data is conditionally independent $x_i \perp x_j | c_k$ for all $i \neq j$.

---

What does this let us do?

\pause

Remember, we are trying to maximise $p(c_k | \mathbf{x})$, so:
\begin{align*}
	p(c_k | \mathbf{x}) &= \frac{p(c_k) p(\mathbf{x} | c_k)}{p(\mathbf{x})}
	= \frac{p(c_k) p(\mathbf{x} | c_k)}{\sum_{k = 1}^K p(c_k) p(\mathbf{x} | c_k)} 
\end{align*}
once we model the numerator, the denominator can be calculated for the entire dataset.

\pause

So, how do we model the numerator?

\pause

We can use the product rule and decompose the probabilities:
\begin{align*}
	p(\mathbf{x} | c_k) p(c_k) &= p(\mathbf{x}, c_k) \\
	&= p(x_1, \ldots, x_p, c_k) \\
	&= p(x_1 | x_2 \ldots, x_p, c_k) p(x_2, \ldots, x_p, c_k) \\
	&= p(x_1 | x_2 \ldots, x_p, c_k) p(x_2 | x_3, \ldots, x_p, c_k) p(x_3 | x_4, \ldots, x_p, c_k) \cdots p(x_p | c_k) p(c_k)
\end{align*}

---

We have:
\begin{align*}
	p(\mathbf{x} | c_k) p(c_k) &= p(x_1 | x_2 \ldots, x_p, c_k) p(x_2 | x_3, \ldots, x_n, c_k) p(x_3 | x_4, \ldots, x_p, c_k) \cdots p(x_p | c_k) p(c_k)
\end{align*}

How do we apply the naive Bayes assumption?

\pause

\begin{align*}
	p(\mathbf{x} | c_k) p(c_k) &= p(x_1 | c_k) p(x_2 | c_k) p(x_3 | c_k) \cdots p(x_p | c_k) p(c_k) \\
	&= p(c_k) \prod_{i=1}^p p(x_i | c_k)
\end{align*}

so, instead of estimating an $p$ dimensional distribution as we did with conditionals, applying the assumption means that we now have $p$ independent 1-dimensional distributions to estimate.

---

\begin{centering}
	\includegraphics[scale=0.6]{tut3_p2c.png}
\end{centering}

---

\begin{centering}
	\includegraphics[scale=0.6]{tut3_p2c2.png}
\end{centering}

---

To estimate our probabilities:

\begin{align*}
	p_j^k = \frac{\text{no. docs. in class } k \text{ that contain } j}{\text{no. docs. in class } k}
\end{align*}

\pause

\begin{align*}
	p_a^+ = \frac{2}{4} \quad p_a^- = \frac{3}{4} \\
	p_b^+ = \frac{3}{4} \quad p_b^- = \frac{1}{4} \\
	p_c^+ = \frac{1}{4} \quad p_c^- = \frac{1}{4} \\
\end{align*}

---

Now, we need to calculate our probabilities to solve

\begin{align*}
	\hat{c} = \argmax_{k \in \{-, +\}} p(c_k) p(\mathbf{x}^{e} | c_k)
\end{align*}

\pause

For the positive class:
\begin{align*}
	p(c_+) p(\mathbf{x}^{e} | c_+) &= p(c_+) \prod_{j \in \{a, b, c\}} p(x_j = x^{e}_j | c_+) \\
	&= p(c_+) \times p(x_a = 1 | c_+) \times p(x_b = 1 | c_+) \times p(x_c = 0 | c_+) \\
	&= p(c_+) \times p_a^+ \times p_b^+ \times (1 - p_c^+) \\
	&= \frac{1}{2} \times \frac{2}{4} \times \frac{3}{4} \times (1 - \frac{1}{4}) \\
	&= \frac{9}{64}
\end{align*}

---

For the negative case:

\pause

\begin{align*}
	p(c_-) p(\mathbf{x}^{e} | c_+) &= p(c_-) \prod_{j \in \{a, b, c\}} p(x_j = x^{e}_j | c_+) \\
	&= p(c_-) \times p(x_a = 1 | c_-) \times p(x_b = 1 | c_-) \times p(x_c = 0 | c_-) \\
	&= p(c_-) \times p_a^- \times p_b^- \times (1 - p_c^-) \\
	&= \frac{1}{2} \times \frac{3}{4} \times \frac{1}{4} \times (1 - \frac{1}{4}) \\
	&= \frac{9}{128}
\end{align*}

\pause

Therefore, we pick $c+$ as it has a larger probability of occuring.

---

b) Repeat a) but with smoothing.

\pause

How does smoothing work?

\pause

\begin{align*}
	p_j^k = \frac{\text{no. docs. in class } k \text{ that contain } j + 1}{\text{no. docs. in class } k + \text{no. possible values of } x}
\end{align*}

\pause

\begin{align*}
	p_a^+ = \frac{2 + 1}{4 + 2} \quad p_a^- = \frac{3 + 1}{4 + 2} \\
	p_b^+ = \frac{3 + 1}{4 + 2} \quad p_b^- = \frac{1 + 1}{4 + 2} \\
	p_c^+ = \frac{1 + 1}{4 + 2} \quad p_c^- = \frac{1 + 1}{4 + 2}
\end{align*}

We then use these values to again find that $c_+$ is the most probable class.

---

c) The same as a), though now using a multinomial distribution instead of a Bernoulli. For an email  $e=abbdebbcc$.

### Form

The classic multinomial distribution is:
\begin{align*}
	P(X = (x_1, \cdots, x_n)) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{j \in V} \theta_{i}^{x_i}
\end{align*}

Applied to a naive Bayes classifier:
\begin{align*}
	p(\mathbf{x} | c_k) = \frac{n!}{\prod_{j \in V} x_j!} \prod_{j \in V} p(x_j | c_k) ^ {x_j}
\end{align*}

---

:::: columns

::: column

\begin{center}
	\includegraphics[scale=0.6]{tut3_p2c3.png}
\end{center}
:::

::: column
\begin{align*}
	\theta_j^k = \frac{\text{no. of times word } j \text{ appears in class } k}{\text{no. of words that appear in class } k}
\end{align*}

\pause

\begin{align*}
	\theta_a^+ = \frac{5}{17} \quad \theta_a^- = \frac{11}{17} \\
	\theta_b^+ = \frac{9}{17} \quad \theta_b^- = \frac{3}{17} \\
	\theta_c^+ = \frac{3}{17} \quad \theta_c^- = \frac{3}{17}
\end{align*}
:::
::::


---

Our data is $x^e = (1, 4, 2)$, if we substitute our values:

\begin{align*}
	p(c_+) p(\mathbf{x} | c_+) &= p(c_+) \frac{n!}{\prod_{j \in V} x_j!} \prod_{j \in V} p(x_j | c_k) ^ {x_j} \\
	&= \frac{1}{2} \times \frac{7!}{1! \times 4! \times 2!} \left( (\theta_{a}^+)^1 \times (\theta_{b}^+)^4 \times (\theta_{c}^+)^2 \right)  \\
	&= \frac{1}{2} \times \frac{7!}{1! \times 4! \times 2!} \left( \frac{5}{17} \times (\frac{9}{17})^4 \times (\frac{3}{17})^2 \right)  \\
	&= 0.0377
\end{align*}

\pause

We can apply the same logic for the negative case and find that,
\begin{align*}
	p(c_-) p(\mathbf{x} | c_-) &= 0.001
\end{align*}

therefore, $c_+$ is the most probable class to assign to this email.

---

d) Repeat c) but with smoothed probabilities for the multinomial.

\begin{align*}
	\theta_j^k = \frac{\text{no. of times word } j \text{ appears in class } k + 1}{\text{no. of words that appear in class } k + |V|}
\end{align*}

here, $V = \{a, b, c\}$ so the cardinality $|V| = 3$.

\pause

\begin{align*}
	\theta_a^+ = \frac{5 + 1}{17 + 3} \quad \theta_a^- = \frac{11 + 1}{17 + 3} \\
	\theta_b^+ = \frac{9 + 1}{17 + 3} \quad \theta_b^- = \frac{3 + 1}{17 + 3} \\
	\theta_c^+ = \frac{3 + 1}{17 + 3} \quad \theta_c^- = \frac{3 + 1}{17 + 3}
\end{align*}


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

So, our solution is then:

\begin{align*}
  \hat{w} = \argmax_{w} \sum_{i=1}^{n} \left[  y_{i} \ln \left( \frac{\sigma(w^{T}x_{i})}{1-\sigma(w^{T}x_{i})} \right) + \ln (1-\sigma(w^{T}x_{i})) \right]
\end{align*}

we can then solve this using optimisation methods (i.e gradient descent).

---

c) An alternative approach to the logistic regression problem is to view it purely from the optimisation perspective. This requires us to pick a loss function and solve for the corresponding minimizer. Write down the MSE objective for logistic regression and discuss whether you think this loss is appropriate.

The MSE objective would be:

\pause

\begin{align*}
	\hat{w} = \argmin_w \norm{y - \sigma(Xw)}_2^2
\end{align*}

is this appropriate?

\pause

This is not an appropriate choice as $y$ is binary (class 0 or 1) and our prediction is real valued. This means we're comparing real class values with probabilities which doesn't make direct sense. The maximum likelihood derivation using a log-loss is the most intuitive and applies in this case as logistic regression predicts probabilities.
