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
title: "Decision Trees"
author: "COMP9417, 23T1"
theme: "Frankfurt"
colortheme: "beaver"
fonttheme: "professionalfonts"
---

# Decision Trees

## Decision Trees

A tree-like model used for both **regression** and **classification**.

\pause

Advantages:

\pause

- Interpretable
- Useful when used in ensemble learning (we'll come back to this notion)

\pause

Disadvantages:

\pause

- Tend to overfit data
- Often innacurate in their most basic form

## Entropy

Entropy essentially measures the *uncertainty* or *surprise* of a random variable.

We define the entropy for a set $S$,
\begin{align*}
  H(S) = \sum_{x \in X} -p(x) \log p(x)
\end{align*}
where $p(x)$ represents the *proportion* of $x$ in $S$.

\pause
Say we have a random variable $X \sim$ Bernoulli($p$). We can define the entropy of $X$:

\pause
\begin{align*}
  H(x) = -(1-p) \log (1-p) - p \log p
\end{align*}

---

\centering
\includegraphics[scale=0.175]{tut4_entropy.png}

## Gain

To measure the *information* we gain by splitting on an attribute $A$ for a dataset $S$, we define:

\pause
\begin{align*}
  \text{Gain}(S, A) = \text{Current entropy} - \text{Entropy if we split on A} \\
\end{align*}

\pause
If we have a dataset $S$ with a feature $A$,
\begin{align*}
  \text{Gain}(S, A) = H(S) - \sum_{v \in V_{A}} \frac{|S_{v}|}{|S|} H(S_{v}) \\
\end{align*}

# 1 (a, b, c)

Give decision trees to represent the following Boolean functions, where the variables $A,B,C$ and $D$ have values $t$ or $f$, and the class value is either `True` or `False`. Can you observe any effect of the increasing complexity of the functions on the form of their expression as decision trees?

## 1a

(a) $A \land \lnot B$

\pause

```
	A = t:
	|	B = f: True
	|	B = t: False
	A = f: False
```

## 1b

(b) $A \lor [B \land C]$

\pause

```
	A = t: True
	A = f:
	|	B = f: False
	|	B = t:
		|	C = t: True
		|	C = f: False
```

## 1c

(c) $A \quad \text{XOR} \quad B$

\pause

```
	A = t:
	|	B = t: False
	|	B = f: True
	A = f:
	|	B = t: True
	|	B = f: False
```

## 1d

(c) $[A \land B] \lor [C \land D]$

\pause

```
	A = t:
	|	B = t: True
	|	B = f:
		|	C = t:
			|	D = t: True
			|	D = f: False
		|	C = f: False
	A = f:
	|	C = t:
		|	D = t: True
		|	D = f: False
	|	C = f: False
```



# 2 (a, b)

## 2a

:::: columns
::: column
Assume we learn a decision tree to predict class $Y$ given attributes $A, B$ and $C$ from the following training set, with no pruning.

\vspace{0.5em}

What would be the training error for this dataset?

\vspace{2.5em}

\only<2->{We can shortcut this process, the attribute combinations $(0, 1, 1)$ and $(1, 1, 1)$ appear in both classes, therefore we will make an error on $2$ points. So, our error is $2/12$ or $1/6$.}

:::

::: column
\begin{center}
\begin{tabular}{| c | c | c || c |}
	\hline
	$A$ & $B$ & $C$ & $Y$ \\
	\hline
	0 & 0 & 0 & 0 \\
	0 & 0 & 1 & 0 \\
	0 & 0 & 1 & 0 \\
	0 & 1 & 0 & 0 \\
	0 & 1 & 1 & 0 \\
	0 & 1 & 1 & 1 \\
	1 & 0 & 0 & 0 \\
	1 & 0 & 1 & 1 \\
	1 & 1 & 0 & 1 \\
	1 & 1 & 0 & 1 \\
	1 & 1 & 1 & 0 \\
	1 & 1 & 1 & 1 \\
	\hline
\end{tabular}
\end{center}
:::
::::

## 2b

One nice feature of decision tree learners is that they can learn trees to do multi-class classification,i.e., where the problem is to learn to classify each instance into exactly one of $k >2$ classes. Suppose a decision tree is to be learned on an arbitrary set of data where each instance has a discrete class value in one of $k >2$ classes. What is the maximum training set error, expressed as fraction, that any dataset could have?

\pause

\vspace{1.5em}

If we have $k$ classes and $k$ points - in the worst case we have a sample for each class. Then we can only classify one point correctly in the entire dataset so our error is:

\begin{align*}
	1 - \frac{1}{k} = \frac{k-1}{k}
\end{align*}

Where we essentially rewrite the problem in terms of fractions of the training set.

# 3

Look at the examples. Can you guess which attribute(s) will be most predictive of the class?

\begin{center}
\begin{tabular}{| c | c | c | c || c |}
	\hline
	\textbf{species} & \textbf{rebel} & \textbf{age} & \textbf{ability} & \textbf{homeworld} \\
	\hline
	pearl & yes & 6000 & regeneration & no \\
	bismuth & yes & 8000 & regeneration & no \\
	pearl & no & 6000 & weapon-summoning & no \\
	garnet & yes & 5000 & regeneration & no \\
	amethyst & no & 6000 & shapeshifting & no \\
	amethyst & yes & 5000 & shapeshifting & no \\
	garnet & yes & 6000 & weapon-summoning & no \\
	diamond & no & 6000 & regeneration & yes \\
	diamond & no & 8000 & regeneration & yes \\
	amethyst & no & 5000 & shapeshifting & yes \\
	pearl & no & 8000 & shapeshifting & yes \\
	jasper & no & 6000 & weapon-summoning & yes \\
	\hline
\end{tabular}
\end{center}

---

You probably guessed that attributes 3 and 4 were not very predictive of the class, which is true. However, you might be surprised to learn that attribute “species” has higher information gain than attribute “rebel”. Why is this?

\pause

Suppose you are told the following: for attribute “species” the Information Gain is $0.52$ and Split Information is $2.46$, whereas for attribute “rebel” the Information Gain is $0.48$ and Split Information is $0.98$.

\pause

Which attribute would the decision-tree learning algorithm select as the split when using the Gain Ratio criterion instead of Information Gain? Is Gain Ratio a better criterion than Information Gain in this case?

---

We are given:

- $\text{Gain}(\text{species}) = 0.52$ and $\text{SI}(\text{species}) = 2.46$
- $\text{Gain}(\text{rebel}) = 0.48$ and $\text{SI}(\text{rebel}) = 0.98$

\pause

The formula for Gain Ratio is just:
\begin{align*}
	\text{GainRatio}(V) = \frac{\text{Gain}(V)}{\text{SI}(V)}
\end{align*}

so, we have $\text{GainRatio}(\text{species}) = 0.21$ and $\text{GainRatio}(\text{rebel}) = 0.49$.

\pause

Therefore we pick **rebel** in this case as it has a higher gain ratio. It is a better choice as *species* only has high gain due to the number of values it takes.
