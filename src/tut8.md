---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{tikzsymbols}
	\usepackage{pgfplots}
	\usepackage{algpseudocode}
	\usepackage{graphicx}
	\usepackage{tikz}
		\usetikzlibrary{positioning}
	\graphicspath{ {./images/} }
	\usepackage{tkz-fct}
	\usetikzlibrary{shapes.misc}
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
	\tikzset{basic/.style={draw,fill=blue!20,text width=1em,text badly centered}}
	\tikzset{input/.style={basic,circle}}
	\tikzset{weights/.style={basic,rectangle}}
	\tikzset{functions/.style={basic,circle,fill=blue!10}}
title: "Unsupervised Learning"
author: "COMP9417, 23T2"
theme: "Frankfurt"
colortheme: "beaver"
fonttheme: "professionalfonts"
---

# Unsupervised Learning

Learning without any labels.

For example,

- Cluster analysis (i.e grouping users of a social media, classifying similar events/data without knowing any other information)
- Signal separation (i.e PCA, SVD)

\pause
The content this week is light, so I'll go straight to the lab to explain it.

# The Missing Learning Theory Tut

:::: columns
::: column
\begin{center}
	\includegraphics[scale=0.4]{learn_meme.png}
\end{center}
:::
::: column
\pause
I'll focus on PAC learning and VC dimension, but we also introduce the \textsc{Winnow} algorithms and the *No Free Lunch theorem* in lectures.
:::
::::

# PAC Learning

**P**robably **A**pproximately **C**orrect (PAC) learning is a concept which allows us to quantify whether a learning algorithm will achieve low *true* error.

\pause
We typically see the example of binary classification here and set the problem as follows:

- Observed data $D$ sampled from a true distribution $\mathcal{D}$
- A *concept* $c$ generates the data i.e we have data $D = \{(x_i, c(x_i))\}_{i=1}^m$
- We aim to model the concept using a hypothesis $h$ from a class of hypotheses $H$

\pause

In this setting, we define the true error of a hypothesis as,
\begin{align*}
	\text{Err}_{\mathcal{D}}(h) = \underset{{x \in \mathcal{D}}}{\text{Pr}} (c(x) \neq h(x))
\end{align*}

---

The **version space** of a learning problem is a subspace of $H$ where the training error (denoted $r$) is 0.

\pause

We say a version space is $\epsilon$-exhausted if for all hypotheses in that space, we have less than $\epsilon$ true error i.e $(\forall h \in VS_{H, D}) \text{Err}_{\mathcal{D}}(h) < \epsilon$. 

\pause

**How many training examples do we need to $\epsilon$-exhaust the version space for a problem?**

---

Say $H$ is finite, and $D$ is a sequence of $m$ independent random samples of a concept $c$.

What is the probability of a hypothesis in the version space having an error greater than or equal to $\epsilon$?

\pause

If $h \in VS_{H, D}$,
\begin{align*}
	\only<2->{\text{Pr}(\text{Err}_{\mathcal{D}}(h) \geq \epsilon)}
	\only<3->{&= (1-\epsilon)^m \\}
	\only<4->{\intertext{by definition, }(1-\epsilon)^m &< e^{-\epsilon m}\\}
	\only<5->{\intertext{Therefore, }\text{Pr}(\text{Err}_{\mathcal{D}}(h) \geq \epsilon) &< |H| e^{-\epsilon m} \text{ for all } h \in H}
\end{align*}

---

We can then bound this probability by some $0 \leq \delta \leq 1$.

\begin{align*}
	|H| e^{-\epsilon m} \leq \delta
\end{align*}

\pause

Using simple log laws, we get
\begin{align*}
	m \geq \frac{1}{\epsilon} (\ln(|H|) + \ln(1/\delta))
\end{align*}

\pause
We now have a bound of the number of examples needed to assure that $(\forall h \in VS_{H, D}) \text{Pr}(\text{Err}_{\mathcal{D}}(h) \leq \epsilon) \geq 1 - \delta$. 

---

We say a concept class $C$ is PAC-learnable by a learner $L$ using a hypothesis class $H$ for all $c \in C$ and distributions $\mathcal{D}$ if for all $0 < \epsilon < 1/2$ and $0 < \delta < 1/2$ the learner outputs a hypothesis $h \in H$ such that,
\begin{align*}
	\text{Err}_{\mathcal{D}}(h) \leq \epsilon
\end{align*}
with probability $1 - \delta$. \pause In time that is polynomial in $1/\epsilon$, $1/\delta$, $m$ and $|C|$.

# VC Dimension

How do we measure model 'complexity'? \pause Vapnik and Chervonenkis had the same question.

First, we define a **dichotomy** of a set as a partitioning of that set into two disjoint subsets. \pause

We also say a set is **shattered** by a hypothesis space if for every dichotomy there is a hypothesis from that space which is consistent with that dichotomy.

---

Lots of big words, what does it mean?

\begin{center}
\begin{tikzpicture}
    \tikzstyle{point}=[thick,draw=black,cross out,inner sep=0pt,minimum width=4pt,minimum height=4pt]
    \begin{axis}[
        legend pos=south west,
        axis x line=middle,
        axis y line=middle,
        grid = major,
        width=6cm,
        height=6cm,
        grid style={dashed, gray!30},
        xmin=0,    % start the diagram at this x-coordinate
        xmax=1,    % end   the diagram at this x-coordinate
        ymin=0,    % start the diagram at this y-coordinate
        ymax=1,    % end   the diagram at this y-coordinate
        xlabel=$x$,
        ylabel=$y$,
        tick align=outside,
        minor tick num=-3,
        enlargelimits=true]
      % \addplot[domain=0:2.5, red, thick,samples=20] {-x+2.5};
      \node[point,label={[label distance=0cm,text=red]135:$0$},red] at (axis cs:1,0) {};
      \node[point,label={[label distance=0cm,text=blue]45:$1$},blue] at (axis cs:0,0) {};
      \node[point,label={[label distance=0cm,text=blue]-135:$1$},blue] at (axis cs:1,1) {};
    \end{axis}
\end{tikzpicture}
\end{center}

---

\begin{center}
\begin{tikzpicture}
    \tikzstyle{point}=[thick,draw=black,cross out,inner sep=0pt,minimum width=4pt,minimum height=4pt]
    \begin{axis}[
        legend pos=south west,
        axis x line=middle,
        axis y line=middle,
        grid = major,
        width=6cm,
        height=6cm,
        grid style={dashed, gray!30},
        xmin=0,    % start the diagram at this x-coordinate
        xmax=1,    % end   the diagram at this x-coordinate
        ymin=0,    % start the diagram at this y-coordinate
        ymax=1,    % end   the diagram at this y-coordinate
        xlabel=$x$,
        ylabel=$y$,
        tick align=outside,
        minor tick num=-3,
        enlargelimits=true]
      % \addplot[domain=0:2.5, red, thick,samples=20] {-x+2.5};
      \node[point,label={[label distance=0cm,text=red]135:$0$},red] at (axis cs:1,0) {};
      \node[point,label={[label distance=0cm,text=red]-45:$0$},red] at (axis cs:0,1) {};
      \node[point,label={[label distance=0cm,text=blue]45:$1$},blue] at (axis cs:0,0) {};
      \node[point,label={[label distance=0cm,text=blue]-135:$1$},blue] at (axis cs:1,1) {};
    \end{axis}
\end{tikzpicture}
\end{center}

\pause We can't shatter this dataset with the space of linear classifiers!

---

The VC-Dimension of a hypothesis space is the size of the largest finite subset of an instance space $\mathcal{X}$ which can be shattered by that hypothesis space (typically denoted $VC(H)$). 

\pause

For the previous example we have $VC(H) = 3$, though for more complex hypothesis classes we can have $VC(H) \equiv \infty$.

\pause

We can also generalise the bound of $m$ from PAC-learning to include possibly non-finite hypothesis classes,
\begin{align*}
	m \geq \frac{1}{\epsilon} (4 \log_2 (2/\delta) + 8 VC(H) \log_2(13/\epsilon))
\end{align*}

---

\begin{center}
	That's it for the term, good luck in the exam period! \Laughey[1.4]
	
	Do myExperience, study hard etc. etc.
\end{center}
