---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{pgfplots}
	\usepackage{algpseudocode}
	\usepackage{graphicx}
	\usepackage{tikz}
		\usetikzlibrary{positioning}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
	\tikzset{basic/.style={draw,fill=blue!20,text width=1em,text badly centered}}
	\tikzset{input/.style={basic,circle}}
	\tikzset{weights/.style={basic,rectangle}}
	\tikzset{functions/.style={basic,circle,fill=blue!10}}
title: "Unsupervised Learning"
author: "COMP9417, 23T1"
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
