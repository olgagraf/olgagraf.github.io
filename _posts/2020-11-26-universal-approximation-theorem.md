The theory of neural networks revolves around the fact that the neural networks are universal approximators. That it, **for any continuous function $f$ defined on a bounded domain, we can find a neural network that approximates $f$ with an arbitrary degree of accuracy**.

However, it is not widely understood why this holds true. Available literature on the topic tends towards two extremes. On one hand, there are simple visual explanations like the one given by Michael Nielsen [[1]](http://neuralnetworksanddeeplearning.com/chap4.html), with plausible, but not mathematically rigorous arguments. On the other hand, there are scientific papers with lengthy proofs which require a solid mathematical background.

In this note, a simple yet mathematically rigorous explanation is presented. In order to read the text, it will suffice to have basic knowledge of calculus and linear algebra. For the sake of simplicity, some statements will be presented without a proof, in this case an intuitive explanation will be provided. However, an overall argument does not give up on rigour and follows a mathematically elegant as well as historically important proof given by George Cybenko in 1989 [[2]](https://web.eecs.umich.edu/~cscott/smlrg/approx_by_superposition.pdf).

## The Formal Statement
Let us start with defining our neural network and formalizing the theorem statement.

<div align="center">
<img src="https://raw.githubusercontent.com/olgagraf/olgagraf.github.io/main/assets/images/nn.jpg" height="400">
</div>

<div align="center">
<sup><b>Figure 1.</b> A visual representation of a neural network with 1 hidden layer which approximates some function $f: \mathbb{R}^n\rightarrow \mathbb{R}$ by computing the function $F(\mathbf{x})$.</sup>
</div>

We consider a feedforward network with $n$ neurons in the input layer, $m$ neurons in the hidden layer, a single neuron in the output layer and some activation function $\phi$ (e.g., sigmoid or ReLU). We can write it down in a compact way,

\begin{equation}
F(\mathbf{x})=\sum_{i=1}^{m}\alpha_i\phi(\mathbf{w}_i\mathbf{x}+b_i),
\end{equation}

where $\mathbf{w}_i, \mathbf{x} \in \mathbb{R}^n$, $\alpha_i, b_i \in \mathbb{R}$.

The functions $f:\mathbb{R}^n\rightarrow \mathbb{R}$ that our neural network is able to approximate belong to the space of continuous functions on some bounded domain. Let us consider an $n$-dimensional unit cube, $I_n=[0,1]^n$, and denote the space of continuous functions on $I_n$ by $C(I_n)$.

Now we can formally state the **Universal Approximation Theorem**.

**Theorem.** *Consider a neural network of the form (1) where $\phi$ is sigmoid or ReLU. Then, given any $f\in C(I_n)$ and $\varepsilon>0$, there exists $F(\mathbf{x})$ for which*

\begin{equation}
|F(\mathbf{x})-f(\mathbf{x})|<\varepsilon\hspace{0.5cm} \textrm{for all}\hspace{0.5cm} \mathbf{x}\in I_n.
\end{equation}

We presented this theorem in one of its early forms. Extensions for other activation functions, arbitrary network width and depth as well as different network architectures are available, but fall beyond the scope of this short note.

## Preliminaries

Before we start with the proof, we'll touch upon some mathematical notions that we will use later on.

**Measure**

In mathematics, a measure on a set is a generalization of the concepts such as length, area, or volume, and can be intuitively interpreted as the size of the set. If we define some measure $\mu$ on $X$, we can integrate over $X$ with respect to that measure,

<div align="center">
$\displaystyle\int\nolimits_X f(x)d\mu(x).$
</div>

In order to consistently apply some results from functional analysis, in the proof we will consider the space of all finite, signed regular Borel measures on $I_n$ and denote it by $M(I_n)$. The usual Riemann integral known from calculus is not expressible in terms of the above integral, but it can provide some intuition about the latter.
<!---
with respect to $\mu\in M(I_n)$
-->

**Closure**

The closure of a set $A$ is denoted by $\overline{A}$ and is defined as the union of $A$ and its boundary. E.g., if $A=(a,b)$ is an open interval on $\mathbb{R}$, then $\overline{A}=[a,b]$. If $A=\mathbb{Q}$ is the set of all rational numbers, then $\overline{A}=\mathbb{R}$. We also say that $\mathbb{Q}$ is dense in $\mathbb{R}$.

## The Proof of Universal Approximation Theorem

We can think that a linear combination of neurons in a hidden layer approximates $f$ similarly to how a partial Fourier sum (i.e., finite linear combination of sines and cosines) approximates any periodic function. Now our goal is to figure out why for

\begin{equation}
S:=\mathrm{span}\\{\phi(\mathbf{w}\cdot\mathbf{x}+b)\hspace{0.1cm}|\hspace{0.1cm} \mathbf{w}\in \mathbb{R}^n, b \in\mathbb{R} \\},
\end{equation}

we have $\overline{S}=C(I_n)$, or, in other words, why the linear combination of activation functions can approximate *any* continuous function.

The crucial reason behind why this holds true is the choice of the activation functions. In the first step, we will show that the chosen activation functions possess a certain useful property. In the second step, we will show that this property implies the ability of a neural network to approximate any continuous function.

**Step 1.** We will state the following Lemma without a proof. The interested reader can find the proof for sigmoid function in the original paper by Cybenko and for ReLU in [[3]](http://math.uchicago.edu/~may/REU2018/REUPapers/Guilhoto.pdf).

**Lemma** (Discriminatory property of activation functions)**.** *Let the activation function $\phi$ be sigmoid or ReLU. Then for a measure $\mu\in M(I_n)$,*

\begin{equation}
\int_{I_n} \phi(\mathbf{w}\cdot\mathbf{x}+b)d\mu(\mathbf{x})=0 
\end{equation}

*for all $\mathbf{w}\in \mathbb{R}^n,  b \in \mathbb{R}$ implies that  $\mu=0$.*

In order to get some intuition about this property, let's resort to a simple univariate example with Riemann integration. Consider mapping $\gamma\left(f(x)\right)=f(x)-f\left({}^1{\mskip -2mu/\mskip -0mu}_2\right)$.

<div align="center">
<img src="https://raw.githubusercontent.com/olgagraf/olgagraf.github.io/main/assets/images/example3.jpg" height="275">
</div>

<div align="center">
<sup><b>Figure 2.</b> Sigmoid function $\sigma$ satisfies the discriminatory property, while $\gamma$ does not.</sup>
</div>

In Figure 2 we see that in case of $\gamma(wx+b)$, its integral on a unit interval will be zero for all $w, b\in \mathbb{R}$. This is not true for the sigmoid function $\sigma(wx+b)$. In this case we get zero for all $w, b\in \mathbb{R}$ only if we integrate over an interval of length zero (essentially, analogous to having $\mu=0$).

In some sense, the discriminatory property prevents functions from "losing" the information conveyed in the linear transformation of $\mathbf{x}$.

**Step 2.** Another way of stating that for any $f\in C(I_n)$ there exists $F(\mathbf{x})$ such that $\|F(\mathbf{x})-f(\mathbf{x})\|<\varepsilon$ for all $\mathbf{x}\in I_n$ is to say that the set $S$ of functions of the form $F(\mathbf{x})$ is dense in $C(I_n)$ or, equivalently, $\overline{S}=C(I_n)$.

For the sake of contradiction, assume $\overline{S}\neq C(I_n)$. This assumption allows us to use a core tool in functional analysis, the Hahn-Banach Theorem. We'll combine it with another result from functional analysis, the Riesz Representation Theorem. By these theorems, there exists

\begin{equation}
L(g)=\int_{I_n}g(\mathbf{x})d\mu(\mathbf{x})\hspace{0.3cm} \textrm{for some}\hspace{0.2cm}\mu\in M(I_n)\hspace{0.2cm} \textrm{and for all}\hspace{0.2cm} g\in C(I_n)
\end{equation}

such that

\begin{equation}
L\neq 0,\hspace{0.2cm} \textrm{but}\hspace{0.2cm}L(g)=0 \hspace{0.2cm} \textrm{for any}\hspace{0.2cm} g\in \overline{S}.
\end{equation}

This should intuitively make sense. Let's look at example from Step 1.

Clearly, $G=\\{\gamma(wx+b)\hspace{0.1cm}\|\hspace{0.1cm} w, b \in\mathbb{R} \\}$ forms a proper subspace of $C(I_1)$ and $\overline{G}\neq C(I_1)$. Therefore, $\gamma(wx+b)$ should satisfy the condition imposed by the theorems. As we've seen in our Riemann integral analogy, indeed $\int_{0}^{1} \gamma(wx+b)dx=0$ for all $w, b\in \mathbb{R}$, but $\int_{0}^{1} \gamma(g(x))dx$ might not be zero for arbitrary $g\in C(I_1)$.

Now let's take $\phi(\mathbf{w}\cdot\mathbf{x}+b)\in\overline{S}$. By the discriminatory property, $L(\phi(\mathbf{w}\cdot\mathbf{x}+b))=0$ implies $\mu=0$ and $L(g)=0$ for all $g\in C(I_n)$ which contradicts (6) and our assumption that $\overline{S}\neq C(I_n)$. Therefore, $\overline{S}=C(I_n)$.
<div align="right"> $\square$ </div>

<!---
**Bold** and _Italic_ and `Code` text
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
```markdown
Syntax highlighted code block
```
-->
