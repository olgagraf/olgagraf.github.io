## Universal Approximation Theorem for Neural Networks

The theory of neural networks revolves around the fact that the neural networks are universal approximators. That it, **for any continuous function <img src="https://render.githubusercontent.com/render/math?math=\large f"> defined on a bounded domain, we can find a neural network that approximates $f$ with an arbitrary degree of accuracy**. However, it is not widely understood why this holds true. Available literature on the topic tends towards two extremes. On one hand, there are simple visual explanations like the one given by Michael Nielsen \href{http://neuralnetworksanddeeplearning.com/chap4.html}{\underline{[1]}}, with plausible, but not mathematically rigorous arguments. On the other hand, there are scientific papers with lengthy proofs which require a solid mathematical background.

In this note, a simple yet mathematically rigorous explanation is presented. In order to read the text, it will suffice to have basic knowledge of calculus and linear algebra. For the sake of simplicity, some statements will be presented without a proof, in this case an intuitive explanation will be provided. However, an overall argument does not give up on rigour and follows a mathematically elegant as well as historically important proof given by George Cybenko in 1989 [2].


<p align="center">
![\huge \int_X f(x)d\mu(x)
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cint_X+f%28x%29d%5Cmu%28x%29%0A)
</p>

![\int_X f(x)d\mu(x)
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cint_X+f%28x%29d%5Cmu%28x%29%0A)

<img src="https://render.githubusercontent.com/render/math?math=\huge e^{i \pi} = -1">


<img src="https://render.githubusercontent.com/render/math?math=F(\mathbf{x})=\sum_{i=1}^{m}\alpha_i\phi(\mathbf{w}_i\mathbf{x}+b_i),">

You can use the [editor on GitHub](https://github.com/olgagraf/olgagraf.github.io/edit/main/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/olgagraf/olgagraf.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
