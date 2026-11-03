# Mathematical syntax highlighting

Existing LaTeX commands retain their backslashes. For example,
$\sum_{n=0}^{\infty} \alpha^n/(n+1) \to \pi$.

The adapter also recognizes some lightweight Markdown expressions. Here are
an operator, a relation, a function, and an arrow:

$$
A \times B \subseteq C,
\qquad \log(2) \le 1,
\qquad f \colon X \to Y.
$$

Numbers, Greek letters, relations, operators, functions, arrows, and
delimiters receive distinct syntax-class colours in the generated PDF.
