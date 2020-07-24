## sigmoid
### $$f(x)=\frac{1}{1+e^{-x}}$$
## tanh
### $$tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$
## Relu
### $$f(x)=max(0,x)$$
## Leaky Relu/ PRelu
### $$f(x)=max(ax,x)$$
#### As for PRelu, the $a$ is rectified so that it should be written as $a_i$
## ELU
### $$f(x)=
\begin{cases}
x,& \text{if x>0}\\
a(e^x-1),& \text{otherwise}
\end{cases}$$
## MAXOut
#### never used this activate function