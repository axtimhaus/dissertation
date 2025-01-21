---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from sympy import symbols, Symbol, sin, cos, asin, sqrt, limit, pi, plot, simplify
```

<img src="../dissertation/img/model_development/particle_shift_rotation.svg" style="background-color: white"/>

```python
rC = Symbol("r_C", positive=True)
r = Symbol("r^c", positive=True)
dom, phi, phiC = symbols("d\\omega phi^c phi_C^c")
```

```python
ds = r / sin((pi - dom) / 2) * sin(dom)
ds
```

```python jupyter={"outputs_hidden": false}
eta = pi - (phi - phiC) - (pi - dom) / 2
eta
```

```python
rCp = sqrt(rC ** 2 + ds ** 2 + 2 * rC * ds * cos(eta))
rCp
```

```python
dphiC = asin(sin(eta) / rCp * ds)
dphiC
```

```python jupyter={"outputs_hidden": false}
limit((rCp - rC) / dom, dom, 0)
```

```python jupyter={"outputs_hidden": false}
limit(ds / dom, dom, 0) 
```

```python
limit(asin(sin(eta) / rC * ds) / dom, dom, 0)
```

```python
limit(sin(eta) / rC * ds / dom, dom, 0)
```

```python

```

```python

```
