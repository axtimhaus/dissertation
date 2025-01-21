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

```python
rC = Symbol("r_C", positive=True)
ds, eta = symbols("ds eta")
```

```python
rCp = sqrt(rC ** 2 + ds ** 2 - 2 * rC * ds * cos(pi - eta))
rCp
```

```python
dphiC = asin(sin(eta) / rCp * ds)
dphiC
```

```python jupyter={"outputs_hidden": false}
limit((rCp - rC) / ds, ds, 0)
```

```python
limit(dphiC / ds, ds, 0)
```

```python

```
