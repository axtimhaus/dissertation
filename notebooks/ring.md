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
from sympy import *
```

```python
rc = Symbol("r_C", positive = True)
phic = Symbol("\\phi_C")
rcp = Symbol("\\dot{r_C}")
phicp = Symbol("\\dot{\\phi_C}")
dt = Symbol("\\delta t")
om = Symbol("\\omega")
```

```python
dx = (rc + rcp * dt) * cos(om + phic + phicp * dt) - rc * cos(om + phic)
dx
```

```python
limit(dx / dt, dt, 0)
```

```python
dy = (rc + rcp * dt) * sin(om + phic + phicp * dt) - rc * sin(om + phic)
dy
```

```python
limit(dy / dt, dt, 0)
```

```python

```
