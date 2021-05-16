# con2020

Description of the model here

## Installation

Install the module using `pip3`:

```bash
pip3 install --user con2020
```

Or using this repo:

```bash
#clone the repo
git clone https://github.com/gabbyprovan/con2020
cd con2020

#EITHER create a wheel and install (X.X.X is the current version number)
python3 setup.py bdist_wheel
pip3 install --user dist/con2020-X.X.X-py3-none-any.whl

#or directly install using setup.py
python3 setup.py insall --user
```

## Usage

The module contains two functions which can be used to access the model  (`Model` and `ModelCart`) and a test function (`Test`). Both model functions will accept scalars, `numpy` arrays, `lists` and `tuples` of coordinates - they each get converted to a `numpy.ndarray`.

```python
import con2020

#call the model using spherical polar coordinates (System III)
Br,Bt,Bp = con2020.Model(r,theta,phi)

#or using cartesian coordinates (System III)
Bx,By,Bz = con2020.ModelCart(x,y,z)

#for a full list of keywords
con2020.Model?
con2020.ModelCart?

#test the model
con2020.Test()
```

The `Test()` function should produce the following:

![](Test.png)

## References

- Connerney, J. E. P., Timmins, S., Herceg, M., & Joergensen, J. L. (2020). A Jovian magnetodisc model for the Juno era. *Journal of Geophysical Research: Space Physics*, 125, e2020JA028138. https://doi.org/10.1029/2020JA028138
- Connerney, J. E. P., Acuña, M. H., and Ness, N. F. (1981), Modeling the Jovian current sheet and inner magnetosphere, *J. Geophys. Res.*, 86( A10), 8370– 8384, doi:[10.1029/JA086iA10p08370](https://doi.org/10.1029/JA086iA10p08370).
- Edwards T.M., Bunce E.J., Cowley S.W.H. (2001), A note on the vector potential of Connerney et al.'s model of the equatorial current sheet in Jupiter's magnetosphere, *Planetary and Space Science,*49, 1115-1123,https://doi.org/10.1016/S0032-0633(00)00164-1.
  