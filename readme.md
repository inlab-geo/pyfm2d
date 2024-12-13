
## Installation

```
pip install git+https://github.com/inlab-geo/pyfm2d
```


## Wrapping Strategy

The idea is to refactor the main into a subroutine inside a module that contains all
variables used by main as global variables, That is they are moved out of the subroutine.
As a consequence they should have global scope that is they exist even when the main that
is now a subroutine is terminated. Thus they are accessible from Python via to be written 
get and set functions.


## Test

Running `test_fm2d.py` from within the test direcrory will succesfully execute fmm.
