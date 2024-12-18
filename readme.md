
## Installation

```
pip install git+https://github.com/inlab-geo/pyfm2dss
```

## Wrapping Strategy

The idea is to refactor the main into a subroutine inside a module that contains all
variables used by main as global variables, That is they are moved out of the subroutine.
As a consequence they should have global scope that is they exist even when the main that
is now a subroutine is terminated. Thus they are accessible from Python via to be written 
get and set functions.

This has now been completed for the  file specifying the source location tpyically called 
`sources.dat`. 

## Tests

Running `test_fmmin2d.py` from within the test directory will run the orignal program 
turned into a subroutine that can be called from python like the fmm executable it reads 
the files from disk

Running `test_run.py` illustrate how for the reorganised/expanded `fmm2dss.f90` now 
allows in python to read the sources from disk by providing a file name, set them 
from python and get them back from an fmm instance.
