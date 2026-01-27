# heat2d

Simple finite element heat equation solver

## Installation

## One time setup

```console
git clone git@github.com:tjfulle/me7540
cd me7540
python3 -m venv venv
source venv/bin/activate
cd Exercises/heat2d
python3 -m pip install -e .
```

## Run an example

```console
python3 -m heat2d
```

Until you finish the redacted sections, the code will raise a `NotImplementedError` error.

To see additional options, execute:

```console
python3 -m heat2d -h
```

## Where to find code to modify

The code to modify is found in [heat.py](./src/heat2d/heat.py).  Look for the two locations in the
function `heat2d` where a `NotImplementedError` error is raised.

## Recommendations

[Assignment 2, Exercise2](./Exercise.pdf) has three parts:

a) Complete the partially redacted code
b) Complete a MMS
c) Complete a pseudo-1d verification

Since the writing of the original assignment, the code has been updated and I now recommend completing Exercise 2 in the following order:

1) Complete the redacted code sections and the psuedo-1d verification problem.  To run just the
verification problem, execute:

   ```console
   python3 -m heat2d verify
   ```

   when you get a color plot showing the temperature evolve from 33˚ on the bottom edge and 366.33
   on the top edge you will have completed this part.
2) Complete part b) above.  You *should not* need to write additional code in `heat2d`.  To run
   just the MMS problem, execute:

   ```console
   python3 -m heat2d mms
   ```

3) Complete part a) above.  To run just this example problem, execute:

   ```console
   python3 -m heat2d example
   ```
