# Resonator design
Package to help designing resonators

## Installation
Clone this repository on your computer, navigate to the top folder of the repository and execute from the command line:
`pip install -e .`

This will install the package with "edit" mode, i.e. if you change the source codeor pull changes from origin, the package installed on you rPC will always be up to date automatically.

## Example usage

Some example scripts are provided in the reso_design directory.

### The `JJ_array` class
The JJ array class is used to design junction array resonators. Here is how it can be used:

```python
from reso_design import JJ_array

d_bottom = 20e-9 # metal thickness
d_top = 40e-9 # metal thickness
tox = 1e-9        # oxide thickness

w = 0.515e-6      # junction width
l_j = 0.5e-6    # junction length
l_s = 3.0e-6    # spacing between junctions
l_u = 4.5e-6    # unit

gap = 40e-6 - w/2 # spacing from ground plane
H = 525e-6      # substrate thickness
RA = 650e-12    # junction normalized RT resistance in Ohm m^2

N0 = 26          # number of junctions

reso = JJ_array(d_top, d_bottom, tox, w, l_j, l_s, l_u, gap, H, N0, RA)

# Print a summary of all the parameters and physical quantities of the resonator
reso.print()
```

It is futhermore possible to print the frequency modulation in in-plane and out-of-plane magnetic fields:

```python
fig = reso.plot_f_of_B_in()
fig2 = reso.plot_f_of_B_out()
```

## Reflectometry resontaor

### Simple example of a linear resonator
```python
# define tank circuit parameters
res = ReflectometryResonator()
res.L = 2e-6
res.fres = 150e6
res.R_device = 100e3
res.spurious = 200e3

# create frequency array
span = res.kappa * 3
freqs = np.linspace(res.fres-span/2, res.fres+span/2, 2000)

# return array with frequency response of the tank circuit
S11 = res.reflectivity(freqs, power=None, nonlinear=False)
```

### Comparing visibility between two different resistances
```python
# resistance values to compare
r1 = 100e3
r2 = 200e3

# define tank circuit parameters
res = ReflectometryResonator()
res.L = 2e-6
res.fres = 150e6
res.R_device = 100e3
res.spurious = 200e3

# create frequency array
span = res.kappa * 3
freqs = np.linspace(res.fres-span/2, res.fres+span/2, 2000)

# return the distance in IQ plane between two different device resistance values
visibility = res.get_visibility(freqs, r1, r2)
```

### Power dependence of a nonlinear resonator
```python
# define tank circuit parameters
# here, the nonlinearity is determined from Ic and the dimensions of the resonator
res = ReflectometryResonator()
res.Lk = 2320e-12
res.n_squares = 500
res.C = 0.184e-12
res.width = 2.7e-6
res.thickness = 50e-9
res.R_device = 200e3
res.Ic = 40e-6 *res.width/20e-6
res.power = -120

# create frequency and power arrays
span = res.kappa * 3
freqs = np.linspace(res.fres-span/2, res.fres+span/2, 2000)
powers = np.linspace(-120, -90, 101)
freq_stack = np.tile(freqs, (len(powers), 1))
power_stack = np.tile(powers, (len(freqs), 1)).T

# return 2D array with frequency response of the nonlinear tank circuit as a function of frequency and power
S11 = res.reflectivity(freq_stack, power_stack, nonlinear=True)
```