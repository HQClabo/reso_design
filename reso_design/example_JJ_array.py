#%% 
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
# %%
# Plot frequency modulation in magnetic field
fig = reso.plot_f_of_B_in()
fig2 = reso.plot_f_of_B_out()