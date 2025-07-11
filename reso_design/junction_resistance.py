import numpy as np
from matplotlib import pyplot as plt
import lmfit

def model_Rtot(N, R_unit, C):
    return C + N*R_unit

def fit_R_vs_N(N_values, R_values, w=None, l_J=None, l_S=None, plot=False):
    """
    Fit the resistance of a junction array vs the number of junctions to find the resistance per unit.
    If the junction geometry is given, calculate also the normalized resistance*area.

    Args: 
        N_values: Array containing the numbers of junctions of the arrays
        R_values: Array containing the measured RT resistance of the arrays (in Ohm)
        w: Width of the junction array
        l_J: Length of the junction
        l_S: Length of the spurious junction

    Returns: dictionary containing keys "R_per_unit", "R_times_area", "fit_report".
    """

    model = lmfit.Model(model_Rtot)
    params = lmfit.Parameters()
    params.add('R_unit', value=1e3, vary=True)
    params.add('C', 0, min=0, vary=True)

    result = model.fit(R_values, params, N=N_values)
    fitted_data = model.eval(result.params, N=np.array(N_values))

    R_unit_fit = result.params.valuesdict()['R_unit']
    C_fit = result.params.valuesdict()['C']

    print(f"------------------------------ Fit result -----------------------------------")
    print(f"Resistance per unit (real + spurious junction): {R_unit_fit:.0f} Ohm")
    print(f"Contact resistance: {C_fit:.0f} Ohm")
    
    if w and l_J and l_S:
        A_J = w * l_J
        A_S = w * l_S
        R_A = R_unit_fit / (1./A_J + 1./A_S)
        print(f"Resistance times area: {R_A*1e12:.0f} Ohm um^2")

    if plot:
        plt.figure()
        plt.plot(N_values, R_values/1e3, marker='o', linewidth=0, label=f"w = {w*1e9:.0f} nm", c='k')
        plt.plot(N_values, fitted_data/1e3, c='r')
        plt.xlabel("N")
        plt.ylabel("$R_{tot} (k\\Omega)$")
        plt.xticks(N_values)
        # plt.title(f"w = {w*1e9:.0f} nm")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {'R_per_unit': R_unit_fit, 'R_times_area': R_A, "fit_report": result}