import numpy as np
from matplotlib import pyplot as plt
import lmfit
import pandas


def import_resistance_csv(path, unit='Ohm', n_header_lines=0):
    if unit == 'Ohm':
        unit_scaling = 1
    elif unit == 'kOhm':
        unit_scaling = 1e3
    df = pandas.read_csv(path, skiprows=n_header_lines)
    array = df.to_numpy()
    n_JJs = array[:,0]
    resistances = array[:,1::] * unit_scaling
    r_avg = resistances.mean(axis=1)
    r_err = np.sqrt(resistances.var(axis=1))
    return n_JJs, r_avg, r_err

def model_Rtot(N, R_unit, C):
    return C + N*R_unit

def fit_R_vs_N(N_values, R_values, R_errors=None, w=None, l_J=None, l_S=None, plot=False, **kwargs):
    """
    Fit the resistance of a junction array vs the number of junctions to find the resistance per unit.
    If the junction geometry is given, calculate also the normalized resistance*area.

    Args: 
        N_values: Array containing the numbers of junctions of the arrays
        R_values: Array containing the measured RT resistance of the arrays (in Ohm)
        R_errors: Array containing the statistical errors of the resistance (in Ohm)
        w: Width of the junction array
        l_J: Length of the junction
        l_S: Length of the spurious junction

    Returns: dictionary containing keys "R_per_unit", "R_times_area", "fit_report".
    """

    model = lmfit.Model(model_Rtot)
    params = lmfit.Parameters()
    params.add('R_unit', value=1e3, vary=True)
    params.add('C', 0, min=0, vary=True)

    result = model.fit(R_values, params, N=N_values, **kwargs)
    fitted_data = model.eval(result.params, N=np.array(N_values))

    R_unit_fit = result.params.valuesdict()['R_unit']
    C_fit = result.params.valuesdict()['C']

    results_dict = {'R_per_unit': R_unit_fit, "fit_report": result}

    print(f"------------------------------ Fit result -----------------------------------")
    print(f"Resistance per unit (real + spurious junction): {R_unit_fit:.0f} Ohm")
    print(f"Contact resistance: {C_fit:.0f} Ohm")
    
    if w and l_J and l_S:
        A_J = w * l_J
        A_S = w * l_S
        R_A = R_unit_fit / (1./A_J + 1./A_S)
        results_dict['R_times_area'] = R_A
        print(f"Resistance times area: {R_A*1e12:.0f} Ohm um^2")

    if plot:
        label = f"w = {w*1e9:.0f} nm" if w else ''
        plt.figure()
        if R_errors is None:
            plt.plot(N_values, R_values/1e3, marker='o', ls='', label=label, c='k')
        else:
            plt.errorbar(N_values, R_values/1e3, yerr=R_errors/1e3, marker='o', ls='', label=label, c='k')
        plt.plot(N_values, fitted_data/1e3, c='r')
        plt.xlabel("N")
        plt.ylabel("$R_{tot} (k\\Omega)$")
        plt.xticks(N_values)
        # plt.title(f"w = {w*1e9:.0f} nm")
        if w:
            plt.legend()
        plt.tight_layout()
        plt.show()
    
    return results_dict