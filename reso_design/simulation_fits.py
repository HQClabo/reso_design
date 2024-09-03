import numpy as np
from matplotlib import pyplot as plt
import lmfit
import csv

####################### Models #########################
def mag(Sdata):
    return 20*np.log10(np.abs(Sdata))

def S21_hanged(f_drive, f_0, k_int, k_ext):
    Delta = f_drive - f_0
    S21 = (Delta - 1j*k_int/2)/(Delta - 1j*(k_int + k_ext)/2)
    return mag(S21)

def S21_transmission(f_drive, f_0, k_int, k_ext):
    Delta = f_drive - f_0
    S21 = (k_ext/2)/(1j*Delta + (k_int + k_ext)/2)
    return mag(S21)

def S11_reflection(f_drive, f_0, k_int, k_ext):
    Delta = f_drive - f_0
    S11 = (1j*Delta + (k_ext - k_int)/2)/(-1j*Delta + (k_int + k_ext)/2)
    return mag(S11)

####################### Auxiliary functions #########################

def val_to_pos(array, val):
	""" Assuming equally spaced values"""
	start = array[0]
	step = array[1] - array[0]
	return int((val - start)/step)
    
def val_to_pos_irregular(array, val):
	"""Assumes sorted array, but with irregular spacings"""
	idx = 0
	while idx < len(array):
		if array[idx] >= val:
			return idx
		idx += 1
	print(f"Val {val} could not be found in array")

def find_HM_points(mag):
    """Find the points in the Sdata at half maximum."""
    idx = 0
    low_idx = 0
    high_idx = 0
    while idx < len(mag):
        if np.isclose(abs(mag[idx]), 3, atol=1e-1):
            if low_idx == 0:
                low_idx = idx
                idx += 5
            else:
                high_idx = idx
                break
        idx += 1
    if idx == len(mag):
        print("Couldn't find the FWHM of the provided Sdata. No point was close enough to 3dB.")
        return None, None
    return low_idx, high_idx

####################### Fitting functions #########################

def fit_scattering(freq, Sdata, model, f0_guess=None, k_ext_guess=None, tan_delta=1e-5, print_result=True, plot_results=True):
    """ Fit the complex scattering parameter with the input-output theory model.

    Args:
        freq: Frequency array (in GHz).
        Sdata: Complex scattering parameter array.
        model: Either "hanged", "transmission" or "reflection". 
        f0_guess: Guess for the resonance frequency.
        k_ext_guess: Guess for the external decay rate.
        tan_delta: Loss tangent provided in Sonnet.
        print_results: Boolean.
        plot_results: Boolean.

    Returns:
        A dictionary with keys "f0", "k_ext", "k_int", "Q_ext", "Q_int"

    """
    # Check that the provided model is correct
    if model not in ["transmission", "reflection", "hanged"]:
        print(f"Error: the provided model is not supported. Check the spelling.")
        raise ValueError
    
    # If not provided, find the guess for f0
    if f0_guess == None:
        f0_guess = freq[np.argmax(np.abs((mag(Sdata))))]

    # Calculate guess for k_int
    Q_int_guess = 1/tan_delta
    k_int_guess = f0_guess/Q_int_guess

    # If not provided, calculate guess for k_ext
    if k_ext_guess == None:
        low_idx, high_idx = find_HM_points(mag(Sdata))
        if low_idx == None or high_idx == None: 
            "No guess could be calculated for k_ext. Please provide the guess yourself."
            raise ValueError
        FWHM = freq[high_idx] - freq[low_idx]
        k_ext_guess = FWHM
    Q_ext_guess = f0_guess/k_ext_guess

    # Restrict range to fit

    # Create parameter object
    params=lmfit.Parameters() # object
    params.add('k_int',value=k_int_guess,vary=True)
    params.add('k_ext',value=k_ext_guess,vary=True)
    params.add('f_0',value=f0_guess,vary=True)

    print("------------------------------------")
    print("Fit guesses")
    print("------------------------------------")
    print(f"f0_guess = {f0_guess:.4f} GHz")
    print(f"k_ext_guess = {k_ext_guess*1e3:.2f} MHz")
    print(f"Q_ext_guess = {Q_ext_guess:.0f}")
    print(f"k_int_guess = {k_int_guess*1e3:.2f} MHz")
    print(f"Q_int_guess = {Q_int_guess:.0f}")


    # Create the lmfit model
    if model == "hanged":
        model_lmfit = lmfit.Model(S21_hanged)
    elif model == "reflection":
        model_lmfit = lmfit.Model(S11_reflection)
    elif model == "transmission":
        model_lmfit = lmfit.Model(S21_transmission)

    # Fit and extract parameters
    fit_result = model_lmfit.fit(mag(Sdata), params, f_drive=freq)
    fitted_data = model_lmfit.eval(fit_result.params, f_drive=freq)
    k_ext_fit = fit_result.params['k_ext'].value
    k_int_fit = fit_result.params['k_int'].value
    f0_fit = fit_result.params['f_0'].value
    Q_int_fit = f0_fit/k_int_fit
    Q_ext_fit = f0_fit/k_ext_fit

    # Print results
    if print_result:
        print("------------------------------------")
        print("Fit results")
        print("------------------------------------")
        print(f"f0 = {f0_fit:.4f} GHz")
        print(f"k_ext = {k_ext_fit*1e3:.2f} MHz")
        print(f"Q_ext = {Q_ext_fit:.0f}")
        print(f"k_int = {k_int_fit*1e3:.3f} MHz")
        print(f"Q_int = {Q_int_fit:.0f}")
        print("------------------------------------")

    # Plot
    if plot_results:
        # Restrict frequency range to plot to 5 FWHM
        k_tot = k_ext_fit + k_int_fit
        f_low = f0_fit - 5*k_tot/2
        f_high = f0_fit + 5*k_tot/2
        idx_low = val_to_pos_irregular(freq, f_low)
        idx_high = val_to_pos_irregular(freq, f_high)
        slice_to_plot = np.s_[idx_low:idx_high]
        # slice_to_plot = np.s_[:]

        fitted_data_x = np.linspace(f_low, f_high, 1000)
        fitted_data = model_lmfit.eval(fit_result.params, f_drive=fitted_data_x)

        plt.figure()
        plt.plot(freq[slice_to_plot], mag(Sdata)[slice_to_plot], '.', color="black", label="Simulation")
        plt.plot(fitted_data_x, fitted_data, color="red", label="Fit")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Mag($S_{21}$)")
        plt.legend()
        plt.grid()
        plt.show()

    return {"f0": f0_fit, "k_ext": k_ext_fit, "k_int": k_int_fit, "Q_ext": Q_ext_fit, "Q_int": Q_int_fit}



def fit_simulation_from_file(simulation_data_file, model, f0_guess=None, k_ext_guess=None, tan_delta=1e-5, print_results=True, plot_results=True):
    """ Fit the complex scattering parameter from a simulation output file. 

    This function assumes that the data was saved from Sonnet using "Output -> S,Y,Z-Parameter file -> Complex format: Real-Imag".

    Args:
        simulation_data_file: Path to the location of simulation output file.
        model: Either "hanged", "transmission" or "reflection". 
        f0_guess: Guess for the resonance frequency.
        k_ext_guess: Guess for the external decay rate.
        tan_delta: Loss tangent provided in Sonnet.
        print_results: Boolean.
        plot_results: Boolean.

    Returns:
        A dictionary with keys "f0", "k_ext", "k_int", "Q_ext", "Q_int" and values from the fit.
        
    """
    col_to_use = (0, 5, 6) if model == "transmission" or model == "hanged" else (0, 1, 2)
    data = np.loadtxt(simulation_data_file, delimiter=',', skiprows=2, usecols=col_to_use)
    freq = data[:,0]
    Sdata = data[:,1] + 1j * data[:,2] 
    return fit_scattering(freq, Sdata, model, f0_guess, k_ext_guess, tan_delta, print_results, plot_results)

