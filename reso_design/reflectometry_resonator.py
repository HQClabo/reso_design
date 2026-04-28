
import numpy as np
import scipy.constants as const
from dataAnalysis import resonator_fitting as res_fit
from matplotlib import pyplot as plt

class ReflectometryResonator:
    """
    This class models a superconducting resonator coupled to a feedline for reflectometry measurements.
    It handles the electromagnetic properties of the resonator, including inductance, capacitance,
    resistance, and kinetic inductance effects. The class supports calculations of resonance frequency,
    linewidth, quality factors, and reflectometry responses.
    
    Properties:
        L: Total inductance in henries (H). Setting L updates C, if fres is given.
        C: Total capacitance in farads (F). Setting C updates fres, if L is given.
        R: Total circuit resistance in ohms (Ω).
        R_device: Device resistance in ohms (Ω).
        R_spurious: Spurious resistance in ohms (Ω) due to leakage or dissipation. Default is 1 GΩ.
        fres: Resonator resonance frequency in hertz (Hz). Setting fres updates C.
        Z0: Characteristic feedline impedance in ohms (Ω). Default is 50 Ω.
        length: Resonator length in metres (m). Determined from L, Lk and width.
        width: Resonator width in metres (m).
        thickness: Film thickness in metres (m).
        n_squares: Number of squares in the resonator geometry (unitless).
        jc: Critical current density in ampères per square metre (A/m²). Setting jc updates Ic.
        Ic: Critical current in ampères (A). Setting Ic updates jc.
        Lk: Kinetic inductance in henries per square (H/sq).
        R_matching: Device resistance required to reach the matching condition.
        L_matching: Total inductance required to reach the matching condition.
        kappa_i: Internal resonator linewidth in hertz (Hz).
        kappa_c: External resonator linewidth in hertz (Hz).
        kappa: Total resonator linewidth in hertz (Hz).
        power: Input power for the resonator resonance in watts (dBm). Setting power updates nph.
        Kerr: Kerr nonlinearity in hertz (Hz).
    
    Methods:
        reflectivity(freq): Calculate reflectometry response.
        get_visibility(freq, R1, R2): Calculate visibility between two device resistance states.
        max_visibility(visibility, axis): Find maximum visibility value and index.
    """

    def __init__(self):
        self._Z0 = 50  # Characteristic feedline impedance in ohms (default is 50 ohms)
        self._L = None  # Inductance in henries (H)
        self._C = None  # Capacitance in farads (F)
        self._R_device = None  # Device resistance in ohms (Ω)
        self._R_spurious = 1e12  # Spurious resistance in ohms (Ω)
        self._fres = None # Resonator frequency in hertz (Hz)
        self._width = None # Inductor width in metres (m)
        self._length = None # Inductor length in metres (m)
        self._thickness = None  # Inductor film thickness in metres (m)
        self._jc = None # Critical current density in ampères per square metre (A/m²)
        self._Ic = None # Critical current in ampères (A)
        self._Lk = None # Kinetic inductance in henries per square (H/sq)
        self._power = None # Input power for the resonator resonance in watts (dBm)
    
    @property
    def L(self):
        """ Total inductance in henries (H). """
        self._check_variable_set(self._L, 'L')
        return self._L
    @L.setter
    def L(self, value):
        self._L = value
        if self._fres is not None:
            self._C = self._get_C_from_fres()
    
    @property
    def C(self):
        """ Total capacitance in farads (F). """
        self._check_variable_set(self._C, 'C')
        return self._C
    @C.setter
    def C(self, value):
        self._C = value
        if self._L is not None:
            self._fres = self._get_fres_from_LC()
    
    @property
    def R(self):
        """ Total circuit resistance in ohms (Ω). """
        return 1/(1/self._R_device + 1/self._R_spurious)
    
    @property
    def R_device(self):
        """ Device resistance in ohms (Ω). """
        self._check_variable_set(self._R_device, 'R_device')
        return self._R_device
    @R_device.setter
    def R_device(self, value):
        self._R_device = value

    @property
    def R_spurious(self):
        """ Spurious resistance in ohms (Ω), due to leakage or dissipation. """
        self._check_variable_set(self._R_spurious, 'R_spurious')
        return self._R_spurious
    @R_spurious.setter
    def R_spurious(self, value):
        self._R_spurious = value

    @property
    def fres(self):
        """ Resonance frequency in hertz (Hz). """
        self._check_variable_set(self._fres, 'fres')
        return self._fres
    @fres.setter
    def fres(self, value):
        self._fres = value
        self._C = self._get_C_from_fres()
    
    @property
    def Z0(self):
        """ Characteristic feedline impedance in ohms (Ω). """
        return self._Z0
    @Z0.setter
    def Z0(self, value):
        self._Z0 = value
    
    @property
    def length(self):
        """ Resonator length in metres (m). """
        return self.L / self.Lk * self.width
    @length.setter
    def length(self, value):
        self.L = value * self.Lk * self.width

    @property
    def width(self):
        """ Resonator width in metres (m). """
        self._check_variable_set(self._width, 'width')
        return self._width
    @width.setter
    def width(self, value):
        self._width = value
    
    @property
    def thickness(self):
        """ Film thickness in metres (m). """
        self._check_variable_set(self._thickness, 'thickness')
        return self._thickness
    @thickness.setter
    def thickness(self, value):
        self._thickness = value

    @property
    def n_squares(self):
        """ Number of squares in the resonator geometry (unitless). """
        return self.length / self.width
    @n_squares.setter
    def n_squares(self, value):
        self.L = value * self.Lk
    
    @property
    def jc(self):
        """ Critical current density in ampères per square metre (A/m²). """
        self._check_variable_set(self._jc, 'jc')
        return self._jc
    @jc.setter
    def jc(self, value):
        self._jc = value
        self._Ic = self._jc*self.width*self.thickness
    
    @property
    def Ic(self):
        """ Critical current in ampères (A). """
        self._check_variable_set(self._Ic, 'Ic')
        return self._Ic
    @Ic.setter
    def Ic(self, value):
        self._Ic = value
        self._jc = self._Ic/(self.width*self.thickness)
    
    @property
    def Lk(self):
        """ Kinetic inductance in henries per square (H/sq). """
        self._check_variable_set(self._Lk, 'Lk')
        return self._Lk
    @Lk.setter
    def Lk(self, value):
        self._Lk = value

    @property
    def R_matching(self):
        """ Device resistance required to reach the matching condition in ohms (Ω). """
        return self.L / (self.C * self.Z0)
    
    @property
    def L_matching(self):
        """ Total inductance required to reach the matching condition in henries (H). """
        return self.C * self.Z0 * self.R
    
    @property
    def Q_i(self):
        """ Total resonator linewidth in hertz (Hz). """
        Z = float(np.sqrt(self.L / self.C))
        return self.R / Z
    
    @property
    def Q_c(self):
        """ Total resonator linewidth in hertz (Hz). """
        Z = float(np.sqrt(self.L / self.C))
        return Z / self.Z0
    
    @property
    def Q(self):
        """ Total resonator linewidth in hertz (Hz). """
        return 1/(1/self.Q_i + 1/self.Q_c)

    @property
    def kappa_i(self):
        """ Internal resonator linewidth in hertz (Hz). """
        return self.fres / self.Q_i
    
    @property
    def kappa_c(self):
        """ External resonator linewidth in hertz (Hz). """
        return self.fres / self.Q_c
    
    @property
    def kappa(self):
        """ Total resonator linewidth in hertz (Hz). """
        return self.fres / self.Q
    
    @property
    def power(self):
        """ Input power for the resonator resonance in watts (dBm). """
        self._check_variable_set(self._power, 'power')
        return self._power
    @power.setter
    def power(self, value):
        self._power = value

    @property
    def Kerr(self):
        """ Kerr nonlinearity in hertz (Hz). """
        nonlinearity = -3/8 * const.hbar*(2*np.pi*self.fres)**2 / (self.L * self.Ic**2)
        return nonlinearity

    def reflectivity(self, freq, power=None, nonlinear=False):
        """
        Calculate the reflectometry response at given frequencies.
        
        Parameters:
            freq (array-like): Frequencies in hertz (Hz).
        
        Returns:
            array-like: Reflectometry response.
        """
        
        if nonlinear:
            if power is None:
                power = self.power
            return res_fit.S11_resonator_reflection_nonlinear(fdrive=freq, power=power, fr=self.fres, kappa=self.kappa, kappa_c=self.kappa_c, a=1, alpha=0, delay=0, Kerr=self.Kerr)
        else:
            return res_fit.S11_resonator_reflection(fdrive=freq, fr=self.fres, kappa=self.kappa, kappa_c=self.kappa_c, a=1, alpha=0, delay=0)
        
    def plot_S11(self, freq, power=None, nonlinear=False, **kwargs):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(12, 4)
        S11 = self.reflectivity(freq, power, nonlinear)

        axes[0].plot(freq, 20*np.log10(np.abs(S11)))
        axes[0].set_xlabel("Freq (Hz)")
        axes[0].set_ylabel("$|S_{11}|$ (dB)")

        axes[1].plot(freq, np.angle(S11, deg=True))
        axes[1].set_xlabel("Freq (Hz)")
        axes[1].set_ylabel("Arg($S_{11}$) (deg)")

        return fig, axes
        
    
    def get_visibility(self, freq, R1, R2, power=None, nonlinear=False):
        """
        Calculate visibility as the separation in IQ plane of the reflectometry signal
        for two different deviceresistance states.
        The original device resistance is restored after the calculation.

        Parameters:
            freq (float): Frequencies in hertz (Hz).
            R1 (float): First device resistance in ohms (Ω).
            R2 (float): Second device resistance in ohms (Ω).

        Returns:
            float: Ratio of reflectance.
        """
        if power is None:
            power = self.power
        R_before = self.R_device
        self.R_device = R1
        S11_R1 = self.reflectivity(freq, power, nonlinear)
        self.R_device = R2
        S11_R2 = self.reflectivity(freq, power, nonlinear)
        visibility = (S11_R1.real - S11_R2.real) + 1j*(S11_R1.imag - S11_R2.imag)
        self.R_device = R_before
        return np.abs(visibility)
    
    def max_visibility(self, visibility, axis=0):
        """
        Calculate the maximum reflectometry visibility, i.e. the separation of two given signals in the IQ plane.

        Parameters:
            visibility (array-like): Difference in the IQ plane between two device resistance states,
                                     calculated using get_visibility.
            axis (int): Axis along which to find the maximum. Default is 0.

        Returns:
            tuple: (max_visibility_value, max_visibility_index)
        """
        max_idx = np.argmax(np.abs(visibility), axis=axis)
        if len(visibility.shape) > 1:
            if axis == 0:
                max_val = visibility[max_idx, range(len(max_idx))]
            else:
                max_val = visibility[range(len(max_idx)), max_idx]
        else:
            max_val = visibility[max_idx]
        return max_val, max_idx

    def _reflectivity_alt(self, freq, R_device=None):
        """
        Calculate the reflectometry response for a range of resistances.

        Parameters:
            freq (array-like): Frequencies in hertz (Hz).
            R_device (float): Device resistance in ohms (Ω). If None, uses self.R.

        Returns:
            array-like: Reflectometry responses for each frequency.
        """
        omega = 2 * np.pi * freq
        if R_device is not None:
            R = 1/(1/self._R_spurious + 1/R_device)
        else:
            R = self.R

        Z_L = 1j * omega * self.L
        Z_C = 1 / (1j * omega * self.C)
        Z_R = R
        Z_total = Z_L + 1/(1/Z_C + 1/Z_R)
        S11 = (Z_total - self.Z0) / (Z_total + self.Z0)
        return S11
    
    def _get_C_from_fres(self):
        return 1 / ((2 * np.pi * self.fres) ** 2 * self.L)
    
    def _get_fres_from_LC(self):
        return 1 / (2 * np.pi * float(np.sqrt(self.L * self.C)))

    def _check_variable_set(self, var, name):
        if var is None:
            raise ValueError(f"{name} is not set. Please set it before proceeding.")