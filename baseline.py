
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import psrchive

############# MOVING AVERAGE FUNCTION #######################
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

############ FITS FILE WITH PSRCHIVE COMMANDS ######################

filename = "J1921+2153_final_add.ar" #replace with your file any .fits or .ar
ar = psrchive.Archive_load(filename)
ar.dedisperse()
ar.remove_baseline()
ar.tscrunch()
ar.fscrunch()
ar.pscrunch()

############## CREATE 2-PERIOD PROFILE #############################

profile = ar.get_Profile(0, 0, 0).get_amps()
nbin = profile.shape[0]
x_vals = np.arange(2 * nbin)

# Get two periods
profile_2period = np.concatenate([profile, profile])

############## DETECT THE ON-PULSE BINS ############################

peak_bin = np.argmax(profile)
on_pulse_bins_1 = list(range(peak_bin - 3, peak_bin + 4))  # I choose to do peak ±3 bins
on_pulse_bins_2 = [b + nbin for b in on_pulse_bins_1]
on_pulse_bins_2period = on_pulse_bins_1 + on_pulse_bins_2
off_pulse_bins_2period = np.setdiff1d(np.arange(2 * nbin), on_pulse_bins_2period)

############## REPLACE ON-PULSE WITH OFF-PULSE MEAN FOR FITTING ####
# Compute off-pulse mean
off_pulse = profile[np.setdiff1d(np.arange(nbin), on_pulse_bins_1)]
off_pulse_mean = np.mean(off_pulse)

# Replace on-pulse region in 2-period profile with the off-pulse mean
fit_profile_2period = profile_2period.copy()
fit_profile_2period[on_pulse_bins_2period] = off_pulse_mean

############## POLYNOMIAL FIT #######################################

degree = 20
poly = Polynomial.fit(x_vals, fit_profile_2period, deg=degree)
fitted_curve = poly(x_vals)

############## APPLY SELECTIVE SMOOTHING ###########################

window_size = 5  # Can be adjusted

# Step 1: Smooth the entire profile
temp_smoothed = moving_average(profile_2period, window_size)

# Step 2: Replace only off-pulse bins
mask_off = np.ones_like(profile_2period, dtype=bool)
mask_off[on_pulse_bins_2period] = False
smoothed_profile = profile_2period.copy()
smoothed_profile[mask_off] = temp_smoothed[mask_off]


############## FINAL PLOT ##########################################

plt.figure(figsize=(8, 4), dpi=200)

plt.plot(fitted_curve, color='red', linestyle='--')
plt.plot(smoothed_profile, color='green')

plt.title("Pulse Profile of J1921+2153", fontsize=16, weight='bold')
plt.xlabel("Phase Bin", fontsize=14, weight='bold')
plt.ylabel("Flux", fontsize=14, weight='bold')

plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12, weight='bold')

plt.grid(True)
plt.tight_layout()
plt.show()
