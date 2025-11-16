
import psrchive
import pandas as pd
import numpy as np
import joypy
import matplotlib.pyplot as plt

# Step 1: Load the pulsar .ar file
archive = psrchive.Archive_load("J1921+2153_60845.82696231648_175_type_2.gbdnorfix.fits_cleaned.ar")

# Step 2: Prepare the data
archive.dedisperse()         # remove interstellar dispersion
archive.fscrunch()           # combine all frequency channels
archive.tscrunch(1)          # keep all subintegrations (no time scrunching)
archive.remove_baseline()    # remove average background noise

# Step 3: Extract pulse profiles (one per subintegration)
nsub = archive.get_nsubint()         # total subintegrations
nbin = archive.get_nbin()            # bins across one pulse
data = []

for i in range(nsub):
    subint = archive.get_Integration(i)  #as in psrchive python code
    profile = subint.get_Profile(0, 0).get_amps()  
    data.append(profile)

# Step 4: Convert to DataFrame (each row = one subintegration)
df = pd.DataFrame(data)

# Step 5: Make the ridgeline plot
fig, axes = joypy.joyplot(
    df,
    overlap=2,
    colormap=plt.cm.plasma,
    fade=True,
    figsize=(12, 8)
)


# Step 6: Label and show
plt.title("Pulse Phase vs Subintegration (PSR J1921+2153)", fontsize=16, fontweight='bold')
plt.xlabel("Pulse Phase Bins", fontsize=14, fontweight='bold')
plt.ylabel("Subintegrations", fontsize=14, fontweight='bold')

plt.show()
