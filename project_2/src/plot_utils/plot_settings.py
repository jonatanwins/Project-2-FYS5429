import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the default DPI to specific value (e.g., 300)
plt.rcParams["figure.dpi"] = 300


# Set up for LaTeX rendering
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["figure.titlesize"] = 20
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
# xtick.labelsize : 16
# ytick.labelsize : 16
