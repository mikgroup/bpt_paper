import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter
import matplotlib
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import sigpy as sp
import csv
import sys
import cfl # For data i/o
from scipy import signal
import data_processing as proc
import run_bpt as run


def load_motion_signals(inpdir, fname, tr=4.4e-3):
    ''' Load motion signals for concept figure '''
    data = np.real(cfl.readcfl(os.path.join(inpdir,fname)))
    t = np.arange(data.shape[0])*tr
    return data, t

def plot_motion_signals(inpdir="./concept", shift=-5):
    ''' Plot respiratory, cardiac, and head signals '''
    trs = [4.4e-3, 8.7e-3, 4.4e-3]
    plt.figure(figsize=(5,10))
    for i, fname in enumerate(["bpt_resp", "bpt_cardiac", "bpt_head"]):
        bpt, t = load_motion_signals(inpdir, fname, tr=trs[i])
        plt.plot(t, proc.normalize(bpt) + i*shift, lw=3)
    plt.axis("off")

def load_kspace(inpdir):
    ''' Load kspace from cfl file '''
    ksp = cfl.readcfl(os.path.join(inpdir, "ksp"))
    return ksp

def calculate_rss(ksp):
    ''' Calculate rss across coils '''
    ksp_f = sp.ifft(ksp, axes=(0,))
    ksp_f_rss = sp.rss(sp.ifft(ksp, axes=(0,)), axes=(-1,))
    return ksp_f_rss

def prepare_data(ksp_f_rss):
    ''' Stack data into a 3D stack '''
    # Collapse into a single time dimension
    tmp = np.transpose(ksp_f_rss[10:240, ...], (1, -1, 0))  # Show only one tone
    ksp_f_r = np.vstack(tmp)  # [N*nph, N, ncoils] - first dim is time dim

    # Number of kspace lines to plot
    Nt, N = ksp_f_r.shape
    Nlines = 50
    inds = np.linspace(0, ksp_f_r.shape[0] - 1, Nlines).astype(int)
    ksp_data = np.abs(ksp_f_r[inds, :].T)

    # Add extra stuff to ksp
    buffer = 26
    ksp_stack = np.empty((ksp_data.shape[0] + buffer, ksp_data.shape[1]))

    # Extra noise
    ksp_stack[:buffer // 2, :] = ksp_data[-buffer // 2:, :]
    ksp_stack[buffer // 2:buffer, :] = ksp_data[-buffer // 2:, :]
    ksp_stack[buffer:, :] = ksp_data

    N_f = N + buffer
    freq_data = 250 / N_f * np.linspace(-N_f / 2, N_f / 2, N_f)[:, None] * np.ones(Nlines)[None, :]  # 256 x Npts

    # time
    t = np.arange(Nlines)

    return ksp_stack, freq_data, t

def plot_kspace(ksp_stack, freq_data, t):
    ''' Create plot as a PolyCollection '''
    # Make vertices
    verts = []
    Nlines = t.shape[0]
    for irad in range(Nlines):
        # Add a zero amplitude at the beginning and the end to get a nice flat bottom on the polygons
        xs = np.concatenate([[freq_data[0, irad]], freq_data[:, irad], [freq_data[-1, irad]]])
        ys = np.concatenate([[0], ksp_stack[:, irad], [0]])
        verts.append(list(zip(xs, ys)))

    # Make PolyCollection
    poly = PolyCollection(verts, facecolors='gray', edgecolors='k')
    poly.set_alpha(0.9)  # Transparency

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(poly, zs=t, zdir='y')

    # Set labels
    ax.set_xlim3d(freq_data.min(), freq_data.max())
    ax.set_ylim3d(t.min(), t.max())
    ax.set_zlim3d(ksp_stack.min(), ksp_stack.max())
    ax.set_xticks(np.linspace(-125, 125, 3).astype(int))
    ax.tick_params(pad=-7, rotation=-10)  # Make numbers closer to ticks
    ax.set_zticks([])  # No amplitude numbers
    ax.set_yticks([])  # No time axis
    ax.view_init(elev=29, azim=-66)

def plot_ksp_f(inpdir):
    ''' Plot 3D ksp vs f figure '''
    # Load kspace
    ksp = load_kspace(inpdir)

    # Calculate rss
    ksp_f_rss = calculate_rss(ksp)

    # Prepare data
    ksp_stack, freq_data, t = prepare_data(ksp_f_rss)

    # Plot kspace
    plot_kspace(ksp_stack, freq_data, t)
    
    
def plot_cardiac_signals(inpdir):
    ''' Plot cardiac signals with overlaid purple points for two coils '''
    # Define BPT and time data
    bpt_cardiac = np.real(cfl.readcfl(os.path.join(inpdir, "bpt_cardiac")))
    tr_card = 8.7e-3  # Seconds
    t_card = np.arange(bpt_cardiac.shape[0]) * tr_card

    # Colors and shift for plot
    colors = ["tab:green", "tab:orange"]
    shift = -8.5

    # Define scattered points to overlay on the plot
    points = (np.arange(0.1, 0.25, 0.05) / (tr_card/2)).astype(int)

    # Plot cardiac signals with points
    plt.figure(figsize=(10, 5.5))
    for i in range(bpt_cardiac.shape[1]):
        plt.plot(t_card, bpt_cardiac[:, i] + i * shift, color=colors[i])
        plt.scatter(t_card[points], bpt_cardiac[points, i] + i * shift, c='darkmagenta', zorder=10, s=70)
    plt.xlim([0, 8])
    plt.yticks([])
    plt.xlabel("Time (s)")

    
def get_H(inpdir, fname):
    ''' Load fields from text file '''
    data = np.loadtxt(os.path.join(inpdir,fname), skiprows=2)
    x,y,z,H = data.T
    return H,x,y,z

def get_H_mult(inpdir, fld_list_sort):
    ''' Load multiplied fields '''
    H_list = []
    for i in range(len(fld_list_sort)):
        H,x,y,z = get_H(inpdir, fld_list_sort[i])
        H_list.append(H)
    H_list = np.array(H_list)

    # Multiply mags
    H_mult = np.array([(H_list[i] * H_list[i+1]) for i in range(1,H_list.shape[0],2)])
    H_mult = np.insert(H_mult, 0, H_list[0,...], axis=0)
    return H_mult, x, y, z

def get_sorted_list(fld_list):
    ''' Sort list of fields by frequency, increasing order '''
    freqs = np.array([float(fld_list[i].split("_")[1]) for i in range(len(fld_list))])
    fld_inds = np.argsort(freqs)

    # Sorted list of folders and freqs
    fld_list_sort = fld_list[fld_inds]
    freqs = freqs[fld_inds]
    return fld_list_sort, freqs

def plot_rect(axs, width=14, height=10, overlap=[6,3], n_rect=3, lw=7):
    ''' Plot rectangular coils on axes '''
    # Rectangular coil overlay
    start_points = [[-3/2*width + overlap[0], -3/2*height + overlap[1]],
                    [-3/2*width + overlap[0], 1/2*height - overlap[1]],
                     [-width/2, -height/2]]
    colors = ["tab:green", "tab:blue", "tab:orange"] # For correct overlapping

    # Plot rectangles
    for j in range(2): # Loop over subplots
        # Vertical lines 
        axs[j].axvline(0, c='w', ls='--')
        axs[j].axvline(10, c='w', ls='--')
        
        for i in range(n_rect): # Loop over ciol
            rgb_color = mcolors.to_rgba(colors[i])
            rect = matplotlib.patches.Rectangle(start_points[i], width, height, color=colors[i])
            
            # Set transparent face color with colored edges and linewidth 
            rect.set_facecolor(list(rgb_color[:-1]) + [0]) # Make transparent
            rect.set_edgecolor(list(rgb_color[:-1]) + [1]) # Make opaque
            rect.set_linewidth(lw)
            
            # Add the patch to the axes
            axs[j].add_patch(rect)
    
    
def plot_fields(H,x,y,z,plane="xy", cmap="jet", marker=".", fig=None, ax=None, figsize=(10,10), title="", xlim=[-0.3,0.3], ylim=[-0.3,0.3], clim=90):
    ''' 2D scatterplot of fields '''
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if plane == "xy":
        x_axis = x
        y_axis = y
    elif plane == "yz":
        x_axis = y
        y_axis = z
    else:
        x_axis = x
        y_axis = z
    clim = [np.percentile(H,0), np.percentile(H,clim)]
    sc = ax.scatter(x_axis*100, y_axis*100, c=H, cmap=cmap, marker=marker, vmin=clim[0], vmax=clim[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

def plot_field_w_coils(inpdir, figsize=(16.7,5), xlim=[-45,35], ylim=[-35,35], lw=7):
    ''' Plot fields at 127.8MHz and 2.4GHz with rectangular coils overlaid '''
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)
    axs = ax.flatten()
    
    # Load fields
    fld_list = np.array([f for f in os.listdir(inpdir) if f.endswith("_xy.fld")])
    fld_list_sort, freqs = get_sorted_list(fld_list)

    # Get multiplied H fields
    H_mult, x, y, z = get_H_mult(inpdir, fld_list_sort)
    H_set = H_mult[[0,-1],...]
    
    # Plot subplot
    for j in range(H_set.shape[0]):
        plot_fields(H_set[j,...],x,y,z,
                    cmap="jet",
                    marker=".",
                    ax=axs[j],
                    fig=fig,
                    figsize=figsize,
                    title = "",
                    plane ="xy",
                    xlim=xlim,
                    ylim=ylim,
                    clim=95)

    # Plot rectangle overlay
    plot_rect(axs, width=14, height=10, overlap=[6,3], n_rect=3, lw=lw)
    
def load_csv(inpdir, fname):
    ''' Load csv as np array. Will return as strings if there are any strings '''
    data = []
    with open(os.path.join(inpdir,fname)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    data = np.array(data)
    return data

def mirror_signal(signal):
    ''' Return a signal concatenated with its mirrored version '''
    sig_flip = np.flip(signal)
    sig_cat = np.concatenate((signal, sig_flip))
    return sig_cat


def plot_flux_all(pos_all, flux_all, f_combined):
    ''' Plot flux over all frequencies and coils '''
    # Plot simulated data
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,5))
    axs = ax.flatten()

    # Loop over freqs
    data_mod = np.empty((flux_all.shape))
    for i in range(len(axs)):
        for c in range(flux_all.shape[0]):
            # Get percent mod
            plot_data = flux_all[c,i,...]
            plot_data_mod = (plot_data/np.mean(plot_data) - 1)*100
            data_mod[c,i,...] = plot_data_mod

            # Plot
            xpos = np.linspace(0,20,plot_data_mod.shape[0]*2)
            axs[i].plot(xpos, mirror_signal(plot_data_mod))

        # Reset labels
        axs[i].set_yticks(np.linspace(np.amin(data_mod[:,i,:]), np.amax(data_mod[:,i,:]), 4).astype(int))
        axs[i].set_xticks([0,10,20], labels=[0,10,0])
        axs[i].set_title(str(f_combined[i]) + " MHz")
        if i < 3:
            axs[i].set_xticks([])
            
            
def plot_flux(flux_plot, pos, xlims, f_combined, mirror=True, fig=None, ax=None, figsize=(10,10), fname="sim"):
    x_start, x_end = xlims
    # Mag and phase
    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    axs = ax.flatten()

    # Loop over freqs
    for i in range(len(axs)):
        # Define range
        n_start = np.where(np.logical_and((pos >= x_start), pos <= x_start + 1))[0][0]
        n_end = np.where(np.logical_and((pos >= x_end), pos <= x_end + 1))[0][0]  

        # Get percent mod
        plot_data = flux_plot[i,...]
        plot_data_mod = (plot_data[n_start:n_end]/np.mean(plot_data[n_start:n_end]) - 1)*100

        # Plot
        if mirror is True:
            xpos = np.linspace(0,20,plot_data_mod.shape[0]*2)
            axs[i].plot(xpos, mirror_signal(plot_data_mod), label=fname)
        else:
            axs[i].plot(pos[n_start:n_end], plot_data_mod, label=fname)
            
        # # Reset labels
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        # axs[i].set_xticks([0,10,20], labels=[0,10,0])
        axs[i].set_title(str(f_combined[i]) + " MHz")
        if i < 3:
            axs[i].set_xticks([])
            
def plot_sim(flux_mult_mag, freqs, xlims=[0,10], mirror=True, figsize=(12,5)):
    # Get combined frequency
    f_combined = proc.get_f_combined(freqs)
             
    ncoils = 3
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    for i in range(ncoils):
        fname = "log_periodic_pa_{}.csv".format(i)
        flux, freqs, pos = proc.get_phantom_flux(inpdir, fname, Npoints=81*4)
        flux_mult_mag, flux_mult_ph = proc.get_phantom_flux_mult(flux, freqs)
        # Plot only magnitude
        plot_flux(flux_mult_mag, pos, xlims, f_combined, mirror=mirror, fig=fig, ax=ax, figsize=(20,10))

    # Adjust spacing
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.2)
    

def plot_overlaid_data(data_all, f_combined, error_samp=400):
    ''' Plot data overlaid from multiple periods '''
    # Plot experimental data
    xpos = np.linspace(0, 10, data_all.shape[-2])
    mask = np.arange(len(xpos)) % error_samp == 0
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,5))
    axs = ax.flatten()
    for i in range(data_all.shape[1]):
        for c in range(data_all.shape[-1]):
            err = np.var(data_all[:,i,:,c], axis=0)
            mean = np.mean(data_all[:,i,:,c], axis=0)
            # Plot mean
            axs[i].plot(xpos,mean)
            axs[i].errorbar(x=xpos[mask], y=mean[mask], yerr=err[mask], fmt='', linestyle='', capsize=1, c='k',lw=1)
            # Plot stdev as a transparent spread
            # axs[i].fill_between(xpos, mean - err, mean + err, alpha=0.8)
            
            
        # Set formatting
        axs[i].set_title(str(f_combined[i]) + " MHz")
        axs[i].set_yticks(np.linspace(np.amin(data_all[:,i,:,:]), np.amax(data_all[:,i,:,:]), 4).astype(int))
        # Reset label
        axs[i].set_xticks([0,5,10], labels=[0,10,0])
        if i < 3:
            axs[i].set_xticks([])

    # Adjust spacing
    # plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.2)
    
def plot_vibration(pt_mag_mat, t_starts, f_combined, period=12, tr=4.3e-3,
                   shifts=[-30, -50, -500, -100, -100, -100],
                   c_inds=np.arange(16,20),
                   figsize=(10,10), lw=2,
                   cutoffs=[1,1,1,5,5,5]):
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    axs = ax.flatten()

    # Plot all signals
    for i in np.arange(pt_mag_mat.shape[0]):
        t_start = t_starts[i]
        pt_mag = pt_mag_mat[i,...]
        t = np.arange(pt_mag.shape[0])*tr - t_start

        # Loop over coils
        for idx, c in enumerate(c_inds):
            pt = proc.filter_sig(pt_mag[...,c], cutoff=cutoffs[i], # Cutoff and fs in Hz
                                fs=1/(tr), order=6, btype='low')
            shift = shifts[i]
            line = axs[i].plot(t, (pt - np.mean(pt)) + shift*idx, label = "Coil {}".format(c-16), lw=lw)

        axs[i].set_xlim([0, period])
        axs[i].set_yticks([])
        axs[i].set_title(str(f_combined[i]) + "MHz")
        if i > 2:
            axs[i].set_xlabel("Time (s)")
    plt.show()

    # Make legend
    lines = []
    labels = []
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)
    legendfig = plt.figure()
    legendfig.legend(lines[:len(c_inds)], labels[:len(c_inds)], loc='center')
    
    
def plot_accel_comparison(pt_mag, accel_d, coil_inds=np.arange(16,20), start_times=[3, 5.3], time_len=8, tr=8.7e-3):
    # Titles of each subplot
    f_combined = proc.get_f_combined()
    titles = f_combined[-2:]

    # Plot
    fig, ax = plt.subplots(figsize=(10,5), nrows=1, ncols=pt_mag.shape[0])
    axs = ax.flatten()

    for i in range(accel_d.shape[0]):
        accel_d_tmp = accel_d[i,...]
        bpt_tmp = pt_mag[i,...]
        shift = -5

        ncoils = bpt_tmp.shape[-1]
        t = np.arange(bpt_tmp.shape[0])*tr - start_times[i]
        for c in range(len(coil_inds)):
            axs[i].plot(t, proc.normalize_c(bpt_tmp, var=True)[:,coil_inds[c]] + c*shift, label="Coil {}".format(c), lw=2);
        axs[i].set_title(str(titles[i]) + " MHz")
        axs[i].set_xlim([0,time_len])
        axs[i].set_xlabel("Time (s)")


        # z accel only
        axs[i].plot(t, proc.normalize(accel_d_tmp[:,-1]) + (len(coil_inds))*shift, label= r'$\Delta$d', lw=2)
        axs[i].set_yticks([])

    plt.subplots_adjust(bottom=0.15, wspace=0.1, hspace=0.4)

    # Make legend
    lines = []
    labels = []
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        print(Line)
        lines.extend(Line)
        labels.extend(Label)
    legendfig = plt.figure()
    legendfig.legend(lines[:len(coil_inds)+1], labels[:len(coil_inds)+1], loc='center')
    plt.show()
    
def plot_pca_3d(pt_pca, ax=None, fig=None, elev=55, azim=-72, figsize=(10,5), title="", show_idx=False, label_interval=5, s=10, colorbar=True, label=True, rotations=[80,10,90], integer=True):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, subplot_kw={'projection': '3d'})
    colors = cm.coolwarm(np.linspace(0, 1, pt_pca.shape[0]))


    # Define data points
    x, y, z = pt_pca.T
    
    # Have to do this so that the xlims are correct
    ax.scatter(x, y, z, c=colors, s=s, edgecolor='black', alpha=0)

    # Add lines between points
    # Define colormap
    cmap = plt.get_cmap('coolwarm')
    # Define segments for lines
    segments = [(np.array([x[i], y[i], z[i]]), np.array([x[(i+1)%len(x)], y[(i+1)%len(y)], z[(i+1)%len(z)]])) for i in range(len(x))]
    # Create Line3DCollection
    lc = Line3DCollection(segments, cmap=cmap, alpha=0.4)
    # Set array to color lines based on colormap
    lc.set_array(np.arange(len(x)))
    # Add Line3DCollection to the plot
    ax.add_collection(lc)
    
    # Actual scatterplot
    ax.scatter(x, y, z, c=colors, s=s, alpha=1)

    

    # Angles, labels, and params
    ax.view_init(elev=elev, azim=azim)
    if label is True:
        lpad = 5 # Spacing for labels
        ax.set_xlabel("PC 1", labelpad=lpad, rotation=rotations[0])
        ax.set_ylabel("PC 2", labelpad=lpad, rotation=rotations[1])
        ax.set_zlabel("PC 3", labelpad=lpad, rotation=rotations[2])
    ax.set_title(title)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        if integer is True:
            axis.set_major_locator(plt.MaxNLocator(nbins=3, integer=True))
            axis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
        else:
            axis.set_major_locator(plt.MaxNLocator(nbins=3))
            axis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
    for axis in ['x', 'y', 'z']:
        ax.tick_params(axis=str(axis), which='both', pad=0)
        
    # Make background transparent?
    # fig.patch.set_alpha(0.5)  # Set alpha value for the background grid
    # ax.patch.set_alpha(0.5)  # Set alpha value for the background grid
    # ax.set_facecolor('lightgrey') 
    # Change the color of the axes lines
    ax.xaxis._axinfo['grid'].update(color='lightgray')  # x-axis
    ax.yaxis._axinfo['grid'].update(color='lightgray')  # y-axis
    ax.zaxis._axinfo['grid'].update(color='lightgray')  # z-axis
    
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Label all data points
    if show_idx is True:
        for label_idx in range(0, pt_pca.shape[0], label_interval):
            label_text = '{}'.format(label_idx)
            ax.text(x[label_idx], y[label_idx], z[label_idx], s=label_text, color='black')
        
    # Set colorbar
    if colorbar is True:
        cbar = fig.colorbar(cm.ScalarMappable(cmap='RdBu'), ax=ax, ticks=[], location="right", pad=0.2, shrink=0.5)
    
def plot_pca_combined(pt_pcas, tr=4.4e-3, figsize=(10,10), colorbar=False):
    ''' Plot 3D PCA traces '''
    titles = ["PT", "BPT"]
    delta_tick = [100,100]
    title_colors = ["tab:green", "tab:purple"]
    fig = plt.figure(figsize=figsize)
    axs = []
    for i in range(2):
        # Same colormap
        pt_pca = pt_pcas[i]
        colors = cm.RdBu(np.linspace(0, 1, pt_pca.shape[0]))
       
        ax = fig.add_subplot(1,2, i+1, projection='3d')
        
        # Change aspect ratio
        ax.grid(True)
        ax.set_box_aspect([3,4,2])  # [x-axis, y-axis, z-axis]
        
        scatter = ax.scatter(pt_pca[...,0], pt_pca[...,1], pt_pca[...,2], c=colors)
        ax.view_init(elev=37, azim=-27)
        
        lpad = -10 # Spacing for labels
        
        # Labels
        rot = 45
        ax.set_xlabel("PC 1", labelpad=lpad, rotation=-65)
        ax.set_ylabel("PC 2", labelpad=lpad, rotation=23)
        ax.set_zlabel("PC 3", labelpad=lpad, rotation=90)
        
        # Remove axis ticks - note that if using xticks instead of xticklabels, the grid disappears!
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        axs.append(ax)

        if i == 1 and colorbar is True:
            cbar = fig.colorbar(cm.ScalarMappable(cmap='RdBu'), ax=ax, ticks=[], location="right", pad=0.2)
            cbar.outline.set_edgecolor('None')
    return fig, axs


def plot_mod(fb_2400, cutoff, num_max, color_dict, figsize=(20,10), t_lims=[0,78], titles=[], title_colors=[], coil_mat=None, sharey=False, shift=-10, c_inds=None):
    ''' Plot filtered percent modulation given pt object '''
    # Subplot order:
        # Row 0
            # 0 - BPT mag
            # 1 - BPT phase
        # Row 1
            # 2 - PT mag
            # 3- PT phase
    npts, N, ncoils = fb_2400.pt_mag_mod.shape
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=figsize, sharey=sharey)
    axs = ax.flatten()
    tr = fb_2400.tr
    t = proc.get_t_axis(N, tr)
    t_start, t_end = t_lims
    line_list = []
    pt_ranges = []
    
    # Plot loop
    for i, subplot in enumerate(axs):
        pt = fb_2400.pt
        # if i <= 1:
        if i%2 == 0:
            # First row is BPT
            pt_plot = pt[0,...]
        else:
            # Second row is PT
            pt_plot = pt[1,...]
        
        if i == 3:
            subplot.set_xlabel("Time (s)")
        
        # Even indices are magnitude
        if i < 2:
        # if i%2 == 0:
            pt_plot = np.abs(pt_plot)
        else:
        # Phase
            pt_plot, _ = proc.get_phase_pinv(pt_plot, k=4, pca=False, c_inds=c_inds)

        # Filter
        pt_plot = proc.filter_c(pt_plot, tr=tr, cutoff=cutoff)
        
        # Set plot limits
        n_start, n_end = [int(t_start/tr), int(t_end/tr)]
        
        # Compute percent mod
        pt_mod = (pt_plot[n_start:n_end,...] / np.mean(pt_plot[n_start:n_end,...], axis=0) - 1)*100
        
        # Calculate range via percentiles
        # pt_range = np.percentile(pt_mod, 99.9, axis=0) - np.percentile(pt_mod, 0, axis=0)
        # pt_range = np.amax(pt_mod, axis=0) - np.amin(pt_mod, axis=0)
        # pt_ranges.append(pt_range)

        # Choose indices based on modulation if not specified
        if coil_mat is None:
            coil_inds = np.flip(np.argsort(np.var(np.abs(pt_mod), axis=0)))
        else:
            coil_inds = coil_mat[i]
        
        # Remove bad means
        means = np.mean(pt_plot[n_start:n_end,...], axis=0)
        thres = 0.1
        bad_means = np.where(np.abs(means) < thres)[0]

        coil_inds_filt = coil_inds.copy()
        for mean in bad_means:
            coil_inds_filt = np.delete(coil_inds_filt, np.where(coil_inds_filt == mean)[0])

        # Plot loop
        for c in range(num_max):
            coil_ind = coil_inds_filt[c]
            line, = subplot.plot(t[n_start:n_end], pt_mod[:, coil_ind] + shift*c,
                        label = "Coil " + str(coil_ind), color=color_dict[coil_ind])
            
            line_list.append(line)
            
        # Set range
        min_val = np.amin(pt_mod[:, coil_inds_filt[:num_max]])
        max_val = np.amax(pt_mod[:, coil_inds_filt[:num_max]])
        pt_range = max_val - min_val
        pt_ranges.append(pt_range)
        # subplot.set_yticks(np.linspace(min_val, max_val, 3))
        
        subplot.set_yticks(np.linspace(min_val, max_val, 3, dtype=int))

        # Set labels
        title = subplot.set_title(titles[i], c='k')
        title.set_position([0.5, 0.0])
        
        # FOR DEBUGGING - set legend
        # subplot.legend()
        
    return axs, line_list, np.array(pt_ranges)

def plot_resp(fb_2400, cutoff=2, t_lims=[0,78], figsize=(10,10), num_max=2):
    # Plot filtered modulation for breathing portion
    bad_inds = np.array([10])
    c_inds = np.delete(np.arange(16),bad_inds)
    color_dict = make_color_dict(N=16)
    
    titles = ["BPT Magnitude", "PT Magnitude", "BPT Phase", "PT Phase"]
    title_colors = ["tab:purple", "tab:purple", "tab:green", "tab:green"]
    coil_mat = None

    axs_resp, line_list, pt_range = plot_mod(fb_2400, cutoff, num_max, color_dict,
                    coil_mat=coil_mat, 
                    figsize=figsize,
                    t_lims=t_lims, titles=titles,
                    title_colors=title_colors,
                    sharey=False, shift=0,
                   c_inds=c_inds)

    # Adjust spacing
    plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3, hspace=0.6)
    
    return axs_resp, pt_range



def get_labels(axs):
    ''' Return list of labels from axis '''
    all_labels = []
    all_handles = []
    for i in range(len(axs)):
        handles, labels = axs[i].get_legend_handles_labels()
        all_labels = all_labels + labels
        all_handles = all_handles + handles
        
    return all_labels, all_handles

def sort_labels(labels):
    ''' Sort labels by number at end '''
    coil_no = np.array([labels[i].split(" ")[-1] for i in range(len(labels))]).astype(int)
    coil_no_sorted, inds = np.unique(coil_no, return_index=True)
    return coil_no_sorted, inds


def make_color_dict(N=16, color_inds=None):
    # Make a color dictionary
    colors = np.array(list(mcolors.TABLEAU_COLORS.keys()))
    css4_colors = ['hotpink', 'navy', 'lime', 'black', 'darkgoldenrod','lightgray']
    colors = list(colors) + list(css4_colors)

    # Predefine dictionary of size N
    if color_inds is None:
        color_inds = np.arange(N)
    color_dict = dict(zip(color_inds,np.array(colors)))
    return color_dict

def plot_bpt_pt_overlay(img_crop_rss, fb_2400, pad=True, use_dict=True, start=[80,40], p_size=[10,1], scales=[-0.02,0.02], shifts=[7.8,4], c=[9,3], figsize=(10,7), t_end=30, show_text=True):
    ''' Plot BPT and PT overlaid on patch of image '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Color dictionary
    if use_dict is True:
        color_dict = make_color_dict(N=16)
        colors = [color_dict[c[0]], color_dict[c[1]]]
    else:
        colors = ["tab:blue", "tab:orange"]

    # Stack parts of image
    im_seg = img_crop_rss[start[0]:start[0]+p_size[0], start[1]:start[1]+p_size[1],:]
    imstack = np.vstack(im_seg.T).T # This gives the correct shape 
    imstack_all = np.vstack((imstack, imstack)) # 2 imstacks for BPT and PT
    im = ax.imshow(imstack_all,"gray", interpolation='lanczos')

    # Plot overlay
    if pad is True:
        bpt_interp = proc.pad_bpt(fb_2400.pt_mag_filtered[0,...], npe=256, nph=100, dda=4)
        pt_interp = proc.pad_bpt(fb_2400.pt_mag_filtered[1,...], npe=256, nph=100, dda=4)
    else:
        bpt_interp = proc.normalize_c(fb_2400.pt_mag_filtered[1,...])
        pt_interp = proc.normalize_c(fb_2400.pt_mag_filtered[0,...])

    # BPT
    x_axis = np.linspace(-0.5,imstack.shape[-1]-0.5,bpt_interp.shape[0], endpoint=True)
    ax.plot(x_axis, scales[0]*bpt_interp[...,c[0]] + shifts[0], lw=4, c=colors[0], alpha=0.8)

    # PT
    ax.plot(x_axis, scales[1]*pt_interp[...,c[1]] + shifts[1], lw=4, c=colors[1], alpha=0.8)
    
    if show_text is True:
    # Text
        ax.text(0.5,0.9, 'BPT', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, c=colors[0], fontsize=24)
        ax.text(0.5,0.4, 'PT', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, c=colors[1], fontsize=24)
    ax.set_xlim([0,t_end*p_size[1]])
    ax.axis("off")

def plot_img_patch(inpdir, img_plot=None, crop_win=40, start=[80,40], p_size=[10,1], c="orange", ylim=None, cmax=1500):
    ''' Plot image with overlaid patch '''
    if img_plot is None:
        ksp = np.squeeze(cfl.readcfl(os.path.join(inpdir,"ksp")))
        # Get image and crop
        img = sp.ifft(ksp, axes=(0,1))
        img_crop = img[crop_win:-crop_win,...]
        img_crop_rss = np.rot90(sp.rss(img_crop, axes=(-1,)),3) # Rotate so chest is facing up
        img_plot = img_crop_rss[...,0]
    else:
        img_crop_rss = None
    
    # Show image with rectangle overlay
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    im = ax.imshow(img_plot,"gray")
    rect = matplotlib.patches.Rectangle([start[1], start[0]],
                             p_size[1],p_size[0],
                             linewidth=1, edgecolor=c,
                             facecolor=c, alpha=0.8)
    ax.add_patch(rect)
    if ylim is not None:
        ax.set_ylim([200,60])
    ax.axis("off")
    im.set_clim(0,cmax)
    
    return img_crop_rss

def plot_cardiac_bpt_pt(inpdir, outdir_list = np.array([127, 300, 800, 1200, 1800, 2400]).astype(str),
                             trs = np.array([4.312, 4.342, 4.321, 4.32, 4.326, 4.33])*1e-3,
                    titles = ["127.8MHz", "300MHz", "800MHz", "1.2GHz","1.8GHz","2.4GHz"],
                    t_start=0, t_end=2, shift=-5):
    ''' Plot cardiac BPT and PT  '''
    freqs = outdir_list
    # Actual plot
    fig, ax = plt.subplots(figsize=(10,3), nrows=1, ncols=5)

    for i in range(len(outdir_list)):
        tr = trs[i]

        pt_mag_full = np.abs(np.squeeze(cfl.readcfl(os.path.join(inpdir, outdir_list[i],"pt_ravel"))))

        # Plot BPT and PT
        pt_mag, bpt_mag = pt_mag_full

        # Sort indices by max energy in cardiac frequency band
        pt_idxs = proc.get_max_energy(pt_mag, tr, f_range=[0.9,3])
        bpt_idxs = proc.get_max_energy(bpt_mag, tr, f_range=[0.9,3])

        # Filter
        pt_filt = proc.filter_c(pt_mag, cutoff=2, tr=tr)
        bpt_filt = proc.filter_c(bpt_mag, cutoff=25, tr=tr)

        # Compare to physio data
        bpt_len = bpt_filt.shape[0]*tr
        ppg = proc.get_physio_waveforms(os.path.join(inpdir, outdir_list[i]), bpt_len,
                                        load_ppg=True, load_ecg=False,
                                        from_front=True)[0]

        t_ppg = np.arange(ppg.shape[0])*10e-3

        # Plot BPT, PT and PPG
        C = 4 # Number of bpt coils to plot
        t = np.arange(pt_mag.shape[0])*tr

        ax[i].plot(t, proc.normalize_c(bpt_filt[:,bpt_idxs[:C]]) + np.arange(C)*shift)
        ax[i].plot(t, -1*proc.normalize(pt_filt[:,21]) + shift*(C))
        ax[i].plot(t_ppg, proc.normalize(ppg) + shift*(C+1))

        # Labels
        ax[i].set_xlim([t_start,t_end])
        ax[0].set_ylabel("Amplitude (a.u.)", labelpad=20)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(4))

def plot_raw_cardiac_v2(inpdir, outdir_list = np.array([127, 300, 800, 1200, 1800, 2400]).astype(str),
                             trs = np.array([4.312, 4.342, 4.321, 4.32, 4.326, 4.33])*1e-3,
                    titles = ["127.8MHz", "300MHz", "800MHz", "1.2GHz","1.8GHz","2.4GHz"],
                    t_start=0, t_end=2, shift=-8, num_max=2):
    ''' Plot coils with most energy for each frequency '''
    freqs = outdir_list
    shifts = [-5, -5, -5, -9, -9, -9]
    ylims = [[-7,7], [-7,7], [-5,5], [-12,7], [-15,7], [-15,7]]
    fig, ax = plt.subplots(figsize=(10,3), nrows=1, ncols=6)

    for i in range(len(outdir_list)):
        tr = trs[i]
        pt_mag_full = np.abs(np.squeeze(cfl.readcfl(os.path.join(inpdir, outdir_list[i],"pt_ravel"))))
        
        # Select PT instead of BPT and filter
        if i == 0:
            pt_mag = proc.filter_c(pt_mag_full[0,...], cutoff=25, tr=tr)
        else:
            pt_mag = proc.filter_c(pt_mag_full[1,...], cutoff=25, tr=tr)
        
        # Sort indices by max energy in cardiac frequency band
        idxs = proc.get_max_energy(pt_mag, tr, f_range=[0.9,3])
        
        # Plot loop
        t = np.arange(pt_mag.shape[0])*tr
        for k in range(num_max):
            ax[i].plot(t, proc.normalize(pt_mag[:,idxs[k]]) + shifts[i]*k, lw=1)
            
        ax[i].set_xlim([t_start,t_end])
        ax[0].set_ylabel("Amplitude (a.u.)")
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        
def plot_bpt_physio(bpt, accel, ppg, ecg, tr=8.7e-3, tr_ecg=1e-3, tr_ppg=10e-3,
                    c=[16], norm_var=True,
                    shift=-5, t_end=10, title='', figsize=(10,10),
                    colors=None, labels=None,
                    xlim = np.array([0,7]),
                   v_shift=0.15, start_loc=0.96):
    ''' Compare BPT to accel and physio signals'''
    # Time axes
    t = np.arange(bpt.shape[0])*tr
    t_ecg = proc.get_t_axis(ecg.shape[0], delta_t=tr_ecg)
    t_ppg = proc.get_t_axis(ppg.shape[0], delta_t=tr_ppg)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Calculate time indices
    xlim_n = (xlim*1/tr).astype(int)
    
    # Plot BPT
    for i in range(bpt.shape[1]):
        line, = ax.plot(t[xlim_n[0]:xlim_n[1]], bpt[xlim_n[0]:xlim_n[1],i] + shift*i, color=colors[i])
    # Accelerometer
    accel_l, = ax.plot(t[xlim_n[0]:xlim_n[1]], proc.normalize(accel[xlim_n[0]:xlim_n[1]], var=norm_var) + shift*(i+1),color=colors[i+1])
    # PPG
    xlim_n = (xlim*1/tr_ppg).astype(int)
    ppg_l, = ax.plot(t_ppg[xlim_n[0]:xlim_n[1]], proc.normalize(ppg[xlim_n[0]:xlim_n[1]], var=norm_var) + shift*(i+2),color=colors[i+2])
    # ECG
    xlim_n = (xlim*1/tr_ecg).astype(int)
    ecg_l, = ax.plot(t_ecg[xlim_n[0]:xlim_n[1]], proc.normalize(ecg[xlim_n[0]:xlim_n[1]], var=norm_var) + shift*(i+3), color=colors[i+3])
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(title)
    
    # Set text
    label_locs_v = np.array([start_loc - v_shift*i for i in range(len(colors))])
    for i in range(len(colors)):        
        # Label the PT/BPT and coil
        ax.text(-0.05, label_locs_v[i], labels[i], ha='center',va='center',
                transform=ax.transAxes, c=colors[i], fontsize=20, rotation=0)
    
    return fig, ax

def plot_8c(inpdir, tr=8.7e-3, cutoff=4, c=[30,24], figsize=(10,10), shift=-6):
    # Load bpt
    bpt = np.squeeze(cfl.readcfl(os.path.join(inpdir,"pt_ravel")))

    # Get ECG and PPG
    [ecg, ppg] = proc.get_physio_waveforms(inpdir, bpt_len=bpt.shape[1]*tr,
                                           tr_ppg=10e-3, tr_ecg=1e-3,
                                           from_front=True,
                                           index=1)
    # Get accel data
    accel, t_accel = proc.get_accel_data(inpdir)
    # Integrate accel -> displacement
    accel_d = proc.get_accel_d(accel, tr=tr, cutoff=cutoff)
    
    # Filter PT
    pt_filt = proc.filter_c(np.abs(bpt[0,...]), tr=tr, cutoff=3)
    
    
    # Plot BPT
    # bpt_stack = proc.normalize(np.abs(bpt[1,:,c[1]]), var=True)
    # bpt_stack = bpt_stack[:,None]
    # Plot BPT
    bpt_stack = np.vstack((proc.normalize(pt_filt[...,c[0]], var=True),
                           proc.normalize(np.abs(bpt[1,:,c[1]]), var=True))).T

    labels = ["PT coil {}".format(c[0] - 16),
              "BPT coil {}".format(c[1] - 16),
              "dBCG-y", "PPG", "ECG"]
    # colors = ["tab:red","tab:gray", "tab:green", "tab:blue"]
    colors = ["tab:brown", "tab:red","tab:orange", "tab:green", "tab:blue"]

    fig, ax = plot_bpt_physio(bpt_stack, accel_d[:,1],
                              ppg, -1*ecg, tr=tr, c=c, norm_var=True,
                              shift=shift, t_end=7, figsize=figsize,
                              labels=labels, colors=colors, v_shift=0.2,
                              title="")
    
def plot_accel_bpt(bpt, accel_d, xlim=[0,5], tr=8.7e-3, cutoff=15, v_shift=0.15, start_loc=0.96, figsize=(10,5), label=True):
    ''' Plot displacement from accel vs BPT-dBCG '''
    bpt_inp = proc.normalize_c(np.abs(bpt[1,...]))
    
    accel_inp = proc.normalize_c(accel_d)
    bpt_d = proc.get_bpt_d(accel_inp, bpt_inp)
    bpt_filt = proc.filter_c(bpt_d, cutoff=cutoff, tr=tr)

    # Labels and colors
    labels = ["BPT-dBCG", "dBCG"]
    colors = ["purple", "tab:orange"]

    # Plot
    t = proc.get_t_axis(bpt.shape[1], delta_t=tr)
    fig, ax = plt.subplots(figsize=figsize)
    shift = -5
    for i, sig in enumerate([bpt_filt, accel_d]):
        ax.plot(t, proc.normalize(sig[:,1]) + shift*i, label=labels[i], color=colors[i])
        ax.set_xlim(xlim)

    if label is True:
        # Set text
        label_locs_v = np.array([start_loc - v_shift*i for i in range(len(colors))])
        for i in range(len(colors)):        
            # Label the PT/BPT and coil
            ax.text(-0.05, label_locs_v[i], labels[i], ha='center',va='center',
                    transform=ax.transAxes, c=colors[i], fontsize=20, rotation=0)
    
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])

def plot_head_pca_combined(pt_pcas, tr=4.4e-3, figsizes=None):
    ''' Plot 3D head motion PCA traces '''
    titles = ["PT", "BPT"]
    delta_tick = [40, 120]
    ylim = [-200, 200] # Time-domain plot y-limits

    # First plot
    fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=figsizes[0])
    for i in range(len(pt_pcas)):
        pt_pca = pt_pcas[i]
        # Set colormap
        if i == 0: # 
            colors = cm.RdBu(np.linspace(0, 1, pt_pca.shape[0]))
        else:
            colors = cm.PRGn(np.linspace(0, 1, pt_pca.shape[0]))
        t = np.arange(pt_pca.shape[0]) * tr
        for j in range(pt_pca.shape[-1]):
            ax1[i].scatter(t, pt_pca[..., j], c=colors, marker=".")
        ax1[i].set_xlabel("Time (s)")
        ax1[i].set_ylabel("Amplitude (a.u.)")
        ax1[i].set_title(titles[i], pad=10)
        ax1[i].set_ylim(ylim)
        ax1[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.subplots_adjust(bottom=0.15, wspace=0.7, hspace=0.2)

    # Second plot
    fig, ax2 = plt.subplots(nrows=1, ncols=2, figsize=figsizes[1], subplot_kw={'projection': '3d'})
    for i in range(len(pt_pcas)):
        pt_pca = pt_pcas[i]
        # Set colormap
        if i == 0: # 
            colors = cm.RdBu(np.linspace(0, 1, pt_pca.shape[0]))
        else:
            colors = cm.PRGn(np.linspace(0, 1, pt_pca.shape[0]))

        # Actual scatterplot
        ax2[i].scatter(pt_pca[..., 0], pt_pca[..., 1], pt_pca[...,2], c=colors)

        # Angles, labels, and params
        ax2[i].view_init(elev=55, azim=-72)
        lpad = 15 # Spacing for labels
        ax2[i].set_xlabel("PC 1", labelpad=lpad)
        ax2[i].set_ylabel("PC 2", labelpad=lpad)
        ax2[i].set_zlabel("PC 3", labelpad=lpad)
        ax2[i].tick_params(axis='both', which='major', pad=2) # Move tick labels away from axis
        ax2[i].xaxis.set_major_locator(plt.MultipleLocator(delta_tick[1]))
        ax2[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax2[i].zaxis.set_major_locator(plt.MaxNLocator(2))

        # Set non square aspect ratio
        ax2[i].grid(True)
        ax2[i].set_box_aspect([3,2,1])  # [x-axis, y-axis, z-axis]
        plt.subplots_adjust(bottom=0, wspace=0.4, hspace=0)

        
def plot_artifacts(inpdir, shift=200):
    ''' Plot BPT with and without artifact correction '''
    # Head motion data
    folder_list = ["volunteer_2pt4_ax",  "multi_pt_ax",
                "volunteer_2pt4_sag","multi_pt_sag"]
    tr = 4.4e-3

    # Get corrected PT
    pt_corr_all = proc.get_corr_pt(folder_list, inpdir=inpdir, filt=False, mod=False) # Zero mean, corrected for artifact
    pt_corr = pt_corr_all[...,0]

    # Uncorrected PT
    pt_uncorr = np.squeeze(cfl.readcfl(os.path.join(inpdir, folder_list[0], "pt_ravel")))

    # Filter
    c = 9
    cutoff = 5
    pt_uncorr_filt = proc.filter_sig(np.abs(pt_uncorr)[:,c,1], cutoff=cutoff, fs=1/tr, order=6, btype='low')
    pt_corr_filt = proc.filter_sig(pt_corr[:,c,1], cutoff=cutoff, fs=1/tr, order=6, btype='low')

    t = np.arange(pt_corr.shape[0])*tr
    plt.figure(figsize=(10,6))
    plt.plot(t-10, pt_uncorr_filt + shift);
    plt.plot(t-10, pt_corr_filt);
    plt.xlim([0,20])
    plt.legend(["Uncorrected", "Corrected"])
    # plt.yticks([])
    plt.xlabel("Time (s)")


def plot_snr_mat(snr_img, snr_cov, figsize=(30,10)):
    ''' Plot SNR from phantom data '''
    # Image
    snr_disp = snr_img.copy()
    cntr = snr_cov[:,0][0].shape[0]//2
    fig, ax = plt.subplots(figsize=figsize)
    img1 = ax.imshow(snr_disp, "jet")
    cb = plt.colorbar(img1, ax=ax)
    ax.set_aspect('auto')
    plt.axis("off")
    for t in cb.ax.get_yticklabels():
         t.set_fontsize(30)
    plt.axhline(cntr, c='w', ls='--', lw=3, alpha=0.7)
    
def plot_line_plot(snr_cov, figsize=(10,7)):
    ''' Plot line plot through SNR map '''
    res = 0.9 # mm
    x = np.arange(snr_cov[:,0][0].shape[1])*res
    cntr = snr_cov[:,0][0].shape[0]//2
    fig = plt.figure(figsize=figsize)
    # Colors
    colors = ['tab:orange','tab:purple','tab:green'];
    for i in range(snr_cov.shape[1]):
        snr_map = snr_cov[:,i][0]
        plt.plot(x,snr_map[cntr,:], c=colors[i], lw=5)
    plt.legend(["No PT","BPT","PT"], frameon=False, loc='best')
    plt.xlabel("Position (mm)")
    plt.ylabel("SNR")
    
def plot_supp_fig(inpdir, c=7, tr=3.1e-3, labels = ["PT", "BPT"], ylabels = ["Modulation (%)", "Amplitude (a.u.)"],
                 titles = ["BPT and PT Modulation", "BPT and PT After Band-pass Filtering"],
                 figsize=(10,10)):
    ''' Plot supplementary BPT and PT results with AIR coil '''
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    for j in range(2): # 2 subplots:
        pt_obj = run.load_bpt_mag_phase(inpdir,
                                        tr=tr, lpfilter=True,
                                        cutoff=5,
                                        threshold=0.1)
        t = np.arange(pt_obj.dims[1])*tr
        # Loop over BPT and PT
        for i in range(2):
            bpt = np.squeeze(pt_obj.pt_mag_filtered)[i,...]
            if j == 1:
                bpt_filt = proc.filter_sig(bpt[...,c], cutoff=np.array([0.5,5]), fs=1/tr, order=3, btype='bandpass')
            else:
                bpt_filt = np.squeeze(pt_obj.pt_mag_mod)[i,...,c]
                # bpt_filt = proc.normalize(bpt[...,c], var=False)
            ax[j].plot(t, bpt_filt, label=labels[i] + " Coil {}".format(c))
            ax[j].set_xlim([0,20])
            ax[j].set_ylabel(ylabels[j])
            if j == 1:
                ax[j].set_xlabel("Time (s)")
            ax[j].set_title(titles[j])
            ax[j].legend()
            
            
def plot_imd(inpdir, fname, figsize=(8,5), plot_range=[-25,20]):
    ''' Plot ADS sim results '''
    f = proc.load_csv(inpdir, fname)[1:,:].astype(np.float64)
    p_in, p_out = f.T

    # Restrict range
    start, end = plot_range
    start_ind = np.where(p_in==start)[0][0]
    end_ind = np.where(p_in==end)[0][0]
    x = p_in[start_ind:end_ind]
    y = p_out[start_ind:end_ind]

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(x, y, '-o')
    plt.xlabel("Input Power (dBm)")
    plt.ylabel("IMD Power (dBm)")
    plt.yticks(np.linspace(np.amin(y), np.amax(y),5).astype(int))
    plt.title("IMD Power vs Input Power for Passive Simulation")
    
    
def plot_bpt_accel(data_mat, tr=8.7e-3, figsize=(10,5), shifts=[-1,-2,-1], scales=[1,2,0.5]):
    ''' Plot BPT at 1.8 and 2.4GHz vs accelerometer vibration '''
    colors = ["tab:brown", "tab:pink", "tab:purple"]
    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(data_mat.shape[0])*tr
    for i in range(data_mat.shape[-1]):
        ax.plot(t, proc.normalize(data_mat[...,i])*scales[i] + i*shifts[i], color=colors[i])
    ax.legend(["BPT - 1863.8MHz", "BPT - 2463.9MHz", "z-displacement"], frameon=False)
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_xlabel("Time (s)")

    
def plot_supp_peaks_bpt(tr=4.4e-3, cutoff=5, figsize=(10,5), shift=-2, title=""):
    '''' Plot supplemental figure of respiratory signal with multiple peaks '''
    # TODO change this
    # basedir = "/mikRAID/sanand/pilot_tone/data/volunteer_sweep_042622/abdomen/"
    basedir = "./data/resp/"
    # inpdir_list = [f for f in os.listdir(basedir) if "fb" in f]
    inpdir_list = ["300_fb", "800_fb", "1200_fb", "1800_fb", "2400_fb"]

    # Construct matrix of BPT data from each frequency
    bpt_corr, bpt = proc.load_corrected_bpt(os.path.join(basedir,inpdir_list[0]), tr=tr, ref_coil=0, cutoff=cutoff)
    bpt_mat = np.empty((len(inpdir_list), bpt_corr.shape[0], bpt_corr.shape[1]))
    for i, inpdir in enumerate(inpdir_list):
        bpt_corr, bpt = proc.load_corrected_bpt(os.path.join(basedir,inpdir), tr=tr, ref_coil=0, cutoff=cutoff)
        bpt_mat[i,...] = bpt_corr

    # Manually plot one coil from each frequency
    c_inds = [14,10,2,1]
    shift = -4
    t = np.arange(bpt.shape[0])*tr
    plt.figure(figsize=figsize)
    c = 0
    for i in [0,1,3,4]: # Skip 1200MHz
        plt.plot(t, proc.normalize(bpt_mat[i,:,c_inds[c]]) + c*shift)
        plt.xlim([30,77])
        # plt.yticks([])
        c += 1
        plt.ylabel("Amplitude (a.u.)")
    plt.xlabel("Time (s)")
    plt.title("BPT-Rx respiratory signal across frequencies")
    plt.legend(["{} MHz".format(i) for i in [300,800, 1800, 2400]], frameon=False)
    
    
def plot_multicoil_vibration(f, bpt_cat, bpt_f, labels, tr=8.7e-3, shifts=[-5,-10], figsize=(10,5)):
    # Append to labels
    label_list = ["BPT-1.8-1", "BPT-2.4-1", "z-displacement"]
    colors = ["tab:green", "tab:olive",
              "tab:brown", "tab:pink", "tab:purple"]
    
    [labels.append(l) for l in label_list]
    
    plt.figure(figsize=figsize)
    # Plot time domain
    t = np.arange(bpt_cat.shape[0])*tr
    plt.subplot(211)
    for i in range(bpt_cat.shape[1]):
        plt.plot(t, proc.normalize(bpt_cat[...,i]) + i*shifts[0], color=colors[i])
    plt.legend(labels, loc='upper right')
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.title("Time domain vibration")

    # Plot frequency domain
    plt.subplot(212)
    for i in range(bpt_f.shape[1]):
        plt.plot(f, proc.normalize(np.abs(bpt_f[...,i]), var=True) + i*shifts[1], color=colors[i])
    plt.xlim([1,7])
    plt.legend(labels, loc='upper right')
    plt.yticks([])
    plt.title("Spectrum of vibration")
    plt.xlabel("Frequency (Hz)")
    
    plt.subplots_adjust(bottom=0.1, wspace=0.1, hspace=0.4)

    
def plot_colored_lines(x, y, ax=None, lw=2, alpha=0.1, smoothness=10):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    # Create a smoothly changing colormap along the line
    colors = np.linspace(np.amin(y), np.amax(y), len(y))
    # Define a colormap
    cmap = plt.cm.coolwarm

    # Normalize the color values
    norm = plt.Normalize(np.amin(y), np.amax(y))

    # Plot
    for i in range(len(y) - 1):
        ax.plot(x[i:i+2], y[i:i+2], c=cmap(norm(colors[i])), lw=lw, alpha=alpha)
        
        
def plot_pca_t(pca, figsize=(15,5), alpha=0.3):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    # x = np.arange(1, pca.shape[1] + 1)
    x = np.arange(pca.shape[1])
    titles = ["Registration Parameters", "BPT", "PT"]
    # Plot with colormap
    for j in range(pca.shape[0]):
        ax[j].set_ylabel("Amplitude (a.u.)")
        ax[j].set_title(titles[j])
        ax[j].set_xlabel("Frame index")
        # for i in range(pca.shape[-1]):
            # Plot colored lines in the background
            # plot_colored_lines(x, pca[j,...,i], ax=ax[j], lw=10, alpha=0.3, smoothness=1)

        # Plot changing background
        cmap = plt.cm.coolwarm
        color_values = np.linspace(0, 1, len(x))
             
        # Plot actual plots
        ax[j].plot(x, pca[j,...])

        for axis in ['x', 'y']:
            ax[j].locator_params(axis=axis, nbins=5)

        # Create a single-pixel height image with color mapped from the colormap
        im = ax[j].imshow([color_values], cmap=cmap, aspect='auto', extent=(x.min(), x.max(), *ax[j].get_ylim()), alpha=alpha)
        # im = ax[j].imshow([color_values], cmap=cmap, aspect='auto', extent=(x.min(), x.max(), pca[j,...].min()*0.9, pca[j,...].max()*1.1), alpha=alpha)

            
        # Legend
        ax[j].legend(["PC {}".format(c) for c in range(3)])
            

    plt.subplots_adjust(wspace=0.4, hspace=0)
    
    
def plot_transform_params(transform_parameters, figsize=(10,10), title="", ax=None, fig=None, res=np.array([1,1,1])):
    ''' Plot 3D rigid registration params '''
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    
    fig.suptitle(title)
    # Rotation
    ax[0].plot(transform_parameters[:, 0:3]/(2*np.pi)*360)
    ax[0].set_title('Rotation parameters')
    ax[0].legend(['yaw','roll','pitch'])
    ax[0].set_ylabel("Rotation (degrees)")

    # Translation
    trans_param = transform_parameters[:,3:]*res
    ax[1].plot(trans_param)
    ax[1].set_title('Translation parameters')
    ax[1].set_ylabel("Translation (mm)")
    ax[1].legend(['H->F translation','A->P translation','R->L translation'])
    ax[1].set_xlabel("Frame index")
    
def plot_cal_inference(data_dir="./data", cal_dir="calibration_small_movement", inf_dir="inference_v2", figsize=(15,10)):
    ''' Plot registration parameters for calibration and inference '''
    # Plot registration parameters for calibration
    calibration_dir = os.path.join(data_dir, "head", cal_dir)
    test_dir = os.path.join(data_dir, "head", inf_dir)
    calibration_params = np.load(os.path.join(calibration_dir, "reg", "rigid_params.npy"))
    test_params = np.load(os.path.join(test_dir, "reg", "rigid_params.npy"))
    
    # Filter
    calibration_params = signal.savgol_filter(calibration_params, window_length=11, polyorder=4, axis=0)
    test_params = signal.savgol_filter(test_params, window_length=11, polyorder=4, axis=0)

    # First column is calibration; second is inference
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    for i in range(len(ax)):
        if i == 0:
            params = calibration_params
            res = np.array([5, 7.7, 7.7])
        else:
            params = test_params
            res = np.array([5, 7, 7])
        plot_transform_params(params, title="", ax=ax[:,i], fig=fig, res=res)
        
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
    
def plot_bpt_pt_head(data_dir="./data", cal_dir="calibration_small_movement", inf_dir="inference_v2", figsize=(15,10), N=4):
    ''' Plot BPT and PT time series data '''
    calibration_dir = os.path.join(data_dir, "head", cal_dir)
    test_dir = os.path.join(data_dir, "head", inf_dir)
    train = np.real(cfl.readcfl(os.path.join(calibration_dir, "bpt"))) # Magnitude
    test = np.real(cfl.readcfl(os.path.join(test_dir, "bpt"))) # Magnitude

    # Combine data across antennas and average
    pt_avg_train, bpt_avg_train = proc.combine_avg_bpt(train, avg=True, norm=False)
    pt_avg_test, bpt_avg_test = proc.combine_avg_bpt(test, avg=True, norm=False)
    
    # Filter
    pt_avg_train = signal.savgol_filter(pt_avg_train, window_length=11, polyorder=4, axis=0)
    pt_avg_test= signal.savgol_filter(pt_avg_test, window_length=11, polyorder=4, axis=0)
    bpt_avg_train = signal.savgol_filter(bpt_avg_train, window_length=11, polyorder=4, axis=0)
    bpt_avg_test= signal.savgol_filter(bpt_avg_test, window_length=11, polyorder=4, axis=0)

    # First column is calibration; second is inference
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    titles = ["PT train", "BPT train", "PT test", "BPT test"]
    c = 0
    for i in range(len(ax)):
        if i == 0:
            pt = pt_avg_train
            bpt = bpt_avg_train
        else:
            pt = pt_avg_test
            bpt = bpt_avg_test
        for j, sig in enumerate([pt, bpt]):
            plot_mod_list(sig, N=N, title=titles[c], ax=ax[j,i], shift=0)
            c += 1

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    
def plot_posterior_coils(inpdir, tr=None, cutoff=2, t_end=60, n_plot=3, titles=["BPT", "PT"], reorder=False, mod_inds=np.array([4,24,4,18])):
    if tr is None:
        tr = 4.4e-3
    # Load PT obj mag, phase, and modulation
    pt_obj = run.load_bpt_mag_phase(inpdir, tr=tr,
                                     lpfilter=True,
                                     cutoff=cutoff)
    # Get BPT and PT
    if reorder is False:
        bpt, pt = pt_obj.pt_mag_filtered
    else:
        pt, bpt = pt_obj.pt_mag_filtered
    
    # Posterior coils
    c = np.arange(pt.shape[-1])
    N = int(t_end/tr)
    bpt_mod = (bpt[:N,c]/np.mean(bpt[:N,c], axis=0) - 1)*100
    pt_mod = (pt[:N,c]/np.mean(pt[:N,c], axis=0) - 1)*100
    
    # Color indices
    color_dict = make_color_dict(N=mod_inds.shape[0], color_inds=mod_inds)
    t = np.arange(N)*tr

    # Plot percent mod for 3 most modulated coils
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,7), sharex=True)
    ctr = 0
    for i, sig in enumerate([bpt_mod, pt_mod]):
        # # Sort by max abs modulation
        # if mod_inds is None:
        #     mod_inds = np.flip(np.argsort(np.amax(np.abs(sig[:N,c]), axis=0)))
        # else:
            
        for j in range(n_plot):
            ax[i].plot(t, sig[:N,mod_inds[ctr]], c=color_dict[mod_inds[ctr]], label="Coil {}".format(mod_inds[ctr]))
            ax[i].set_yticks(np.round(np.linspace(ax[i].get_ylim()[0], ax[i].get_ylim()[1], 5)))
            ax[i].set_ylabel("Modulation (%)")
            ax[i].set_title(titles[i])
            ax[i].legend()
            ctr += 1
        plt.subplots_adjust(wspace=0, hspace=0.3)
    ax[-1].set_xlabel("Time (s)")
    return pt[:N,c], bpt[:N,c]

def plot_var_pca(pt_pca, bpt_pca, pt_var_exp, bpt_var_exp, suptitle="", shift = -5, figsize=(10,10)):
    # Plot explained variance
    ncomps = pt_pca.shape[-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1,ncomps+1), np.cumsum(pt_var_exp)*100, '-o')
    ax.plot(np.arange(1,ncomps+1), np.cumsum(bpt_var_exp)*100, '-o')
    ax.axhline(95, ls='--', c='r')
    ax.legend(["PT", "BPT"])
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Cumulative explained variance for PT and BPT")
    fig.suptitle(suptitle)
    
    
def plot_snr(pt_full, tr=4.4e-3, figsize=(10,8), t_range=[0,78]):
    # Coil inds with max SNR
    means, mods, ranges = proc.get_means_mods_ranges(pt_full, t_start=t_range[0], t_end=t_range[1], tr=tr)

    # Sort by mean
    mean_inds = proc.sort_coils(means)
    pt_inds = mean_inds[1,:]
    
    # mod_inds = proc.sort_coils(mods)
    # pt_inds = mod_inds[0,:]

    # Plot
    titles = ["Mean (\u221D SNR)", "Percent modulation", "Signal range (\u221D CNR)"]
    labels = ["Mean", "Percent modulation","Signal range"]
    legend = ["BPT", "PT"]
    fig, ax = plt.subplots(figsize=figsize, nrows=3, ncols=1, sharex=True) 
    for i, sig in enumerate([means, mods, ranges]):
        for j in range(2):
            ax[i].plot(sig[j,pt_inds].T, '-o', label=legend[j] + ", mean = {}".format(np.round(np.mean(sig, axis=1)[j])))
        ax[i].set_xticks(np.arange(pt_inds.shape[0]),labels=pt_inds)
        ax[i].legend(frameon=False)
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(labels[i])
        ax[-1].set_xlabel("Coil index")
        
        
def plot_im2(im2_fname, title="IP2 at 2.4GHz", ax=None, figsize=(10,5)):
    ''' Plot IMD from measured data '''
    
    # Load data
    a = np.loadtxt(im2_fname, delimiter=',', skiprows=1)
    P_in = a[:,0]
    P_out = a[:,3]
    IM2 = a[:,-1]
    
    # Get linear fit
    P_in_ext, fund, IMD, IP2 = proc.load_imd(im2_fname)
    
    print("IP2 = {}".format(IP2))
    
    # Plots
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    
    # Plot data points
    ax.plot(P_in,P_out, '-o',alpha=0.5)
    ax.plot(P_in, IM2, '-o',alpha=0.5)
    
    # Linear fit
    ax.plot(P_in_ext, fund, c='tab:blue', label="Fundamental")
    ax.plot(P_in_ext, IMD, c='tab:orange',label="Second-order IMD")
    
    # Calculated IP2 point
    ax.axvline(IP2[0], ls='--', alpha=0.5)
    ax.axhline(IP2[1], ls='--', alpha=0.5)
    
    # Labels
    ax.set_xlabel("Input Power (dBm)")
    ax.set_ylabel("Output Power (dBm)")
    ax.set_title(title)
    ax.legend()
    
    
def plot_colorbar():
    # Create a dummy figure and axes
    fig, ax = plt.subplots(figsize=(1, 6))  # Adjust the size as needed

    # Create a ScalarMappable with a dummy color map (coolwarm)
    sm = plt.cm.ScalarMappable(cmap='coolwarm')
    sm.set_array([])  # Set an empty array

    # Create colorbar
    cbar = plt.colorbar(sm, ax=ax)

    # Hide the axes
    ax.axis('off')

    # Remove the outline of the colorbar
    cbar.outline.set_visible(False)

    # Remove the numbers from the colorbar
    # cbar.ax.yaxis.set_major_formatter(NullFormatter())
    cbar.ax.xaxis.set_ticks([])
    cbar.ax.yaxis.set_ticks([])

    # Set colorbar position to the center
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    # Show colorbar
    plt.show()
    
def plot_mod_list(pt_avg_train, N=5, title="", ax=None, figsize=(10,5), shift=-5, inds=None):
    ''' Plot N most modulated coils '''
    if inds is None:
        pt_mod_list = proc.get_mod_list(pt_avg_train)
    else:
        pt_mod_list = inds
        
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax.plot(proc.normalize_c(pt_avg_train[...,pt_mod_list[:N]], var=False) + np.arange(N)*shift)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(title)

def test_plot(pts, bpts, N=4, figsize=(10,5), inds=None):
    # First column is calibration; second is inference
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    titles = ["PT train", "BPT train", "PT test", "BPT test"]
    c = 0
    for i in range(len(ax)):
        pt = pts[i]
        bpt = bpts[i]
        for j, sig in enumerate([pt, bpt]):
            plot_mod_list(sig, N=N, title=titles[c], ax=ax[j,i], shift=0, inds=inds)
            c += 1

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    
def plot_all_pca_3d(pca, figsize=(15,5), elev=-102, azim=-14, titles=["Registration Parameters", "BPT", "PT"]):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, subplot_kw={'projection': '3d'})
    options = [False, True, True]

    for i in range(3):
        sig = pca[i,...]
        plot_pca_3d(sig, ax[i], fig=fig, elev=elev, azim=azim, title=titles[i], show_idx=False, s=40, colorbar=False, integer=options[i])
        plt.subplots_adjust(wspace=0, hspace=0)
        

def plot_correlations(pca, inds, vals, figsize=(13,10)):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    for i in range(3):
        # Define signals to plot
        reg_pc = pca[0,...,i]
        bpt_pc = pca[1,...,inds[i,0]]
        pt_pc = pca[2,...,inds[i,1]]

        # Get relative scales
        bpt_scale = np.amax(reg_pc)/np.amax(bpt_pc)
        pt_scale = np.amax(reg_pc)/np.amax(pt_pc)

        ax[i].plot(reg_pc, label="Registration PC {}".format(i)) # Transform params
        ax[i].plot(bpt_pc*bpt_scale, label="BPT PC {}, corr = {}".format(inds[i,0], vals[i,0])) # BPT
        ax[i].plot(pt_pc*pt_scale, label="PT PC {}, corr = {}".format(inds[i,1], vals[i,1])) # PT
        ax[i].set_title("Registration PCs vs PT and BPT")
        ax[i].legend()
        if i == 2:
            ax[i].set_xlabel("Frame index")

    plt.subplots_adjust(wspace=0, hspace=0.5)
    
    
def plot_error(bpt_fit, param_pca, mae, figsize=(13,10)):
    plt.figure(figsize=figsize)
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(bpt_fit[...,i])
        plt.plot(param_pca[...,i])
        plt.legend(["Regressed PT", "Param PC"])
        plt.title("Regressed PC {}, mean absolute error = {}".format(i, mae[i]))
        if i == 2:
            plt.xlabel("Frame index")
            
        plt.subplots_adjust(wspace=0, hspace=0.3)
        
        
def plot_reg_results(transformParameters, res=2, figsize=(12,6)):
    ''' Plot image registration results '''
    fig, ax = plt.subplots(1,2, figsize=figsize)
    fig.suptitle('Bulk motion estimates')
    ax[0].plot(transformParameters[:, 0]/(2*np.pi)*360, '-o')
    ax[0].legend(['In-plane rotation (clockwise)'])
    ax[0].set_ylabel('Rotation angle [degrees]')
    ax[0].set_xlabel('Frame index')
    
    ax[1].plot(transformParameters[:, 1:]*res, '-o')
    ax[1].legend(['Horizontal translation (<-- = +)', 'Vertical translation (V = +)'])
    ax[1].set_ylabel('Displacement [mm]')
    ax[1].set_xlabel('Frame index')
    
    
def plot_pt_ecg_ppg(tr=8.7e-3, cutoff=4, shift=-4, figsize=(10,7), data_dir="./data"):
    ''' Plot PT vs ECG and PPG '''
    inpdir = os.path.join(data_dir, 'dbcg')
    # Load data
    bpt = np.squeeze(cfl.readcfl(os.path.join(inpdir,"pt_ravel")))
    accel, _ = proc.get_accel_data(inpdir)
    accel_d = proc.get_accel_d(accel, tr=tr, cutoff=cutoff)

    # Get ECG and PPG
    [ecg, ppg] = proc.get_physio_waveforms(inpdir, bpt_len=bpt.shape[1]*tr,
                                           tr_ppg=10e-3, tr_ecg=1e-3,
                                           from_front=True,
                                           index=1)

    # Filter PT
    pt_filt = proc.filter_c(np.abs(bpt[0,...]), tr=tr, cutoff=3)

    # Plot
    t = np.arange(bpt.shape[1])*tr
    t_ecg = proc.get_t_axis(ecg.shape[0], delta_t=1e-3)
    t_ppg = proc.get_t_axis(ppg.shape[0], delta_t=10e-3)

    plt.figure(figsize=figsize)
    plt.plot(t_ppg, proc.normalize(ppg))
    plt.plot(t, proc.normalize(np.abs(pt_filt[...,30])) + 0.5*shift)
    plt.plot(t_ecg, -1*proc.normalize(ecg) + 2*shift)
    plt.xlim([0,5])
    plt.legend(["PPG", "PT", "ECG"], loc="upper right")
    plt.xlabel("Time (s)")
    plt.title("PT vs PPG and ECG")
    plt.ylabel("Amplitude (a.u.)")
    plt.xticks(np.round(np.linspace(0,5,10),1))
    
def plot_regression(params, pt_list, bpt_list, figsize=(13,10)):
    ''' Plot regression comparison '''
    fnames = ["bpt_reg.png", "pt_reg.png"]
    for j in range(2):
        # BPT / PT
        plt.figure(figsize=figsize)
        if j == 0:
            plot_list = bpt_list
            label = "BPT"

        else:
            plot_list = pt_list
            label = "PT"

        suptitles = ["Top","Side","Both"]

        # Both antennas, top antenna, side antenna
        for i, pt_sig in enumerate(plot_list):
            pt_fit, param_pca, error = proc.lin_reg(params, pt_sig, window_length=11, polyorder=4, ncomps=3)
            mae = np.round(np.mean(np.abs(error), axis=0),4)
            plt.subplot(3,1,i+1)
            plt.plot(pt_fit[...,1])
            plt.plot(param_pca[...,1])
            plt.title(suptitles[i] + ", MAE = {}".format(mae[1]))
            legend = ["Regressed {}".format(label), "PC 1"]
            plt.legend(legend)
            plt.suptitle("Regression with {}".format(label))
            plt.ylabel("Amplitude (a.u.)")
            plt.subplots_adjust(wspace=0, hspace=0.3)
            
        if i == 2:
            plt.xlabel("Frame index")

        # Save
        plt.savefig(fnames[j], dpi=300, transparent=True)


def plot_basic(inpdir, tr=3.1e-3, cutoff=15, shift=-5, xlim=[0,10], ax=None, figsize=(10,5), title="", N=5, c_inds=None):
    ''' Plot coils '''
    pt_obj = run.load_bpt_mag_phase(inpdir,
                                    tr=tr, lpfilter=True,
                                    cutoff=cutoff, threshold=0.5)
    pt = np.squeeze(pt_obj.pt_mag_filtered)
    t = np.arange(pt_obj.dims[1])*tr
    ncoils = pt.shape[-1]
    print(pt.shape)
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Percent mod
    if c_inds is None:
        # Plot the top N most modulated coils
        pt_mod = np.squeeze(pt_obj.pt_mag_mod)
        mod_sort = np.flip(np.argsort(np.mean(np.abs(pt_mod), axis=0))) # Mean abs modulation
        ax.plot(t, proc.normalize_c(pt_mod[...,mod_sort[:N]], var=False) + np.arange(N)*shift);
        ax.legend(mod_sort[:N])
    else:
        ax.plot(t, proc.normalize_c(pt[...,c_inds], var=False) + np.arange(c_inds.shape[0])*shift)
        ax.legend(c_inds)
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    
def get_combined_colors():
    ''' Get colormap with tab 20 and tab 20c colors '''
    # Get the tab20 and tab20c colormaps
    tab20_cmap = plt.cm.get_cmap('tab20')
    tab20c_cmap = plt.cm.get_cmap('tab20b')

    # Get colors from the tab20 colormap
    tab20_colors = [tab20_cmap(i) for i in range(20)]

    # Get colors from the tab20c colormap
    tab20c_colors = [tab20c_cmap(i) for i in range(10)]

    # Combine the colors from both colormaps
    # combined_colors = [color for pair in zip(tab20_colors, tab20c_colors) for color in pair]
    combined_colors = tab20_colors + tab20c_colors
    return combined_colors

    
def plot_coil(sort_indices, N_max=100, psize=5, lim=[100,160], zero_index=False, head_coil=True, title="", coords=None, figsize=(10,8)):
    ''' Make a 3d scatterplot of indices '''
    ncoils = sort_indices.shape[0]
    # Creating figure
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection = "3d")
    nx = 1
    
    # Generate indices
    if coords is None:
        x = np.arange(256)
        X,Y,Z = np.meshgrid(x,x,x)
    else:
        X,Y,Z = coords
        
    colors = get_combined_colors()
    
    if zero_index:
        annotations = np.arange(ncoils)
    else:
        annotations = np.arange(ncoils) + 16
    
    center_locs = np.empty((ncoils, 3))
    for coil_idx in range(ncoils):
        # Creating plot
        scatter = ax.scatter(X.flatten()[sort_indices[coil_idx, :N_max]],
                     Y.flatten()[sort_indices[coil_idx, :N_max]],
                     Z.flatten()[sort_indices[coil_idx, :N_max]],
                     s=psize, label='Coil {}'.format(annotations[coil_idx]),
                     color=colors[coil_idx*nx])
        # Text label
        x_center = np.sum(X.flatten()[sort_indices[ coil_idx, :N_max]])/N_max
        y_center = np.sum(Y.flatten()[sort_indices[ coil_idx, :N_max]])/N_max
        z_center = np.sum(Z.flatten()[sort_indices[ coil_idx, :N_max]])/N_max
        center_locs[coil_idx,:] = np.array([x_center, y_center, z_center])

        ax.text(x_center, y_center, z_center,
                 annotations[coil_idx], 
                 color='k',
                zorder=scatter.get_zorder()+30,
                 size=12)
        
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        if head_coil:
            ax.set_zlim(lim[::-1]) # So that coils 8/9 are the most anterior
        plt.title(title)
        
        ax.set_xlabel('X (L-R)')
        ax.set_ylabel('Y (A-P)')
        ax.set_zlabel('Z (H-F)')

    ax.view_init(elev=0, azim=0)
    plt.show()
    
    return X,Y,Z,center_locs, ax
    