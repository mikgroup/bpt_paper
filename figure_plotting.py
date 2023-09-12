import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sigpy as sp
import csv
import sys
import cfl # For data i/o
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

def plot_rect(axs, w=10, h=19, overlap=0.5, n_rect=3):
    ''' Plot rectangular coils on axes '''
    # Rectangular coil overlay
    x_start = np.array([-1.5*w + overlap,0.5*w - overlap,-0.5*w])
    y_start = np.array([0,0,h/2]) - h/2
    z_start = np.array([0,0,0])

    colors = ["tab:blue","tab:green","tab:orange"]

    # Plot rectangles
    for j in range(2):
        # Vertical lines 
        axs[j].axvline(0, c='w', ls='--')
        axs[j].axvline(10+h, c='w', ls='--')
        for i in range(n_rect):
            rgb_color = mcolors.to_rgba(colors[i])
            rect = matplotlib.patches.Rectangle((y_start[i], x_start[i]), h, w, color=colors[i])
            # Set transparent face color with colored edges and linewidth 
            rect.set_facecolor(list(rgb_color[:-1]) + [0]) # Make transparent
            rect.set_edgecolor(list(rgb_color[:-1]) + [1]) # Make opaque
            rect.set_linewidth(7)
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

def plot_field_w_coils(inpdir, figsize=(16.7,5), xlim=[-45,35], ylim=[-35,35]):
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
    plot_rect(axs, w=10, h=19, overlap=0.5, n_rect=3)
    
    
def load_rocker_data(rocker_dir, experiment_list, cutoff=5, N=2, tr=4.3e-3):
    ''' Load full matrix of rocker data '''
    inpdir = os.path.join(rocker_dir, experiment_list[0])
    pt_obj = run.load_bpt_mag_phase(inpdir=inpdir,
                                   tr=tr,
                                    ref_coil=0,
                                    lpfilter=True,
                                    cutoff=cutoff)
    # Load data
    Npts, nro, ncoils = pt_obj.dims
    pt_mag = np.empty((len(experiment_list), nro, ncoils))
    c_mat = np.empty((len(experiment_list), ncoils-N))
    
    for i in range(len(experiment_list)):
        inpdir = os.path.join(rocker_dir, experiment_list[i])
        pt_obj = run.load_bpt_mag_phase(inpdir=inpdir,
                                        tr=tr,
                                        ref_coil=0,
                                        lpfilter=True,
                                        cutoff=cutoff)
        
        # Get mag and phase in percent mod units
        pt_mag[i,...] = np.squeeze(pt_obj.pt_mag_filtered)
        
        # Record which coils are saved
        c_mat[i,...] = c_inds
    
    return pt_mag, c_mat.astype(int)

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

def get_phantom_flux(inpdir, fname, Npoints=16):
    ''' Load flux from array of coils '''
    data = load_csv(inpdir, fname)
    data_vals = data[:,[0,2,3]].astype(float)
    data_re = data_vals[::2]
    data_im = data_vals[1::2]

    # Separate by freqs
    freqs = data_re[:,0][::Npoints] # MHz
    pos = data_vals[:,1][::2][:Npoints]*0.1 # cm
    Nfreqs = len(freqs)

    flux_re = np.array([data_re[i*Npoints:(i+1)*Npoints,...] for i in range(Nfreqs)])
    flux_im = np.array([data_im[i*Npoints:(i+1)*Npoints,...] for i in range(Nfreqs)])
    flux_all = flux_re + 1j*flux_im # Nfreqs, Npoints, 3
    
    flux = flux_all[:,:,-1] # Nfreqs, Npoints
    return flux, freqs, pos

def get_phantom_flux_mult(flux, freqs):
    ''' Multiply flux '''
    Nfreqs = flux.shape[0]
    flux_mult_mag = np.array([np.abs(flux[i,:]) * np.abs(flux[i+1,:]) for i in range(1,Nfreqs,2)])
    flux_mult_mag = np.insert(flux_mult_mag, 0, np.abs(flux[0,:]), axis=0)
    
    # Get combined freqs
    return flux_mult_mag

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
            
        # Reset labels
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        axs[i].set_xticks([0,10,20], labels=[0,10,0])
        axs[i].set_title(str(f_combined[i]) + " MHz")
        if i < 3:
            axs[i].set_xticks([])
            
def plot_sim(inpdir, xlims=[0,10], mirror=True, figsize=(12,5)):
    # Load simulated data
    fname = "log_periodic_pa_{}.csv".format(0)
    flux, freqs, pos = proc.get_phantom_flux(inpdir, fname, Npoints=81*4)
    flux_mult_mag, flux_mult_ph = proc.get_phantom_flux_mult(flux, freqs)
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
    
def plot_meas_data(pt_mag, f_combined, figsize=(12,5), c=np.array([9,8,10]), tr=4.3e-3):
    # Plot experimental data
    T = 100/13 * 2 + 0.5 # period, seconds
    
    # Manually define start times
    t_starts = np.flip(np.array([3.89, 1.212,
                         5, 6.49,
                         4.2, 6.79])) # Long


    tlims = np.array([[t_starts[i], t_starts[i] + T] for i in range(len(t_starts))])
    
    # Load data
    # pt_mag = np.real(cfl.readcfl(os.path.join(inpdir, "pt_mag")))

    # Plot magnitude
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    axs = ax.flatten()
    # Loop over freqs
    for i in range(len(axs)):
        t = np.arange(pt_mag.shape[1])*tr

        # Define range
        t_start, t_end = tlims[i]
        n_start = np.where(np.logical_and((t >= t_start), t <= t_start + 2*tr))[0][0]
        n_end = np.where(np.logical_and((t >= t_end), t <= t_end + 2*tr))[0][0]  

        # Get percent mod
        plot_data = pt_mag[i,:,c].T
        plot_data_mod = (plot_data[n_start:n_end,...]/np.mean(plot_data[n_start:n_end,...], axis=0) - 1)*100
        xpos = np.linspace(0,20,plot_data_mod.shape[0])
        
        # Plot
        axs[i].plot(xpos, plot_data_mod)
        axs[i].set_title(str(f_combined[i]) + " MHz")
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))

        # Reset label
        axs[i].set_xticks([0,10,20], labels=[0,10,0])

        if i < 3:
            axs[i].set_xticks([])

    # Adjust spacing
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.2)
    
def plot_vibration(pt_mag_mat, t_starts, f_combined, period=12, tr=4.3e-3,
                   shifts=[-30, -50, -500, -100, -100, -100],
                   c_inds=np.arange(16,20),
                   figsize=(10,10), lw=2,
                   cutoffs=[1,1,1,5,5,5]):
    # Plot
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize)
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
    # print(lines)
    legendfig = plt.figure()
    legendfig.legend(lines[:len(coil_inds)+1], labels[:len(coil_inds)+1], loc='center')
    
    
    
    
def plot_pca_combined(pt_pcas, tr=4.4e-3, save=True, figsize=(10,10), colorbar=False):
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

        # Set labels
        title = subplot.set_title(titles[i], c='k')
        title.set_position([0.5, 0.0])
    return axs, line_list

def plot_resp(fb_2400, cutoff=2, t_lims=[0,78], figsize=(10,10), num_max=2):
    # Plot filtered modulation for breathing portion
    bad_inds = np.array([10])
    c_inds = np.delete(np.arange(16),bad_inds)
    color_dict = make_color_dict(N=16)
    
    titles = ["BPT Magnitude", "PT Magnitude", "BPT Phase", "PT Phase"]
    title_colors = ["tab:purple", "tab:purple", "tab:green", "tab:green"]
    coil_mat = None

    axs_resp, line_list = plot_mod(fb_2400, cutoff, num_max, color_dict,
                    coil_mat=coil_mat, 
                    figsize=figsize,
                    t_lims=t_lims, titles=titles,
                    title_colors=title_colors,
                    sharey=False, shift=0,
                   c_inds=c_inds)

    # Adjust spacing
    plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3, hspace=0.6)
    
    return axs_resp



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


def make_color_dict(N=16):
    # Make a color dictionary
    colors = np.array(list(mcolors.TABLEAU_COLORS.keys()))
    css4_colors = ['hotpink', 'navy', 'lime', 'black', 'darkgoldenrod','lightgray']
    colors = list(colors) + list(css4_colors)

    # Predefine dictionary of size N
    color_inds = np.arange(N)
    color_dict = dict(zip(np.arange(N),np.array(colors)))
    return color_dict

def plot_bpt_pt_overlay(img_crop_rss, fb_2400, start=[80,40], p_size=[10,1], scales=[-0.02,0.02], shifts=[7.8,4], c=[9,3]):
    ''' Plot BPT and PT overlaid on patch of image '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
    
    # Color dictionary
    color_dict = make_color_dict(N=16)

    # Stack parts of image
    im_seg = img_crop_rss[start[0]:start[0]+p_size[0], start[1]:start[1]+p_size[1],:]
    imstack = np.vstack(im_seg.T).T # This gives the correct shape 
    imstack_all = np.vstack((imstack, imstack)) # 2 imstacks for BPT and PT
    im = ax.imshow(imstack_all,"gray", interpolation='lanczos')

    # Plot overlay
    bpt_interp = proc.pad_bpt(fb_2400.pt_mag_filtered[0,...], npe=256, nph=100, dda=4)
    pt_interp = proc.pad_bpt(fb_2400.pt_mag_filtered[1,...], npe=256, nph=100, dda=4)

    # BPT
    x_axis = np.linspace(-0.5,imstack.shape[-1]-0.5,bpt_interp.shape[0], endpoint=True)
    ax.plot(x_axis, scales[0]*bpt_interp[...,c[0]] + shifts[0], lw=4, c=color_dict[c[0]], alpha=0.8)

    # PT
    ax.plot(x_axis, scales[1]*pt_interp[...,c[1]] + shifts[1], lw=4, c=color_dict[c[1]], alpha=0.8)
    # Text
    ax.text(0.5,0.9, 'BPT', horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes, c=color_dict[c[0]], fontsize=24)
    ax.text(0.5,0.4, 'PT', horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes, c=color_dict[c[1]], fontsize=24)
    ax.set_xlim([0,30*p_size[1]])
    ax.axis("off")

def plot_img_patch(inpdir, crop_win=40, start=[80,40], p_size=[10,1]):
    ''' Plot image with overlaid patch '''
    ksp = np.squeeze(cfl.readcfl(os.path.join(inpdir,"ksp")))
    # Get image and crop
    img = sp.ifft(ksp, axes=(0,1))
    img_crop = img[crop_win:-crop_win,...]
    img_crop_rss = np.rot90(sp.rss(img_crop, axes=(-1,)),3) # Rotate so chest is facing up
    
    # Show image with rectangle overlay
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    im = ax.imshow(img_crop_rss[...,0],"gray")
    rect = matplotlib.patches.Rectangle([start[1], start[0]],
                             p_size[1],p_size[0],
                             linewidth=1, edgecolor='orange',
                             facecolor='orange', alpha=0.8)
    ax.add_patch(rect)
    ax.set_ylim([200,60])
    ax.axis("off")
    im.set_clim(0,1500)
    return img_crop_rss


def plot_raw_cardiac(inpdir, outdir_list = np.array([127, 300, 800, 1200, 1800, 2400]).astype(str),
                             trs = np.array([4.312, 4.342, 4.321, 4.32, 4.326, 4.33])*1e-3,
                    titles = ["127.8MHz", "300MHz", "800MHz", "1.2GHz","1.8GHz","2.4GHz"],
                    t_start=0, t_end=2, shift=-8, num_max=2):
    ''' Plot coils with most energy for each frequency '''
    freqs = outdir_list
    shifts = [-5, -5, -5, -9, -9, -9]
    ylims = [[-7,7], [-7,7], [-5,5], [-12,7], [-15,7], [-15,7]]
    # Actual plot
    plt.figure(figsize=(10,3))
    colors = ["tab:red", "tab:cyan"]

    for i in range(len(outdir_list)):
        pt_mag = np.abs(np.squeeze(cfl.readcfl(os.path.join(inpdir, outdir_list[i],"pt_ravel"))))
        tr = trs[i]
        # Sort indices by max energy in cardiac frequency band
        energy, idx = proc.get_max_energy(pt_mag, tr, f_range=[1,15])
        idxs = np.flip(np.argsort(energy))

        # Plot
        plt.subplot(1,6,i+1)
        t = np.arange(pt_mag.shape[0])*tr
        for k in range(num_max):
            plt.plot(t, proc.normalize(pt_mag[:,idxs[k]]) + shifts[i]*k, lw=1, color=colors[k])
        plt.xlim([t_start,t_end])
        plt.yticks([])
        plt.ylim(ylims[i])
        if i == 5:
            plt.xlabel("Time(s)")
    
    
def plot_cardiac_ica(inpdir, outdir_list = np.array([127, 300, 800, 1200, 1800, 2400]).astype(str),
                             trs = np.array([4.312, 4.342, 4.321, 4.32, 4.326, 4.33])*1e-3,
                            titles = ["127.8MHz", "300MHz", "800MHz", "1.2GHz","1.8GHz","2.4GHz"],
                             t_start=0, t_end=2, shift=-8):
    ''' Plot cardiac signals after PCA, ICA, and filtering '''
    # Filter cutoffs
    cutoffs = np.array([3, 3, 3, 20, 20, 20]).astype(int)
    k = 3 # Number of ICA comps
    fig, axs = plt.subplots(1,6, sharex=True,figsize=(10,3))
    ax = axs.flatten()
    for i in range(len(outdir_list)):
        # Get signals
        tr = trs[i] # Slightly different TR for each scan
        bpt = np.squeeze(cfl.readcfl(os.path.join(inpdir, outdir_list[i],"pt_ravel")))
        # Filter and PCA/ICA
        bpt_med, PCA_comps, ICA_comps, bpt_filt = proc.extract_cardiac(bpt, tr, k=k,
                                                                  lpfilter=True,
                                                                  cutoff=cutoffs[i],
                                                                  whiten_med=True)

        # Account for DDA
        bpt_pad = proc.pad_bpt(bpt_filt, npe=256, nph=30, dda=4, kind="linear")
        t = np.arange(bpt_pad.shape[0])*tr

        # Compare to physio data
        bpt_len = bpt_pad.shape[0]*tr
        ecg, ppg = proc.get_physio_waveforms(os.path.join(inpdir, outdir_list[i]), bpt_len,
                                        tr_ppg=10e-3, tr_ecg=1e-3,
                                        load_ppg=True, load_ecg=True,
                                        from_front=True)

        t_ppg = np.arange(ppg.shape[0])*10e-3
        t_ecg = np.arange(ecg.shape[0])*1e-3

        # Color
        if i == 0:
            color = "tab:green"
        else:
            color = "darkmagenta"

        # ICA comp index to compare
        energy, idx = proc.get_max_energy(ICA_comps,tr,f_range=[5,15])
        ax[i].plot(t, proc.normalize(bpt_pad[:,idx[0]]), lw=1, c=color)
        ax[i].plot(t_ppg, proc.normalize(ppg) + shift*1, lw=1, color="tab:orange")
        ax[i].plot(t_ecg, proc.normalize(ecg) + shift*2, lw=1, color="tab:blue")
        ax[i].set_xlim([t_start,t_end])
        ax[i].set_yticks([])

        
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

def plot_8c(inpdir, tr=8.7e-3, cutoff=4, c=[30,24]):
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
    # Plot BPT
    bpt_stack = proc.normalize(np.abs(bpt[1,:,c[1]]), var=True)
    bpt_stack = bpt_stack[:,None]

    labels = ["BPT coil {}".format(c[1] - 16),
              "dBCG-y", "PPG", "ECG"]
    colors = ["tab:red","tab:gray", "tab:green", "tab:blue"]
    fig, ax = plot_bpt_physio(bpt_stack, accel_d[:,1],
                              ppg, -1*ecg, tr=tr, c=c, norm_var=True,
                              shift=-6, t_end=7, figsize=(10,10),
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
    colors = ["purple", "tab:gray"]

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

        
def plot_artifacts(inpdir):
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
    shift = 200
    plt.figure(figsize=(10,6))
    plt.plot(t-10, pt_uncorr_filt + shift);
    plt.plot(t-10, pt_corr_filt);
    plt.xlim([0,20])
    plt.legend(["Uncorrected", "Corrected"])
    plt.yticks([])
    plt.xlabel("Time (s)")


def plot_snr(snr_img, snr_cov, figsize=(30,10)):
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
                bpt_filt = proc.normalize(bpt[...,c])
            ax[j].plot(t, bpt_filt, label=labels[i] + " Coil {}".format(c))
            ax[j].set_xlim([0,20])
            ax[j].set_ylabel(ylabels[j])
            if j == 1:
                ax[j].set_xlabel("Time (s)")
            ax[j].set_title(titles[j])
            ax[j].legend()
