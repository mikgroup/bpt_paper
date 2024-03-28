import numpy as np
import os
import sys
import csv

# Data I/O
import cfl

# BPT processing libraries
import sigpy as sp
import run_bpt as run
from scipy import signal
from scipy import stats
import scipy.io as sio
from sklearn.decomposition import FastICA, PCA
from scipy.interpolate import interp1d
import scipy.integrate as integ
from scipy.optimize import lsq_linear
from scipy import stats

# TODO CHANGE THIS
sys.path.append("/mikRAID/sanand/pilot_tone/head_motion/code/head_moco_bpt")
import cartesian as cart


def get_coeffs(cutoff, fs, order=5, btype='low'):
    ''' Generate the low pass filter coefficients '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype,
                        analog=False)
    return b, a

def normalize(sig, var=True):
    ''' Subtract mean and divide by std for 1D signal '''
    if var:
        return (sig - np.mean(sig))/np.std(sig)
    else:
        return (sig - np.mean(sig, axis=0))

def normalize_c(sig, var=True):
    ''' Whiten each coil data '''
    sig_out = sig.copy()
    for c in range(sig_out.shape[-1]):
        sig_out[...,c] = normalize(sig[...,c], var=var)
    return sig_out


def filter_sig(sig, cutoff, fs, order=6, btype='low'):
    ''' Filter the signal sig with desired cutoff in Hz and sampling freq fs in Hz '''
    # Get the filter coefficients so we can check its frequency response.
    b, a = get_coeffs(cutoff, fs, order, btype)
    # Filter
    # sig_filt = signal.filtfilt(b, a, sig, padlen=50)
    sig_filt = signal.filtfilt(b, a, sig, padlen=len(sig)-1)
    return sig_filt

def filter_c(bpt, cutoff=1, tr=4.4e-3):
    ''' Low pass or bandpass filter over all coils '''
    bpt_filt = np.empty(bpt.shape)
    # Check filter type  - NOTE: may cause bugs if this fails
    if type(cutoff) in [int, np.int64, float, np.float64]:
        btype = 'lowpass'
    else:
        btype = 'bandpass'
          
    # Filter per coil
    for c in range(bpt.shape[-1]):
        bpt_filt[...,c] = filter_sig(bpt[...,c],
                                     cutoff=cutoff, # Cutoff and fs in Hz
                                     fs=1/(tr), order=6, btype=btype)
    return bpt_filt

def med_filt_c(sig, kernel_size=11):
    ''' Median filter over all coils '''
    sig_out = sig.copy()
    for c in range(sig.shape[-1]):
        sig_out[...,c] = signal.medfilt(sig[...,c], kernel_size=kernel_size)
    return sig_out


def get_phase(pt, ref_idx=0):
    ''' Get phase for each coil relative to ref coil '''
    pt_phase = np.empty(pt.shape)
    for i in range(pt.shape[-1]):
        pt_phase[:,i] = np.unwrap(np.angle(pt[:,i]*np.conj(pt[:,ref_idx])))
        pt_phase[:,i] *= 180/np.pi # Convert to degrees
    return pt_phase

def get_percent_mod(pt):
    ''' Calculate percent modulation relative to the mean. Expects data in shape [npoints, ncoils]. '''
    pt_mod = (pt/np.mean(pt,axis=0)-1)*100
    return pt_mod

def load_multi_experiment_data(rocker_dir, experiment_list, tr=4.4e-3, cutoff=1, lpfilter=True):
    ''' Load full matrix of experiment data '''
    # Load first folder just to get dimensions
    inpdir = os.path.join(rocker_dir, experiment_list[0])
    pt_obj = run.load_bpt_mag_phase(inpdir=inpdir,
                                   tr=tr,
                                    ref_coil=0,
                                    lpfilter=lpfilter,
                                    cutoff=cutoff)
    Npts, nro, ncoils = pt_obj.dims
    pt_mag = np.empty((len(experiment_list), nro, ncoils))
    
    # Load remaining data from each folder in the list
    for i in range(len(experiment_list)):
        inpdir = os.path.join(rocker_dir, experiment_list[i])
        pt_obj = run.load_bpt_mag_phase(inpdir=inpdir,
                                        tr=tr,
                                        ref_coil=0,
                                        lpfilter=lpfilter,
                                        cutoff=cutoff)
        
        # Get mag and phase in percent mod units
        if lpfilter is True:
            pt_mag[i,...] = np.squeeze(pt_obj.pt_mag_filtered)
        else:
            pt_mag[i,...] = np.squeeze(pt_obj.pt_mag)
    
    return pt_mag

def get_overlaid_data(pt_mag, t_starts, tr=4.3e-3, nperiods=6, buffer=0.3):
    # Segment data by period
    T = 2*100 / 13
    c = np.array([12, 11, 13]) # Coil arrays
    N = int(np.ceil(T/tr))
    t = np.arange(pt_mag.shape[1])*tr
    tlims = np.array([[t_starts[i], t_starts[i] + T] for i in range(len(t_starts))])

    data_all = np.empty((nperiods, len(t_starts), N, len(c))) # [nperiods, nfreqs, npoints, ncoils]
    for i in range(len(t_starts)):
        t_start, t_end = tlims[i]

        for j in range(nperiods):
            n_start = np.where(np.logical_and((t >= t_start), t <= t_start + 2*tr))[0][0]
            n_end = n_start + N

            # Get percent mod
            plot_data = pt_mag[i, :, c].T
            plot_data_mod = (plot_data[n_start:n_end, ...] / np.mean(plot_data[n_start:n_end, ...], axis=0) - 1) * 100

            # Put in array
            data_all[j,i,...] = plot_data_mod

            # Increment start and end
            t_start += T + buffer
            t_end += T + buffer
            
    return data_all

def load_csv(inpdir, fname):
    ''' Load csv as np array. Will return as strings if there are any strings '''
    data = []
    with open(os.path.join(inpdir,fname)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    data = np.array(data)
    return data

def get_f_combined(freqs=None):
    ''' Get average of BPT frequencies '''
    if freqs is None: # Hardcode freqs
        freqs = np.array([127.8, 300, 427.8, 800, 927.8, 1200, 1327.8, 1800, 1927.8, 2400, 2527.8])*1e-3
    f_combined = np.array([(freqs[i] + freqs[i+1])/2 for i in range(1,freqs.shape[0],2)])
    f_combined = np.insert(f_combined, 0, freqs[0])
    f_combined = np.round(f_combined*1e3,2)
    return f_combined


def load_phantom_flux(inpdir, fname_prefix, Npoints=81, ncoils=3):
    ''' Load flux for all coils '''
    # Get size and number of points
    fname = fname_prefix + "_{}.csv".format(0)
    flux, freqs, pos = get_phantom_flux(inpdir, fname, Npoints=Npoints)
    f_combined = get_f_combined(freqs)
    nfreqs = len(f_combined)

    # Loop over all coils
    flux_all = np.empty((ncoils, nfreqs, Npoints))
    pos_all = np.empty((ncoils, Npoints))
    for i in range(ncoils):
        fname = fname_prefix + "_{}.csv".format(i)
        flux, freqs, pos = get_phantom_flux(inpdir, fname, Npoints=Npoints)
        flux_mult_mag, flux_mult_ph = get_phantom_flux_mult(flux, freqs)
        flux_all[i,...] = flux_mult_mag
        pos_all[i,...] = pos
        
    return flux_all, pos_all, f_combined

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
    Nfreqs = flux.shape[0]
    flux_mult_mag = np.array([np.abs(flux[i,:]) * np.abs(flux[i+1,:]) for i in range(1,Nfreqs,2)])
    flux_mult_ph = np.array([np.unwrap(np.angle(flux[i,:] * flux[i+1,:])) for i in range(1,Nfreqs,2)])
    
    flux_mult_mag = np.insert(flux_mult_mag, 0, np.abs(flux[0,:]), axis=0)
    flux_mult_ph = np.insert(flux_mult_ph, 0, np.angle(flux[0,:]), axis=0)
    
    # Get combined freqs
    return flux_mult_mag, flux_mult_ph


def get_accel_data(inpdir, fname=None):
    ''' Load accelerometer data from file '''
    # Load fname as file that starts with 'data'
    if fname is None:
        fname = [f for f in os.listdir(inpdir) if f.startswith('data')][0]
    data = np.loadtxt(os.path.join(inpdir,fname))
    t = data[:,0]*1e-6 # seconds
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    accel = np.vstack([x,y,z]).T
    return accel[1:,:],t # Return in the same shape as BPT

def dbl_int(accel, tr=8.7e-3, cutoff=1, get_v=False):
    ''' Double integrate acceleration -> displacement '''
    # Filter out fluctuations in accelerometer signal
    accel_filt = filter_sig(accel, cutoff=cutoff, fs=1/tr, order=6, btype='high')
    accel_v = integ.cumtrapz(accel_filt, dx=tr, initial=0)
    accel_d = integ.cumtrapz(normalize(accel_v, var=False), dx=tr, initial=0)
    if get_v is True: # Get velocity
        return accel_d, accel_v
    else:
        return accel_d

def get_accel_d(accel, tr=8.7e-3, cutoff=3, get_v=False):
    ''' Get integrated acceleration -> displacement for all axes '''
    accel_d = np.empty((accel.shape[0],3))
    if get_v is True:
        # Optionally get velocity
        accel_v = np.empty((accel.shape[0],3))
        for i in range(accel_d.shape[-1]):
            d, v = dbl_int(accel[:,i], tr=tr, cutoff=cutoff, get_v=True)
            accel_d[:,i] = d
            accel_v[:,i] = v
        return accel_d, accel_v
    else:
        # Get displacement
        for i in range(accel_d.shape[-1]):
            accel_d[:,i] = dbl_int(accel[:,i], tr=tr, cutoff=cutoff, get_v=False)
        return accel_d
    
def get_bpt_d(accel_d, bpt_inp, return_vals=False, norm=True):
    ''' Find coefficients to linearly combine BPT to match displacement'''
    bpt_d = np.empty(accel_d.shape)
    # [ncoils, naxes]
    coeffs = np.empty((bpt_inp.shape[-1], accel_d.shape[-1]))
    
    for i in range(accel_d.shape[1]):
        # Optionally normalize
        if norm is True:
            accel_inp = normalize(accel_d[:,i])
        else:
            accel_inp = accel_d[:,i]
        opt_vals = lsq_linear(bpt_inp, accel_inp)
        # Compute linear combination across BPT coils
        bpt_d[:,i] = lin_comb(opt_vals.x, bpt_inp)
        coeffs[:,i] = opt_vals.x
        
    if return_vals is False:
        return bpt_d
    else:
        return bpt_d, coeffs

# Try least squares fit to calculate coeffs of x, y and z
def lin_comb(x, accel_d):
    return np.sum(x[i] * accel_d[:,i] for i in range(accel_d.shape[1]))
    
def load_and_integrate_accel(inpdir, folder_list, tr=8.7e-3, cutoff=3):
    ''' Load and integrate acceleration to displacement '''
    
    # Get shape
    accel,t_ac = get_accel_data(os.path.join(inpdir,folder_list[0]))
    accel_mat = np.empty(list([len(folder_list)]) + list(accel.shape))
    
    for i, folder in enumerate(folder_list):
        # Load accel data and integrate
        accel,t_ac = get_accel_data(os.path.join(inpdir,folder))
        accel_d = get_accel_d(accel, tr=tr, cutoff=cutoff, get_v=False)
        accel_mat[i,...] = accel_d
    
    return accel_mat

def np_pca(data, threshold=True, k=10):
    ''' PCA along coil dim'''
    U,S,VH = np.linalg.svd(data, full_matrices=False)
    # Back project
    if threshold:
        X = np.dot(U[...,:k],np.diag(S[:k]))
    else:
        X = np.dot(U,np.diag(S))
    return X,S


def get_pca(pt1, pt2=None, n_components=2):
    ''' Do PCA with sklearn '''
    # Compute PCA of experiment 1
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(pt1)

    # Transform for pt 2
    if pt2 is None:
        pt_pca = pca.transform(pt1)
    else:
        pt_pca = pca.transform(pt2)
    
    # Explained variance of components
    var_exp = pca.explained_variance_ratio_

    return pt_pca, var_exp


def get_S(ref_idx, c):
    ''' Return matrix with one row of -1 '''
    S = np.zeros((c,c))
    S[ref_idx,:] = -1
    return S

def get_filtered_phase(pt, ref_idx, k=3):
    ''' Project phase onto k principal components '''
    bpt_phase = np.unwrap(np.angle(pt.T * np.conj(pt[:,ref_idx]).T).T, axis=0)
    U,S,VH = np.linalg.svd(bpt_phase, full_matrices=False)
    data_filt = bpt_phase @ VH[:,:k] @ VH[:,:k].T
    return data_filt

def get_phase_pinv(pt, k=3, pca=True, c_inds=None):
    N,c = pt.shape
    # Specify input coils
    if c_inds is not None:
        pt = pt[...,c_inds]
        N, c = pt.shape
    
    # Get full A matrix
    A_all = np.vstack([np.eye(c) + get_S(i,c).T for i in range(c)]) # [c^2 x c]

    # Get full data matrix
    if pca is True: # Filter with PCA first
        pt_phase_all =  np.vstack([get_filtered_phase(pt, ref_idx=i, k=k).T for i in range(c)]) # [c^2 x N]
    else:
        pt_phase_all =  np.hstack([np.angle(pt.T * np.conj(pt[:,i]).T).T for i in range(c)]).T

    # Compute pseudo inverse
    A_all_pinv = np.linalg.pinv(A_all) # [c x c^2]

    # Compute phase
    pt_phase = (A_all_pinv @ np.unwrap(pt_phase_all, axis=1)).T # [N x c]
    
    return pt_phase, pt

def get_t_axis(N, delta_t):
    ''' Get time axis based on number of samples and sample spacing '''
    return np.arange(N)*delta_t
     
def pad_bpt(bpt, npe=256, nph=30, dda=4, kind="linear"):
    ''' Interpolate BPT signal to account for dda in FIESTA scans'''
    # Create array of sampled points and points to interpolate to
    npts = (npe+dda)*nph
    interp_inds = np.arange(npts)
    del_inds = np.array([np.arange(dda) + npe*i for i in range(nph)]).flatten()
    sampled_inds = np.delete(interp_inds, del_inds)
    # Interpolate with scipy
    ncoils = bpt.shape[-1]
    bpt_interp = np.empty((npts, ncoils))
    for c in range(ncoils):
        f = interp1d(sampled_inds, bpt[...,c], kind=kind, fill_value="extrapolate")
        bpt_interp[...,c] = f(interp_inds)
    return bpt_interp




def extract_cardiac(bpt, tr, k, cutoff, whiten_med=True, lpfilter=True):
    ''' Extract cardiac signal by normalizing, PCA, ICA, and filtering '''
    if whiten_med:
        # Whiten and med filter
        bpt_norm = normalize_c(np.abs(bpt), var=False)
        bpt_med = med_filt_c(bpt_norm, kernel_size=3)
    else:
        bpt_med = bpt.copy()

    # PCA
    PCA_comps,S = np_pca(bpt_med, threshold=True, k=k)

    # ICA
    ica = FastICA(n_components=k)
    ICA_comps = ica.fit_transform(PCA_comps)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    
    # Optionally bandpass or lowpass filter
    if lpfilter:
        bpt_filt = filter_c(ICA_comps, cutoff=cutoff, tr=tr)        
    else:
        bpt_filt = ICA_comps.copy()
    return bpt_med, PCA_comps, ICA_comps, bpt_filt


def pad_bpt(bpt, npe=256, nph=30, dda=4, kind="linear"):
    ''' Interpolate BPT signal to account for dda in FIESTA scans'''
    # Create array of sampled points and points to interpolate to
    npts = (npe+dda)*nph
    interp_inds = np.arange(npts)
    del_inds = np.array([np.arange(dda) + npe*i for i in range(nph)]).flatten()
    sampled_inds = np.delete(interp_inds, del_inds)
    # Interpolate with scipy
    ncoils = bpt.shape[-1]
    bpt_interp = np.empty((npts, ncoils))
    for c in range(ncoils):
        f = interp1d(sampled_inds, bpt[...,c], kind=kind, fill_value="extrapolate")
        bpt_interp[...,c] = f(interp_inds)
    return bpt_interp


def load_physio(inpdir, ftype="PPG"):
    ''' Load physio waveform from text file '''
    # Check for text file in the input directory that starts with appropriate name
    physio_fnames = [f for f in os.listdir(inpdir) if f.startswith(ftype)]
    physio = []
    for i in range(len(physio_fnames)):
        physio.append(np.loadtxt(os.path.join(inpdir,physio_fnames[i]),
                    comments="#", delimiter=",", unpack=False))
    return np.array(physio)

def crop_physio(phys, bpt_len, tr_phys=1e-3, from_front=True):
    ''' Crop first ~30s of physio waveform '''
    phys_len = phys.shape[0]*tr_phys
    phys_diff = phys_len - bpt_len # seconds
    if from_front is True: # Remove from front
        phys_crop = phys[int(phys_diff//tr_phys):]
    else:
        phys_crop = phys[-int(bpt_len//tr_phys):]
    return phys_crop

def get_physio_waveforms(inpdir, bpt_len=None,
                         tr_ppg=10e-3, tr_ecg=1e-3,
                         load_ppg=True, load_ecg=True, from_front=True, index=0):
    ''' Load ECG and PPG data based on input directory. First ECG by default '''
    phys_waveforms = [] # Order is [ecg, ppg]
    if load_ecg is True:
        ecg = load_physio(inpdir, ftype="ECG")[index,:] # First ECG
        ecg_crop = crop_physio(ecg, bpt_len, tr_phys=tr_ecg, from_front=from_front)
        phys_waveforms.append(ecg_crop)
    if load_ppg is True:
        ppg = np.squeeze(load_physio(inpdir, ftype="PPG"))
        ppg_crop = crop_physio(ppg, bpt_len, tr_phys=tr_ppg, from_front=from_front)
        phys_waveforms.append(ppg_crop)
    return phys_waveforms

def get_max_energy(pt,tr,f_range=[0.8,1]):
    ''' Get the coil with max energy in cardiac band '''
    energy = np.empty(pt.shape[-1])
    N = pt.shape[0]
    # Get PSD
    pt_f_all = np.empty((N*4, pt.shape[1]))
    
    # Normalize
    pt_norm = normalize_c(np.abs(pt), var=False)
    
    for i in range(pt.shape[-1]):
        # pt_f,f = proc.zpad_1d(pt[:,i],fs=1/tr, N=None)
        pt_f = sp.fft(pt_norm[:,i], oshape=(pt_f_all.shape[0],))
        pt_f_all[:,i] = np.abs(pt_f)**2
        
    # Normalize
    M = pt_f_all.shape[0]
    f = np.arange(-M/2, M/2)*1/(tr*M)
    # pt_f_all /= np.amax(pt_f_all)
    f_ind = np.where(np.logical_and(f >= f_range[0],f <= f_range[1]))[0]
    energy = np.sum(pt_f_all[f_ind,:], axis=0)
    max_inds = np.flip(np.argsort(energy))
    
    return max_inds

def zpad_1d(inp, fs, N=None):
    ''' Zero-padded 1D FFT '''
    if N is None:
        N = inp.shape[0]
    inp_zp = np.pad(inp,(N,0),'constant', constant_values=0)
    win_orig = np.hanning(inp.shape[-1])
    win_new = np.ones(inp_zp.shape[-1])
    win_new[:inp.shape[-1]//2] = win_orig[:inp.shape[-1]//2]
    win_new[-inp.shape[-1]//2:] = win_orig[-inp.shape[-1]//2:]
    inp_f_zp = sp.fft(inp_zp*win_new)
    N_zp = inp_zp.shape[0]
    f_zp = fs*np.arange(-N_zp/2,N_zp/2)*1/N_zp
    return inp_f_zp, f_zp

def get_corr_pt(folder_list, inpdir,
                filt=True, cutoff=0.5, mod=True, tr=4.4e-3):
    ''' Get array of corrected PT (for multi-BPT)'''
    # Get shape
    pt_shape = np.squeeze(cfl.readcfl(os.path.join(inpdir, folder_list[0], "pt_ravel"))).shape
    pt_all = np.empty((pt_shape[0], pt_shape[1], 2, len(folder_list))) # 2 antennas
    # Put all the data into an array
    # Shape is (nro, ncoils, nantennas, nfolders) = (7680, 22, 2, 4)
    for j in range(len(folder_list)):
        folder = folder_list[j]
        pt = np.squeeze(cfl.readcfl(os.path.join(inpdir, folder, "pt")))
        for i in range(2): # Antennas
            # Remove artifact
            pt_corr, _ = lin_correct_all_phases(np.abs(pt[...,i]),
                                                  corr_drift=False, demean=False)  
            # Filter
            if filt is True:
                pt_corr = filter_c(pt_corr, cutoff=cutoff, tr=tr)
            # Compute percent mod
            if mod is True:
                pt_mod = get_percent_mod(pt_corr)
                pt_all[...,i,j] = pt_mod
            else:
                pt_all[...,i,j] = pt_corr
    return pt_all

def get_norm_pt(pt_all, f_idx, t_start=10, t_end=24, tr=4.4e-3, mtype='all'):
    ''' Get normalized, antenna-stacked PT based on folder index f_idx '''
    # Stack data from antennas together
    pt_ax = np.concatenate((pt_all[...,0,f_idx[0]], pt_all[...,1,f_idx[0]]), axis=-1)
    pt_sag = np.concatenate((pt_all[...,0,f_idx[1]], pt_all[...,1,f_idx[1]]), axis=-1)
    
    # t indices
    t = np.arange(pt_ax.shape[0])*tr
    t_inds = np.where(np.logical_and(t >= t_start, t <= t_end))[0]
    
    # Concatenate
    if mtype == 'all':
        pt = np.concatenate((pt_ax[t_inds,...], pt_sag[t_inds,...]), axis=0)
    elif mtype == 'ax': # axial
        pt = pt_ax[t_inds,...]
    else:
        pt = pt_sag[t_inds,...]
        
    # Normalize by removing the mean
    pt_norm = (pt - np.mean(pt, axis=0))
    return pt_norm


def lin_correct_all_phases(bpt_inp, corr_drift=True, demean=True):
    ''' Linear correct across phases '''
    # Try linear fit across all phases for single coil
    npe, ncoils, nph = bpt_inp.shape
    bpt_corr = np.empty((npe*nph, ncoils))
    
    # Generate vectors
    y,x = np.meshgrid(np.linspace(-1,1,nph), np.linspace(-1,1,npe))
    yy = y.flatten('F') # Drift over phases
    xx = x.flatten('F') # Drift over phase encodes
    c = np.ones((npe*nph))
    mtx = np.array([yy, xx, c]).T # Flatten fortran order; [npe*nph x 3]
    
    #  Fit per-coil
    artifact_est = np.empty(bpt_corr.shape)
    for i in range(ncoils):
        data = bpt_inp[:,i,:] # size [npe x nph]
        b = data.flatten('F') # Data
        coeffs = np.linalg.inv(mtx.T @ mtx) @ mtx.T @ b
        
        # Correct for drift over phases
        if corr_drift:
            artifact_est[:,i] = yy*coeffs[0] + xx*coeffs[1] + np.ones(npe*nph)*coeffs[2]
        
        # Correct just for linear change within phases
        else:
            if demean:
                artifact_est[:,i] = xx*coeffs[1] + np.ones(npe*nph)*coeffs[2]
            else:
                artifact_est[:,i] = xx*coeffs[1]
                
        # Remove artifact
        bpt_corr[:,i] = b - artifact_est[:,i]
        
    return bpt_corr, artifact_est

def load_bpt_accel(inpdir, folder_list, tr=8.7e-3, c=17, int_cutoff=3, filter_cutoff=5, t_starts=[2.57, 2.9, 1.89], T=12):
    ''' Load BPT and accelerometer data into a matrix '''
    # Convert time to index
    n_starts = (np.array(t_starts) * 1/tr).astype(int)
    Nsamp = int(T/tr)
    n_ends = n_starts + Nsamp
    
    # Load data matrix - BPT-1.8, BPT-2.4, accel
    data_mat = np.empty((Nsamp,3))
    for i, folder in enumerate(folder_list):
        if i < 2:
            # Get filtered BPT
            pt_obj = run.load_bpt_mag_phase(os.path.join(inpdir,folder),
                                    tr=tr,
                                    lpfilter=True,
                                    cutoff=filter_cutoff,
                                   threshold=0.7)
            npts, nro, ncoils = pt_obj.dims
            pt_mag = np.squeeze(pt_obj.pt_mag_filtered)
            data = pt_mag[n_starts[i]:n_ends[i],c]
        else: # Accelerometer
            accel, t = get_accel_data(os.path.join(inpdir, folder), fname=None)
            accel_d = get_accel_d(accel, tr=tr, cutoff=int_cutoff, get_v=False)
            data = accel_d[n_starts[-1]:n_ends[-1],2]
        
        data_mat[...,i] = data
        
    return data_mat

def bpt_reshape(bpt, dims, fwd=True):
    ''' Reshape data '''
    # Reshape data to the right dimensions
    npe, ncoils, nph = dims
    if fwd: # (npe, ncoils, nph) --> (npe*nph, ncoils)
        bpt_t = np.transpose(bpt,(0,2,1)) # Put coils as the last dimension - this is so that reshaping works as intended
        bpt_r = np.reshape(bpt_t, (npe*nph,ncoils), order="F")
    else: # (npe*nph, ncoils) --> (npe, ncoils, nph)
        bpt_r = np.reshape(bpt, (npe,nph,ncoils), order="F")
        bpt_r = np.transpose(bpt_r, (0,2,1)) # (npe, ncoils, nph)
        
    return bpt_r


def load_corrected_bpt(inpdir, tr=4.4e-3, ref_coil=0, cutoff=5):
    ''' Load respiratory BPT and correct for artifact '''
    # Load PT obj mag, phase, and modulation
    fb = run.load_bpt_mag_phase(inpdir, tr=tr, ref_coil=ref_coil, threshold=0.2, lpfilter=True, cutoff=cutoff)
    bpt = fb.pt_mag[0,...]
    
    # Median, then lp filter
    bpt = med_filt_c(bpt, kernel_size=3)
    bpt = filter_c(bpt, cutoff=cutoff, tr=tr)
    
    # Correct artifact
    ksp = cfl.readcfl(os.path.join(inpdir,"ksp"))
    nro, npe, nph, ncoils = ksp.shape
    bpt_r = bpt_reshape(bpt, dims=(npe,ncoils,nph), fwd=False)
    bpt_corr, artifact = lin_correct_all_phases(bpt_r, corr_drift=False, demean=False)
    
    return bpt_corr, bpt


def load_multicoil_vibration(tr=8.7e-3, t_start=4, T=4, filter_cutoff=np.array([1,10]), c=np.array([18,19,26]), inpdir="./data/vibration/accel"):
    ''' Load vibration data into a single matrix and get spectrum '''
    labels = ["BPT-1.8-{}".format(coil-16) for coil in c]
    # Load 1800 data for all coils
    pt_data_full = load_multi_experiment_data(inpdir, experiment_list=["1800"], tr=tr, lpfilter=True, cutoff=filter_cutoff).squeeze()
    n_start = int(t_start/tr)
    Nsamp = int(T/tr)
    bpt_vibration = pt_data_full[n_start:n_start + Nsamp,c]

    # Load accelerometer + BPT data
    folder_list = ["1800", "2400", "accel"]
    data_mat = load_bpt_accel(inpdir, folder_list, tr=tr, c=17, int_cutoff=2.5, filter_cutoff=filter_cutoff, t_starts=[t_start, 3.7, 3], T=T)

    # Concatenate
    bpt_cat = np.concatenate((bpt_vibration, data_mat), axis=1)

    # Get spectrum
    N = bpt_cat.shape[0]*5
    bpt_f = np.empty((N, bpt_cat.shape[1]), dtype=np.complex64)
    for i in range(bpt_cat.shape[1]):
        bpt_f[...,i] = sp.fft(bpt_cat[...,i], oshape=(N,))
    f = np.arange(-N/2, N/2)*1/(tr*N)
    
    return f, bpt_cat, bpt_f, labels

def get_combined_filtered_bpt(folder_list, window_length=15, polyorder=3, data_dir="./data"):
    ''' Load BPT and PT from ScanArchive, combine, and filter with savgol filter '''
    # Extract BPT from ScanArchive
    pts = []
    bpts = []
    trs = []
    
    # Assume folder_list is [train_folder, test_folder]
    for folder in folder_list:
        inpdir = os.path.join(data_dir, "head", folder)

        # Get BPT
        ksp, ksp_zp, tr, bpt = cart.extract_ksp_bpt(inpdir, threshold=0.01, distance=1, hires=False, remove_slices=False)

        # Combine and average over frames
        pt, bpt = combine_avg_bpt(bpt, avg=True, norm=False)

        # Filter
        pt_filt = signal.savgol_filter(pt, window_length=window_length, polyorder=polyorder, axis=0)
        bpt_filt = signal.savgol_filter(bpt, window_length=window_length, polyorder=polyorder, axis=0)

        # Append
        pts.append(pt_filt)
        bpts.append(bpt_filt)

    # Load transform params for test data
    transform_params = np.load(os.path.join(data_dir, "head", folder_list[1], "reg", "rigid_params.npy"))
    transform_params_filt = signal.savgol_filter(transform_params, window_length=window_length, polyorder=polyorder, axis=0)
    
    return pts, bpts, transform_params_filt


def get_bpt_pt_pca(pts, bpts, transform_params_filt, ncomps=3):
    # PCA
    pt_pca, pt_var_exp = get_pca(normalize_c(pts[0], var=False),
                                pt2=normalize_c(pts[1], var=False), n_components=ncomps)
    bpt_pca, bpt_var_exp = get_pca(normalize_c(bpts[0], var=False),
                                    pt2=normalize_c(bpts[1], var=False), n_components=ncomps)

    # PCA of params themselves as ground truth
    param_pca, param_var_exp = get_pca(normalize_c(transform_params_filt, var=False), n_components=ncomps)

    # Return PCs and explained variance
    var_exp = np.array([param_var_exp, bpt_var_exp, pt_var_exp])
    pca = np.array([param_pca, bpt_pca, pt_pca])
    return pca, var_exp


def load_pca_data(data_dir="./data", train_folder="calibration_small_movement", test_folder="inference_v2", ncomps=3, combine=True, antenna_inds=[0,1]):
    ''' Take PCA of training data and project test data onto learned PCs '''
    # Training data
    train_inpdir = os.path.join(data_dir, "head", train_folder)
    train = np.real(cfl.readcfl(os.path.join(train_inpdir, "bpt"))) # Magnitude
    tr_train = np.load(os.path.join(train_inpdir, "tr.npy"))

    # Test data
    test_inpdir = os.path.join(data_dir, "head", test_folder)
    test = np.real(cfl.readcfl(os.path.join(test_inpdir, "bpt"))) # Magnitude
    tr_test = np.load(os.path.join(test_inpdir, "tr.npy"))
    transform_params = np.load(os.path.join(test_inpdir, "reg", "rigid_params.npy"))

    # Get PCA
    pt_avg_train, bpt_avg_train = combine_avg_bpt(train, avg=True, combine=combine, antenna_inds=antenna_inds)
    pt_avg_test, bpt_avg_test = combine_avg_bpt(test, avg=True, combine=combine, antenna_inds=antenna_inds)

    # Separate training and test data
    pt_pca, pt_var_exp = get_pca(pt_avg_train, pt2=pt_avg_test, n_components=ncomps)
    bpt_pca, bpt_var_exp = get_pca(bpt_avg_train, pt2=bpt_avg_test, n_components=ncomps)

    # PCA of params themselves as ground truth
    param_pca, param_var_exp = get_pca(transform_params, n_components=ncomps)
    
    # Return PCs and explained variance
    var_exp = np.array([param_var_exp, bpt_var_exp, pt_var_exp])
    pca = np.array([param_pca, bpt_pca, pt_pca])
    return pca, var_exp

def combine_avg_bpt(train, tr=None, avg=True, cutoff=5, norm=True, combine=True, antenna_inds=[0,1]):
    ''' Combine BPT and PT across antennas and average '''
    if combine is True:
        pt_combined = np.concatenate((train[0,...], train[-1,...]), axis=-1)
        bpt_combined = np.concatenate((train[1,...], train[2,...]), axis=-1)
    else: # Pick data from a single antenna
        pt_combined = train[antenna_inds[0],...]
        bpt_combined = train[antenna_inds[1],...]
    
    if avg is True:
        # Average
        pt_avg_train = np.mean(pt_combined, axis=0)
        bpt_avg_train = np.mean(bpt_combined, axis=0)
    else: # Concatenate
        npoints, nph, ncoils = pt_combined.shape
        pt_avg_train = filter_c(pt_combined.reshape((npoints*nph,ncoils), order="F"), cutoff=cutoff, tr=tr)
        bpt_avg_train = filter_c(bpt_combined.reshape((npoints*nph,ncoils), order="F"), cutoff=cutoff, tr=tr)
        
    # Additionally remove mean
    if norm is True:
        pt_avg_train = normalize_c(pt_avg_train, var=False)
        bpt_avg_train = normalize_c(bpt_avg_train, var=False)
        
    return pt_avg_train, bpt_avg_train


def get_mod_list(pt_avg_train):
    ''' Sort by max abs modulation '''
    pt_mod = (pt_avg_train / np.mean(pt_avg_train, axis=0) - 1)*100
    mod_range = np.amax(np.abs(pt_mod), axis=0)
    pt_mod_list = np.flip(np.argsort(mod_range))
    return pt_mod_list


def sort_coils(sorting_arr):
    ''' Return sorted array from most to least '''
    if len(sorting_arr.shape) == 2: # If 3d
        return np.fliplr(np.argsort(sorting_arr))
    else:
        return np.flip(np.argsort(sorting_arr))

def get_mod(pt_seg):
    ''' Get percent mod for BPT and PT, npts in first dim '''
    bpt_mod = get_percent_mod(pt_seg[0,...])
    pt_mod = get_percent_mod(pt_seg[1,...])
    mod = np.stack((bpt_mod, pt_mod))
    return mod

def get_norm(pt_seg):
    ''' Normalize for BPT and PT, npts in first dim '''
    bpt_norm = normalize_c(pt_seg[0,...], var=False)
    pt_norm = normalize_c(pt_seg[1,...], var=False)
    norm = np.stack((bpt_norm, pt_norm))
    return norm

def get_range(pt_seg):
    ''' Compute range for BPT and PT, npts in first dim '''
    pt_norm = get_norm(pt_seg)
    ranges = np.amax(pt_norm, axis=1) - np.amin(pt_norm, axis=1)
    return ranges

def get_means_mods_ranges(pt_full, t_start=0, t_end=78, tr=4.4e-3):
    ''' Compute means, percent modulations, and ranges for BPT and PT '''
    # Define start and end times
    t_start, t_end = 0,78
    n_start, n_end = [int(t_start/tr), int(t_end/tr)]

    pt_seg = pt_full[:,n_start:n_end,...]
    
    # Compute mean, mods, range
    means = np.mean(pt_seg, axis=1)
    mods = np.amax(np.abs(get_mod(pt_seg)),axis=1)
    ranges = get_range(pt_seg)
    return means, mods, ranges


def load_imd(im2_fname):
    ''' Load intermodulation data and get linear fit '''
    a = np.loadtxt(im2_fname, delimiter=',', skiprows=1)
    P_in = a[:,0]
    P_out = a[:,3]
    IM2 = a[:,-1]

    # Linear fit
    m_lin, b_lin, r_lin, p_lin, std_err = stats.linregress(P_in, P_out)
    m_im2, b_im2, r_im2, p_im2, std_err = stats.linregress(P_in, IM2)
    
    # IP2 - find crossing point
    IP2_in = (b_im2-b_lin)/(m_lin-m_im2)
    IP2_out = m_lin*IP2_in + b_lin
    
    # Calculate linear fit
    P_in_ext = np.arange(-15,20)
    fund = m_lin*P_in_ext + b_lin
    IMD = m_im2*P_in_ext + b_im2
    IP2 = np.array([IP2_in, IP2_out])
    
    return P_in_ext, fund, IMD, IP2

def remove_drift(pt):
    ''' Remove drift via linear fit '''
    drift_x = np.arange(pt.shape[0])
    pt_corr = np.empty(pt.shape)
    
    # Remove separately for each coil
    for i in range(pt.shape[1]): # coils + antennas
        p = np.polyfit(drift_x, pt[:,i], deg=1)
        drift_y = np.polyval(p, drift_x)
        pt_corr[:,i] = pt[:,i] - drift_y
        
    return pt_corr

def remove_drift_all(pts, bpts):
    ''' Remove drift for pt and bpt '''
    pt_corr = pts.copy()
    bpt_corr = bpts.copy()
    
    for i in range(len(pts)):
        pt_corr[i] = remove_drift(pts[i])
        bpt_corr[i] = remove_drift(bpts[i])
    return pt_corr, bpt_corr