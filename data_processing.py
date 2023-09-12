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
    sig_filt = signal.filtfilt(b, a, sig, padlen=50)
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

def load_rocker_data(rocker_dir, experiment_list, tr_dict=None, cutoff=1):
    ''' Load full matrix of rocker data '''
    # Get dims
    if tr_dict is None:
        tr = 4.3e-3
    else:
        tr = tr_dict[experiment_list[0]]*1e-6
        
    # Load first folder just to get dimensions
    inpdir = os.path.join(rocker_dir, experiment_list[0])
    pt_obj = run.load_bpt_mag_phase(inpdir=inpdir,
                                   tr=tr,
                                    ref_coil=0,
                                    lpfilter=True,
                                    cutoff=cutoff)
    Npts, nro, ncoils = pt_obj.dims
    pt_mag = np.empty((len(experiment_list), nro, ncoils))
    
    # Load remaining data from each folder in the list
    for i in range(len(experiment_list)):
        if tr_dict is None:
            tr = 4.3e-3
        else:
            tr = tr_dict[experiment_list[i]]*1e-6
        inpdir = os.path.join(rocker_dir, experiment_list[i])
        pt_obj = run.load_bpt_mag_phase(inpdir=inpdir,
                                        tr=tr,
                                        ref_coil=0,
                                        lpfilter=True,
                                        cutoff=cutoff)
        
        # Get mag and phase in percent mod units
        pt_mag[i,...] = np.squeeze(pt_obj.pt_mag_filtered)
    
    return pt_mag

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
    
def get_bpt_d(accel_d, bpt_inp):
    ''' Find coefficients to linearly combine BPT to match displacement'''
    bpt_d = np.empty(accel_d.shape)
    for i in range(accel_d.shape[1]):
        accel_inp = normalize(accel_d[:,i])
        opt_vals = lsq_linear(bpt_inp, accel_inp)
        bpt_d[:,i] = lin_comb(opt_vals.x, bpt_inp)
    return bpt_d

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
    var_exp = np.sum(pca.explained_variance_ratio_)

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
    # Get PSD
    pt_f_all = np.empty((pt.shape[0]*2, pt.shape[1]))
    for i in range(pt.shape[-1]):
        pt_f,f = zpad_1d(pt[:,i],fs=1/tr, N=None)
        pt_f_all[:,i] = np.abs(pt_f)**2
    # Normalize
    pt_f_all /= np.amax(pt_f_all)
    f_ind = np.where(np.logical_and(f >= f_range[0],f <= f_range[1]))[0]
    energy = np.sum(pt_f_all[f_ind,:], axis=0)
    max_inds = np.flip(np.argsort(energy))
    return energy, max_inds

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
            pt_corr = lin_correct_all_phases(np.abs(pt[...,i]),
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
    for i in range(ncoils):
        data = np.abs(bpt_inp[:,i,:]) # size [npe x nph]
        b = data.flatten('F') # Data
        coeffs = np.linalg.inv(mtx.T @ mtx) @ mtx.T @ b
        if corr_drift: # correct for drift over phases
            bpt_corr[:,i] = b - (yy*coeffs[0] + xx*coeffs[1] + np.ones(npe*nph)*coeffs[2])
        else:
            if demean:
                bpt_corr[:,i] = b - (xx*coeffs[1] + np.ones(npe*nph)*coeffs[2])
            else:
                bpt_corr[:,i] = b - (xx*coeffs[1])
    return bpt_corr