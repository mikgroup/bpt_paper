import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import subprocess
# Sigpy is used for centered FFT and plotting
import sigpy as sp
# For reading CFL files
import cfl
# Processing functions
import data_processing as proc
# from util.detect_peaks import detect_peaks 
from scipy.signal import find_peaks



class PT():
    ''' Abstraction for PT or BPT data. '''
    def __init__(self, pfile_id='', inpdir='', outdir='', ksp=None, pt=None, threshold=0.5,
                dims=[], freqs=[], tr=None, fs=250, ppg=None, resp=None, ref_coil=0,
                ecg=None):
        self.inpdir = inpdir
        # If inpdir is not empty and outdir is empty, make the outdir the same except for results
        if inpdir:
            if not outdir:
                split_list = inpdir.split("/")
                final_folder = split_list[-1]
                basedir = inpdir[:-len(split_list[-1])-len(split_list[-2])-1] # Remove the two last folders
                outdir = os.path.join(basedir,"results",final_folder)
        # Raw data related
        self.outdir = outdir
        self.pfile_id = pfile_id
        self.ksp = ksp # kspace data
        
        # PT extraction-related
        self.dims = dims # [npe*ncoils, nphases, npts]
        self.freqs = freqs # PT freqs
        self.tr = tr # in seconds
        self.fs = fs # kHz
        self.threshold = threshold
        self.ref_coil = ref_coil
        
        # PT-related
        self.pt = pt
        self.pt_mag = None
        self.pt_phase = None
        
        # Post-processing
        self.pt_mag_filtered = None
        self.pt_phase_filtered = None
        self.pt_mag_mod = None
        self.pt_phase_mod = None
        self.pca_comps = None
        self.ica_comps = None

        # Peripherals
        self.ppg = ppg
        self.resp = resp
        self.ecg = ecg

    def read_ksp(self):
        ''' Convert from Pfile to kspace data with PfileToBart '''
        # Make output directory
        if not os.path.isdir(self.outdir): 
            print("Making directory " + str(self.outdir))
            os.mkdir(self.outdir)
        
        # Check if ksp exists
        if self.ksp is None:
            scan_path = os.path.join(os.path.dirname(__file__), "bin/") # Location of PfileToBart
            cmd = os.path.join(scan_path,"PfileToBart")
            cmd += " --pfile " + self.pfile_id + " --output " + os.path.join(self.outdir,"ksp")
            subprocess.run(cmd.split())

            # Read kspace dimensions - only works for 2D data with either slices or phases
            ksp = cfl.readcfl(os.path.join(self.outdir,"ksp"))
            print("ksp shape = {}".format(ksp.shape))

            # Update attributes
            self.ksp = ksp
        else:
            ksp = self.ksp
        return ksp

    def get_pt(self):
        ''' Extract the pilot tone '''
        # First check if pt exists
        inpfile = os.path.join(self.outdir, "pt_ravel.cfl")
        if os.path.exists(inpfile):
            print("PT exists! Reading file {}".format(inpfile))
            pt = np.squeeze(cfl.readcfl(os.path.join(self.outdir,"pt_ravel")))
            
            # Dimensions
            if len(pt.shape) == 2: # npe*nphases, ncoils
                self.dims = np.array(list([1]) + list(pt.shape)) # npts, npe*nphases, ncoils
                self.pt = np.reshape(pt, self.dims)
            else:
                self.dims = np.array(pt.shape) # npts, npe*nphases, ncoils
                self.pt = pt
                
            # Return pt array
            return pt
            
        else: # if PT doesn't exist, extract it based on peak in F{ksp}
            # Extract PT from ksp
            ksp = self.read_ksp()

            # Get frequency
            ksp_f = sp.ifft(ksp, axes=(0,)) # [nread, npe, nphases/nslices, ncoils]
            N, npe, nph, ncoils = ksp_f.shape

            # Detect the peak in the rss kspace based on threshold
            ksp_f_rss = sp.rss(ksp_f, axes=(1,2,3))
            loc, _ = find_peaks(ksp_f_rss/np.amax(ksp_f_rss), threshold=self.threshold, distance=5)
            # loc = detect_peaks(ksp_f_rss/np.amax(ksp_f_rss),
            #                             mph=self.threshold, mpd=5)
            f = np.arange(-N/2,N/2)*self.fs/N
            freqs = f[loc]
            if not self.freqs: # If list is empty
                self.freqs = freqs
            else:
                freqs = np.array(self.freqs)
            print("PT freqs = {}".format(freqs))
            
            # Write BPT
            bpt = ksp_f[loc,...]
            bpt_r = np.reshape(bpt, (loc.shape[0], npe*nph,ncoils), order="F") # shape is [npts, npe*nph, ncoils]
            self.pt = bpt_r
            cfl.writecfl(os.path.join(self.outdir, "pt_ravel"), bpt_r)
            
            # Dimensions
            self.dims = bpt_r.shape
            
            # return PT
            return bpt_r
            
    def extract_comps(self, pt_data, cutoff=None, k=4, lpfilter=True):
        '''' Perform PCA and ICA '''
        # This function has some bug in it - TODO debug this
        # PCA
        pca_comps,S = proc.np_pca(pt_data, threshold=True, k=k)
        # ICA
        ica_comps = proc.ica(pca_comps, k=k)
        # Filter
        if lpfilter:
            ica_comps = proc.filter_c(ica_comps, cutoff=cutoff, tr=self.tr)
        
#         # PCA
#         npe, ncoils, nph, npt = self.dims
#         pca_comps = np.empty((npe*nph, k, npt))
        
#         for i in range(self.dims[-1]):
# #             pca_comps[...,i], S = proc.np_pca(pt_data[...,i], threshold=True, k=k)
#             pca_comp, S = proc.np_pca(pt_data[...,i], threshold=True, k=k)
#             pca_comps[...,i] = pca_comp
        
#         # ICA
#         ica_comps = np.empty((npe*nph, k, npt))
#         for i in range(self.dims[-1]):
#             ica_comps[...,i] = proc.ica(pca_comps[...,i], k=k)
        
#         if lpfilter:
#             for i in range(self.dims[-1]):
#                 ica_comps[...,i] = proc.filter_c(ica_comps[...,i], cutoff=cutoff, tr=self.tr)
        
        self.pca_comps = pca_comps
        self.ica_comps = ica_comps
            
    def get_pt_mod(self, mag=True):
        ''' Compute percent modulation of magnitude and phase '''
        # Initialize with multiple possible PTs
        if mag is True:
            # Magnitude
            pt_mag_mod = np.empty(self.dims) # npts, npe*nph, ncoils
            for i in range(self.dims[0]):
                pt_mag_mod[i,...] = proc.get_percent_mod(self.pt_mag_filtered[i,...])
                self.pt_mag_mod = pt_mag_mod
        else:
            # Phase
            pt_phase_mod = np.empty(self.dims) # npts, npe*nph, ncoils
            for i in range(self.dims[0]):
                pt_phase_mod[i,...] = proc.get_percent_mod(self.pt_phase_filtered[i,...])
                self.pt_phase_mod = pt_phase_mod
            # Set ref coil phase to 0
            self.pt_phase_mod[...,self.ref_coil] = np.zeros(self.dims[:-1]) # npts, nph*nph
            
    def filter_pt(self, cutoff, med_filt=True, kernel_size=3,
                  norm=True, var=True, correct_eddy=True, corr_drift=False,
                  lpfilter=True, mag=True):
        ''' Subtract mean, median filter, correct for artifacts, and lp/bpfilter '''
        if self.pt is None:
            self.get_pt()
            
        # Process either mag or phase
        if mag:
            pt = np.abs(self.pt).reshape(self.dims)
            # # Append extra dimension 
            # if self.dims[0] == 1: # 1 PT
            #     pt = pt[None,...]
            self.pt_mag = pt
            
        else: # Phase
            pt = np.empty(self.dims) # npts, npe*nph, ncoils
            # Loop over number of pts
            for i in range(self.dims[0]):
                pt[i,...] = proc.get_phase(self.pt[i,...], ref_idx=self.ref_coil)
            self.pt_phase = pt
            
        # Optionally normalize (remove mean and divide by std)
        if norm:
            for i in range(self.dims[0]):
                pt[i,...] = proc.normalize_c(pt[i,...], var=var)
        
        # Additional median filter
        if med_filt:
            for i in range(self.dims[0]):
                pt[i,...] = proc.med_filt_c(pt[i,...], kernel_size=kernel_size)
                
        # Additional eddy current correction
        if correct_eddy:
            for i in range(self.dims[0]):
                # Reshape (npe*nph, ncoils) --> (npe, ncoils, nph)
                # TODO need to rewrite this
                pt_r = proc.bpt_reshape(pt[...,i], fwd=False, dims=self.dims[:-1])
                pt[i,...] = proc.lin_correct_all_phases(pt_r, corr_drift=corr_drift)
                
        # Low/bandpass filter
        pt_filt = pt.copy()
        if lpfilter:
            for i in range(self.dims[0]):
                pt_filt[i,...] = proc.filter_c(pt[i,...], cutoff=cutoff, tr=self.tr)
            
        # Create mag or phase attribute
        if mag:
            self.pt_mag_filtered = pt_filt
        else:
            self.pt_phase_filtered = pt_filt
        # Return
        return pt_filt
            
    def load_physio(self, ftype="PPG"):
        ''' Load physio waveform from text file '''
        # Check for text file in the input directory that starts with appropriate name
        physio_fnames = [f for f in os.listdir(self.inpdir) if f.startswith(ftype)]
        physio = [] # List to allow for multiple waveforms per physio type
        for i in range(len(physio_fnames)):
            physio_fname = physio_fnames[i]
            physio.append(np.loadtxt(os.path.join(self.inpdir,physio_fname),
                            comments="#", delimiter=",", unpack=False))
            if "ECG" in physio_fname:
                self.ecg = np.squeeze(np.array(physio))
            elif "PPG" in physio_fname:
                self.ppg = np.squeeze(np.array(physio))
            else:
                self.resp = np.squeeze(np.array(physio))
