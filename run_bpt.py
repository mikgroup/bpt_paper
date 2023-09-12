import numpy as np
import os
from get_bpt import PT

def get_pfile_list(inpdir, sort=False):
    file_list = np.array([f for f in os.listdir(inpdir) if f.endswith(".7")])
    if sort:
        num_list = np.array([int(f[1:-2]) for f in os.listdir(inpdir) if f.endswith(".7")])
        args = np.argsort(num_list)
        file_list = file_list[args] # Sort by pfile number
    return file_list

def load_bpt_mag_phase(inpdir, tr=4.3e-3, ref_coil=0, threshold=0.5, lpfilter=True, cutoff=15):
    ''' Load BPT object, get mag and phase, and filter '''
    pfile_fname = [f for f in os.listdir(inpdir) if f.startswith('P')][0]
    # Create object
    pt_obj = PT(pfile_id = os.path.join(inpdir, pfile_fname),
            inpdir=inpdir,
            outdir=inpdir,
            tr=tr, threshold=threshold,
            ref_coil=ref_coil)
    pt = pt_obj.get_pt()

    # Get filtered magnitude
    pt_mag = pt_obj.filter_pt(cutoff, med_filt=False,
                      norm=False, var=False, correct_eddy=False, corr_drift=False,
                      lpfilter=lpfilter, mag=True)
    
    # Get filtered phase
    pt_phase = pt_obj.filter_pt(cutoff, med_filt=False,
                      norm=False, var=False, correct_eddy=False, corr_drift=False,
                      lpfilter=lpfilter, mag=False)
    
    # Get percent mod in mag and phase
    pt_obj.get_pt_mod(mag=True)
    pt_obj.get_pt_mod(mag=False)
    return pt_obj

def load_bpt(inpdir, outdir, pfile_name=None, get_ksp=True):
    ''' Extract BPT based on pfile '''
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    tr = 8.7e-3 # seconds
    # Get pfile name if not specified
    if pfile_name is None:
        pfile_name = [f for f in os.listdir(inpdir) if f.startswith('P')][0]
    pt_obj = PT(pfile_id=os.path.join(inpdir, pfile_name),
                inpdir=inpdir,
                outdir=outdir,
                tr=tr)
    pt_obj.get_pt()
    bpt = np.squeeze(pt_obj.pt)
    if get_ksp is True:
        return bpt, pt_obj.ksp
    else:
        return bpt

if __name__ == '__main__':
    # Volunteer 1/26/22
    inpdir = '/mikQNAP/sanand/pilot_tone/data/volunteer_012522'
    outdir= '/mikQNAP/sanand/pilot_tone/results/volunteer_012522/'
    pfile_list = get_pfile_list(inpdir, sort=True)
    tr = 4.4 # ms
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outdir_list = ['side_2p4_nomotion','side_2p4_ax_shake','side_2p4_sag_nod',
                    'side_5p8_nomotion','side_5p8_ax_shake','side_5p8_sag_nod',
                    'top_5p8_nomotion','top_5p8_ax_shake','top_5p8_sag_nod',
                    'top_2p4_nomotion','top_2p4_ax_shake','top_2p4_sag_nod']
    print(pfile_list)
    print(outdir_list)
    for i in range(len(outdir_list)):
        pt_obj = PT(pfile_id=os.path.join(inpdir,pfile_list[i]),
                   inpdir=inpdir,
                   outdir=os.path.join(outdir,
                   outdir_list[i]), tr=tr)
        pt_obj.get_pt()
