# **Beat Pilot Tone: Simultaneous Radio-Frequency Motion Sensing and Imaging at Arbitrary Frequencies in MRI**
**Authors:** Suma Anand<sup>1</sup>, Michael Lustig<sup>1</sup>

<sup>1</sup> University of California, Berkeley

The notebook <code>figure_plotting.ipynb</code> reproduces the figures and processing from the manuscript "Beat Pilot Tone: Simultaneous Radio-Frequency Motion Sensing and Imaging at Arbitrary Frequencies in MRI" submitted to Magnetic Resonance in Medicine.

A preprint of this paper is on [arXiv](https://arxiv.org/abs/2306.10236). 

Author of this code: [Suma Anand](https://people.eecs.berkeley.edu/~sanand/), sanand@berkeley.edu

## Instructions

### Method 1: Google Colab (recommended!)
To run this notebook in Google Colab, click on [this link](https://colab.research.google.com/github/mikgroup/bpt_paper/blob/main/figure_plotting.ipynb).
Alternatively, you can double click on the notebook in your browser and click on the "Open in Colab" icon.
After you've opened the notebook, run all the cells. Note that this requires you to sign in with a Google account.

### Method 2: Clone and install conda environment
This method requires having conda and git installed. 
To install conda, follow the installation instructions at [this link](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) or for whichever operating system you have.
You may have to add the conda installation directory to $PATH to ensure commands work normally.
If you don't have git installed, install it with the below command on Linux systems:

```
sudo apt install -y git
```

To run the notebook locally, clone the repo, install the conda environment and launch Jupyter Lab:

```
git clone https://github.com/mikgroup/bpt_paper.git
cd bpt_paper
conda env create -f bpt_paper_env.yml
conda activate bpt_env
jupyter lab

```


If the activation command does not work, you can also try:

```
source activate bpt_paper_env
```

or:

```
activate bpt_paper_env
```

Then, you can run through all the cells EXCEPT for the first one, which is required only for Google Colab.
Note: Method 2 may fail on Mac or Windows; it has been tested on a Linux machine running Ubuntu 20.04.6 LTS. 
