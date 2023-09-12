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
After you've opened the notebook, run all the cells.

### Method 2: Clone and install conda environment
This method requires having conda installed. 
To run the notebook locally, clone the repo, install the conda environment and launch Jupyter Lab:

```
git clone https://github.com/mikgroup/bpt_paper.git
cd bpt_paper
conda env create -f bpt_paper_env.yml
conda activate bpt_paper_env
jupyter lab

```
Then, you can run through all the cells EXCEPT for the first one, which is required only for Google Colab.
