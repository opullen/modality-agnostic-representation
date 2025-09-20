# modality-agnostic-representation
Code for the MICCAI 2025 ShapeMI workshop paper "A Modality-Agnostic Representation for Scoliosis Phenotyping"

A real-valued sine low-dimensional Fourier Sine Series is used to model the shape of the spine. This allows the spine to be modeled as a ten real-valued sine components, which are fed-into a simple neural network to classify the Phenotypic characteristics of Adolescent Idiopathic Scoliosis. The model is trained soley on DXA dervied representations but a single model can classify across DXA, X-Ray and MRI derived representations.


![alt text](plots/fourier_process.png "An overview of the process and classification")


We use a discrete Sine Transform to decompose the sine signal into it's component parts and the Sine componenets are summed for a full or partial reconstruction.

![alt text](plots/summed_sine_coefficients.png "Partial Reconstruction of a Curve")

The results of examples of our method are shown.  
<p float="left">
  <img src="plots/dxa_results_plot.png" alt="DXA" width="200"/>
  <img src="plots/xray_results_plot.png" alt="X-ray" width="228"/>
  <img src="plots/mri_results_plot.png" alt="MRI" width="220"/>
</p>

To install the environment and run the code run the lines below.
<pre><code>conda create -n scoliosis_fourier python=3.12
conda activate scoliosis_fourier
pip install -r requirements.txt</code></pre>
