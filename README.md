PyPS (Python Persistent Scatterer) scripts for processing InSAR time series. 
Author: Kyle Murray

When using these scripts please cite: 
Murray, K. D., & Lohman, R. B. (2018). Short-lived pause in Central California subsidence after heavy winter precipitation of 2017. Science Advances, 4(8), eaar8144.

Step 1. 
- Get the data from ASF
- Process to make coregistered SLCs.. I use the stack processor from ISCE
- In working directory you need a merged/SLC and merged/geom_master directory

Making the time series
Step 3. 
- Open setup_PyPS.py and change alks and rlks to whatever you want to downlook to
- run setup_PyPS.py
This takes a look at all your data and saves a dictionary of parameters with the geometry info, dimensions, time stuff, etc.  It is saved as params.npy and used in all further scripts.

Step 4. 
- run makeGamma0.py
This outputs the Gamma0.int file which gives a measurement of the phase stability at each pixel. it is downlooked, and used by smartLook.py in the next step, and used later for masking. 

Step 5.
- run smartLook.py
Downlooks all the ifgs, gamma0, and re-downlooks the geom files 

Step 6. 
- run runSnaphu.py
Unwraps all the ifgs. You can change the number of tiles and cores to use.

Step 7. 
- run refDef.py
- Choose if you want to remove a linear plane or not. 
- Does the SBAS-like inversion, flattens (using stable non-deforming pixels) and saves a .npy file which is the resulting stack, and runs invertRates.py which saves a 'rates.npy' file which are the long-term averaged rates for each pixel.

Step 8. 
- copy script plotTS.py to your working dir.
This is a script that needs to be customized to your time series.  It will plot profiles across the rate map, and do time series at individual pixels. Also profiles across the time series stack to check for flatness/ramps etc. I probably won't keep this script updated.
