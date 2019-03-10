PyPS
note: ASF api doesn't seem to distinguish between paths.  it will just download all paths crossing the bbox. I used asf to download all the files and then output 'files' list from get_S1 function to split the files up into their paths.  

Making the coregistered IFGS:
Step 1. 
- It is possible to use the scripts getS1.py and setupStack.py to find data, but perhaps easier to just go to ASF vertex to find the search parameters.  The main things you will need are:
    lat/lon point or rectangle
    flight direction (Ascending or Descending)
    path number 
    swath number(s)

Step 2.
- Make a new directory for everything
- Copy the script setup_params.py and put it in the new directory
- Open setup_params.py and change any/all the definitions in there. 
- Change any of the arguments at the top. Use use_asf and dl=True to download the data. Otherwise leave them as false to first make sure what is available. 
- Keep run=False to make sure the stack has enough dates that cover your area of interest.  You may need to make your bounding box smaller to get more dates. 
- You can change run=True if you think you're ready to run isce. 

Now wait a day or more (depending on how much data).  You should now have a stack of coregistered interferograms, DEM, geom_files, etc.  The important stuff is in the dir called merged.  

Making the time series
Step 3. 
Now it's time to do the time series stuff.  
- Open setup_PyPS.py and change alks and rlks to whatever you want to downlook to
- run setup_PyPS.py
This takes a look at all your data and saves a dictionary of parameters with the geometry info, dimensions, time stuff, etc.  It is saved as params.npy and used in all further scripts.

Step 4. 
- run makeGamma0.py
This outputs the Gamma0.int file which gives a measurement of the phase stability at each pixel. it is downlooked, and used by smartLook.py in the next step, and used later for masking. 

Step 5.
- run smartLook.py
Downlooks all the ifgs

Step 6. 
- run runSnaphu.py
Unwraps all the ifgs

Step 7. 
- run refDef.py
Does the SBAS inversion and saves a .npy file which is the resulting stack. 

Step 8. 
- run invertRates.py
Does a seasonal and secular inversion to the output from the previous step. Outputs a rate map in TS/rates*

Step 9. 
- copy script plotTS.py to your working dir.
This is a script that needs to be customized to your time series.  It will plot profiles across the rate map, and do time series at individual pixels. Also profiles across the time series stack to check for flatness/ramps etc. 


