import numpy as np
import isceobj
from matplotlib import pyplot as plt
import os
from scipy.signal import convolve2d as conv2

# SET THESE
gamThresh = 0
corThresh = 0 

params = np.load('params.npy',allow_pickle=True).item()
geom = np.load('geom.npy',allow_pickle=True).item()
locals().update(params)
locals().update(geom)

if not os.path.isdir('snaphu'):
    os.mkdir('snaphu')
    
intfile_filt1='snaphu/smallfilt.int.npy'
intfile_filtw1='snaphu/smallfiltw.int.npy'
intfile_filt2='snaphu/bigfilt.int.npy'
mask1=np.load('gam.npy')

def myfilt(infile,maskfile,outfile,countfile,rx,ry,newnx,newny,windowtype=2,outtype=1):
    windx=np.exp(- (np.arange(np.dot(- rx,2),np.dot(rx,2))) ** 2 / 2 / (rx / 2) ** 2)
    windy=np.exp(- (np.arange(np.dot(- ry,2),np.dot(ry,2))) ** 2 / 2 / (ry / 2) ** 2)
    xsum=np.sum(windx)
    ysum=np.sum(windy)
    windx=windx / xsum
    windy=windy / ysum
    ry=np.floor(len(windy) / 2)
    fidi = isceobj.createIntImage(); fidi.load(infile + '.xml')
    #out files
    mask = np.load(maskfile)
    mask=mask == 1
    in_ = fidi.memMap()[:,:,0]
    in_=np.exp(np.multiply(1j,in_))
    in_[np.isnan(in_)]=0
    in_[~mask]=0
    good=in_ != 0
    win = windx[:, np.newaxis] * windy[np.newaxis,:]
    goodsum=conv2(good,win,'same')
    datasum=conv2(in_,win,'same')
    out=datasum / goodsum
    #write count
    if countfile:
        np.save(countfile,goodsum)
    np.save(outfile,np.angle(out))

i=0
for i,p in enumerate(pairs):
    p = pairs[i]
    intfile = intdir + '/' + p + '/' + 'fine_lk.r4'
    corfile = intdir + '/' + p + '/' + 'cor_lk.r4'
    intmask = intdir + '/' + p + '/' + 'fine_lk.msk.npy'

    intfile_psfilt = intdir + '/' + p  + '/' + 'psfilt.int'
    intfile_2pi = intdir + '/' + p  + '/' + '2pi.unw'
    intfile_unw_file = intdir + '/' + p  + '/' + 'new.unw'
    
    if not os.path.isfile(intfile_unw_file + 'i'):
        print('unwrapping ' + intfile)
#        os.remove('maskfill.int')
#        os.remove('snaphu/snaphu.out')
#        os.remove('snaphu/snaphu.in')
#        os.remove('snaphu/snaphu.msk')
       
        # Load correlation file
        im = isceobj.createImage()
        im.load(corfile + '.xml')
        mask2 = im.memMap()[:,:,0]

        # Make the mask using the gamma0 and correlation metrics
        mask = np.zeros(mask2.shape)
        mask_ids = np.where((mask1 > gamThresh) | (mask2 > corThresh))
        mask[mask_ids] = 1
        np.save(intmask,mask) # Write file out
        
        # Write the mask the the snaphu directory
        ou = im.clone()
        ou.filename = 'snaphu/snaphu.msk'
        ou.dump(str(ou.filename)  + '.xml') # Write out xml
        mask = np.asarray(mask,dtype=np.float32)
        mask.tofile(ou.filename ) # Write file out
        
        if i == len(pairs)-1:
            fig,ax = plt.subplots(1,3)
            ax[0].imshow(mask1>gamThresh)
            ax[1].imshow(mask2>corThresh)
            ax[2].imshow(mask)
        
#        plt.figure();plt.hist(mask1.flatten(),50)
        
        #filter twice and fill masked area
        myfilt(intfile,intmask,intfile_filt1,intfile_filtw1, 5,5,nxl,nyl,2,1)
        myfilt(intfile,intmask,intfile_filt2,0,40,40,nxl,nyl,2,1)
        
        im = isceobj.createImage()
        im.load(intfile + '.xml')
        a = im.memMap()[:,:,0].copy()
        b = np.load(intfile_filt1)
        c = np.load(intfile_filtw1)
        d = np.load(intfile_filt2)
        b = np.exp(np.multiply(1j,b))
        d = np.exp(np.multiply(1j,d))
        m1=mask == 1
        
        #masked, then d.
        out = np.zeros(b.shape)
        out[m1]=a[m1]
        out[~m1]=np.angle( np.multiply(b[~m1],c[~m1]) + np.multiply(d[~m1],(1 - c[~m1])) )
        
        out[np.isnan(b)]=np.angle(d[np.isnan(b)])
        out[np.isnan(a)]=np.nan

        outc = np.zeros((nyl,nxl*2))
        outc[:, np.arange(0,nxl*2,2)] = np.cos(out)
        outc[:, np.arange(1,nxl*2,2)] = np.sin(out)
        outc[np.isnan(outc)]=0
        outc = np.asarray(outc,dtype=np.float32)
        
        im1 = ou.clone()
        im1.scheme = 'BSQ'
        im1.width = nxl
        im1.filename = 'snaphu/maskfill.int'
        im1.dump('snaphu/maskfill.int.xml')
        outc.tofile(im1.filename)
        
        # Filter the mask fill
        command = 'python /home/kdm95/Software/isce2/contrib/stack/topsStack/FilterAndCoherence.py -i snaphu/maskfill.int -f snaphu/snaphu.in -s 0.5'
        os.system(command)
        
        os.chdir('snaphu')
        command='/home/insar/OUR_BIN/LIN/snaphu -f snaphu.conf'
        os.system(command)
        os.chdir('..')
        # Save an xml file for snaphu.out and load it in as b
        out = isceobj.createIntImage() # Copy the interferogram image from before
        out.scheme =  'BIP' #'BIP'/ 'BIL' / 'BSQ' 
        out.dataType = 'FLOAT'
        out.filename = 'snaphu/snaphu.out'
        out.width = params['nxl']
        out.length = params['nyl']
        out.dump(out.filename + '.xml') # Write out xml
        out.renderHdr()
        out.renderVRT()
        out.load(out.filename + '.xml')
        b = out.memMap()[:,:,0]
         
        #add 2pis to unfiltered

        im = out.clone()
        im.load(intfile + '.xml')
        a = im.memMap()[:,:,0]
        intfile_2pi = np.round((b-a)/2/np.pi)
        intfile_unw = a+2*np.pi*intfile_2pi  
        
        
        out.filename = intfile_unw_file
        out.dump(out.filename + '.xml')
        intfile_unw.tofile(out.filename)
        out.renderHdr()
        out.renderVRT()
        
    else:
        print('done unwrapping ' + intfile)
        
