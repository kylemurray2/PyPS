#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Modified from Fringe: tops2vrt.py
Takes a bounding box and makes vrt files for the slcs and geom files
so that we only do the analysis on the area of interest.


'''
import numpy as np
import os
import glob
import datetime
from osgeo import gdal
import argparse


# ds = gdal.Open('/d/HI/S1/Asc/Oahu/Fringe/Sequential/compressedSlc/20171220/20171220.slc.vrt')
# slc = ds.GetVirtualMemArray()

# ps = np.load('./ps.npy',allow_pickle=True).all()

if not os.path.isdir(ps.workdir + '/vrts'):
    os.mkdir(ps.workdir + '/vrts')
    os.mkdir(ps.workdir + '/vrts/slcs')
    os.mkdir(ps.workdir + '/vrts/geom')
    

 
def radarGeometryTransformer(latfile, lonfile, epsg=4326):
    '''
    Create a coordinate transformer to convert map coordinates to radar image line/pixels.
    '''

    driver = gdal.GetDriverByName('VRT')
    inds = gdal.OpenShared(latfile, gdal.GA_ReadOnly)
    tempds = driver.Create('', inds.RasterXSize, inds.RasterYSize, 0)
    inds = None

    tempds.SetMetadata({'SRS' : 'EPSG:{0}'.format(epsg),
                        'X_DATASET': lonfile,
                        'X_BAND' : '1',
                        'Y_DATASET': latfile,
                        'Y_BAND' : '1',
                        'PIXEL_OFFSET' : '0',
                        'LINE_OFFSET' : '0',
                        'PIXEL_STEP' : '1',
                        'LINE_STEP' : '1'}, 'GEOLOCATION')

    trans = gdal.Transformer( tempds, None, ['METHOD=GEOLOC_ARRAY'])

    return trans

def lonlat2pixeline(lonFile, latFile, lon, lat):

    trans = radarGeometryTransformer(latFile, lonFile)

    ###Checkour our location of interest
    success, location = trans.TransformPoint(1, lon, lat, 0.)
    if not success:
        print('Location outside the geolocation array range')

    return location


def getLinePixelBbox(geobbox, latFile, lonFile):

    south,north, west, east = geobbox

    se = lonlat2pixeline(lonFile, latFile, east, south)
    nw = lonlat2pixeline(lonFile, latFile, west, north)

    ymin = np.int(np.round(np.min([se[1], nw[1]])))
    ymax = np.int(np.round(np.max([se[1], nw[1]])))

    xmin = np.int(np.round(np.min([se[0], nw[0]])))
    xmax = np.int(np.round(np.max([se[0], nw[0]])))

    print("x min-max: ", xmin, xmax)
    print("y min-max: ", ymin, ymax)

    return ymin, ymax, xmin, xmax


def runMain(inps):
    ##Parse command line
    inps = argparse.Namespace()
    inps.indir   = '../merged/'
    inps.stackdir= 'coreg_stack'
    inps.geomdir = 'geometry'
    inps.slcs    = 'slcs'


    ###Get ann list and slc list
    slclist = glob.glob(os.path.join(inps.indir,'SLC','*','*.slc.full'))
    num_slc = len(slclist)

    print('number of SLCs discovered: ', num_slc)
    print('we assume that the SLCs and the vrt files are sorted in the same order')

    slclist.sort()


    ###Read the first ann file to get some basic things like dimensions
    ###Just walk through each of them and create a separate VRT first
    if not os.path.exists(inps.outdir):
        print('creating directory: {0}'.format(inps.outdir))
        os.makedirs(inps.outdir)
    else:
        print('directory "{0}" already exists.'.format(inps.outdir))

    data = []
    dates = []

    width = None
    height = None

    print('write vrt file for each SLC ...')
    for ind, slc in enumerate(slclist):

        ###Parse the vrt file information.
        metadata = {}
        width = None
        height = None
        path = None

        ds = gdal.Open(slc , gdal.GA_ReadOnly)
        width = ds.RasterXSize
        height = ds.RasterYSize
        ds = None

        metadata['WAVELENGTH'] = 0.05546576
        metadata['ACQUISITION_TIME'] = os.path.basename(os.path.dirname(slc))

        path = os.path.abspath(slc)

        tag = metadata['ACQUISITION_TIME']

        vrttmpl='''<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">
        <sourceFilename>{PATH}</sourceFilename>
        <ImageOffset>0</ImageOffset>
        <PixelOffset>8</PixelOffset>
        <LineOffset>{linewidth}</LineOffset>
        <ByteOrder>LSB</ByteOrder>
    </VRTRasterBand>
</VRTDataset>'''


#        outname =  datetime.datetime.strptime(tag.upper(), '%d-%b-%Y %H:%M:%S UTC').strftime('%Y%m%d')

        outname = metadata['ACQUISITION_TIME']
        out_file = os.path.join(inps.outdir, '{0}.vrt'.format(outname))
        print('{} / {}: {}'.format(ind+1, num_slc, out_file))
        with open(out_file, 'w') as fid:
            fid.write( vrttmpl.format(width=width,
                                     height=height,
                                     PATH=path,
                                     linewidth=8*width))

        data.append(metadata)
        dates.append(outname)


    ####Set up single stack file
    if os.path.exists( inps.stackdir):
        print('stack directory: {0} already exists'.format(inps.stackdir))
    else:
        print('creating stack directory: {0}'.format(inps.stackdir))
        os.makedirs(inps.stackdir)

    latFile = os.path.join(inps.indir, "geom_reference", "lat.rdr.full.vrt")
    lonFile = os.path.join(inps.indir, "geom_reference", "lon.rdr.full.vrt")

    # setting up a subset of the stack
    if inps.geobbox:
        # if the bounding box in geo-coordinate is given, this has priority
        print("finding bbox based on geo coordinates of {} ...".format(inps.geobbox))
        ymin, ymax, xmin, xmax = getLinePixelBbox(inps.geobbox, latFile, lonFile)

    elif inps.bbox:
        # if bbox in geo not given then look for line-pixel bbox
        print("using the input bbox based on line and pixel subset")
        ymin, ymax, xmin, xmax = inps.bbox

    else:
        # if no bbox provided, the take the full size
        ymin, ymax, xmin, xmax = [0 , height, 0 , width]

    xsize = xmax - xmin
    ysize = ymax - ymin

    slcs_base_file = os.path.join(inps.stackdir, 'slcs_base.vrt')
    print('write vrt file for stack directory')
    with open(slcs_base_file, 'w') as fid:
        fid.write( '<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">\n'.format(xsize=xsize, ysize=ysize))

        for ind, (date, meta) in enumerate( zip(dates, data)):
            outstr = '''    <VRTRasterBand dataType="CFloat32" band="{index}">
        <SimpleSource>
            <SourceFilename>{path}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="CFloat32"/>
            <SrcRect xOff="{xmin}" yOff="{ymin}" xSize="{xsize}" ySize="{ysize}"/>
            <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
        </SimpleSource>
        <Metadata domain="slc">
            <MDI key="Date">{date}</MDI>
            <MDI key="Wavelength">{wvl}</MDI>
            <MDI key="AcquisitionTime">{acq}</MDI>
        </Metadata>
    </VRTRasterBand>\n'''.format(width=width, height=height,
                                xmin=xmin, ymin=ymin,
                                xsize=xsize, ysize=ysize,
                                date=date, acq=meta['ACQUISITION_TIME'],
                                wvl = meta['WAVELENGTH'], index=ind+1,
                                path = os.path.abspath( os.path.join(inps.outdir, date+'.vrt')))
            fid.write(outstr)

        fid.write('</VRTDataset>')

    ####Set up latitude, longitude and height files

    if os.path.exists( inps.geomdir):
        print('directory {0} already exists.'.format(inps.geomdir))
    else:
        print('creating geometry directory: {0}'.format(inps.geomdir))
        os.makedirs( inps.geomdir)


    vrttmpl='''<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <VRTRasterBand dataType="Float64" band="1">
      <SimpleSource>
        <SourceFilename>{PATH}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="Float64"/>
        <SrcRect xOff="{xmin}" yOff="{ymin}" xSize="{xsize}" ySize="{ysize}"/>
        <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
      </SimpleSource>
    </VRTRasterBand>
</VRTDataset>'''

    print('write vrt file for geometry dataset')
    layers = ['lat','lon','hgt','los','shadowMask','incLocal']
    for ind, val in enumerate(layers):
        with open( os.path.join(inps.geomdir, val+'.vrt'), 'w') as fid:
            fid.write( vrttmpl.format( xsize = xsize, ysize = ysize,
                                       xmin = xmin, ymin = ymin,
                                       width = width,
                                       height = height,
                                       PATH = os.path.abspath( os.path.join(inps.indir, 'geom_reference', val+'.rdr.full.vrt')),
                                       linewidth = width * 8))

