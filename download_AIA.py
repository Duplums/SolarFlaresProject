#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:26:29 2018

@author: xtwang
"""
import drms
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from sunpy.map import Map
from sunpy.instr.aia import aiaprep
import sunpy.wcs as wcs
import sunpy.coordinates
def download_AIA_data(rec_time, doCut = True, jsoc_serie = 'aia.lev1_euv_12s',
                      wavelength = [94,131,171,193,211,304,335]):
    '''
    Download the AIA data from a given time which is same to the HMI data in
    this project.
    
    Parameters:
    ----------------------------
    rec_time: String (in JSOC time format) 
        The HMI record time, locate the corresponding AIA data, notice 
        that the record time maybe not exactly the same
    
    doCut: Bool (Default is True)
        Control if the AIA data is cropped
    
    jsoc_serie: String    
        The jsoc series to get the AIA data, 
        default is 'aia.lev1_euv_12s'
    
    wavelength: List
        The wavelengths for the AIA data, 
        all avaiable are [94,131,171,193,211,304,335]
       
    Returns:
    ----------------------------  
    For now, the return is a dictionary, data part are image datas,
    keys are channels (wavelength). [Output HDF5 files in the future.]
    '''
    
    c = drms.Client()
    
    # Query the HMI data, because can't directly get the record time from the
    # HMI fits files
    ds_HMI = '{}[{}]'.format('hmi.sharp_cea_720s[1-7256]', rec_time)
    keys_HMI, segments_HMI = c.query(ds_HMI,
                                     key=drms.const.all, seg='magnetogram')
    url_hmi_cea = 'http://jsoc.stanford.edu' + segments_HMI.magnetogram[0]
    hmi_image = fits.open(url_hmi_cea)
    
    ds = '{}[{}]'.format(jsoc_serie, rec_time)
    avaiable_wavelengths = [94,131,171,193,211,304,335]
    for wlength in wavelength:
        if wlength not in avaiable_wavelengths:
            raise RuntimeError('Wavelength not avaiable!')
    ds = ds + str(wavelength)
    keys_AIA, segments_AIA = c.query(ds, key=drms.const.all, seg='image')
    # The all aia urls
    urls_aia = 'http://jsoc.stanford.edu' + segments_AIA['image'] 
    
    aia_data_dict = dict()
    for chnl in range(len(wavelength)):
        url = urls_aia[chnl]
        image_file = download_file(url, cache=False)
        aiamap = aiaprep(Map(image_file))
        fullsize_aia_image_data = np.array(aiamap.data, dtype=np.int16)
        if doCut:
            aia_image_data = cut_AIA(keys_AIA.loc[chnl], keys_HMI,
                                     fullsize_aia_image_data, hmi_image)
        else:
            aia_image_data = fullsize_aia_image_data
        wavelnth = str(int(aiamap.wavelength.value))
        aia_data_dict[wavelnth] = aia_image_data

    return aia_data_dict,keys_AIA, keys_HMI

def cut_AIA(keys_AIA, keys_HMI, aia_image_data, hmi_image):
    '''
    Identifying which AIA pixels ae included in the CEA-coordinate SHARP
    bounding box.
    
    Parameters:
    ----------------------------    
    keys_AIA: String List
        The AIA data keys, for the data crop
    
    keys_HMI: String List
        The HMI data keys, fot the data crop
    
    aia_image_data: Numpy array (2-dimensional)
        The AIA image data, size 4096*4096, after doing aiaprep

    hmi_image: Astropy fits file object
        The fits file object of the HMI data

    Returns:
    ----------------------------
    cropped_aia_image_data: Numpy array (2-dimensional)
        The cropped AIA data
    '''
    XDIM_CEA = hmi_image[1].data.shape[1]
    YDIM_CEA = hmi_image[1].data.shape[0]
    aia_mask = np.full([4096, 4096], np.nan)

    for j in range(int(YDIM_CEA)):
        for i in range(int(XDIM_CEA)):
            x_hg = keys_HMI.CRVAL1[0] - 0.5*XDIM_CEA*keys_HMI.CDELT1[0]\
                + i*keys_HMI.CDELT1[0]
            y_hg = keys_HMI.CRVAL2[0] - 0.5*YDIM_CEA*keys_HMI.CDELT1[0]\
                + j*keys_HMI.CDELT1[0]
            HPC_out = wcs.convert_hg_hpc(x_hg, y_hg,
                                         b0_deg=keys_HMI.CRLT_OBS[0],
                                         l0_deg=keys_HMI.CRLN_OBS[0])
            x_aia = int(((HPC_out[0])/keys_AIA.CDELT1) + keys_AIA.CRPIX1)
            y_aia = int(((HPC_out[1])/keys_AIA.CDELT1) + keys_AIA.CRPIX2)
            aia_mask[y_aia, x_aia] = 1.0

    cropped_aia_image_data = aia_image_data * aia_mask
    
    return cropped_aia_image_data





