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
import matplotlib.pyplot as plt
import math

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
    hmi_image = fits.getdata(url_hmi_cea)
    
    phi_c = keys_HMI ['CRVAL1'][0]
    lambda_c = keys_HMI['CRVAL2'][0]
    dx = keys_HMI['CDELT1'][0]
    dy = keys_HMI['CDELT2'][0]
    nx = hmi_image.shape[1]
    ny = hmi_image.shape[0]
    
    ds = '{}[{}]'.format(jsoc_serie, rec_time)
    avaiable_wavelengths = [94,131,171,193,211,304,335]
    for wlength in wavelength:
        if wlength not in avaiable_wavelengths:
            raise RuntimeError('Wavelength not avaiable!')
    ds = ds + str(wavelength)
    keys_AIA, segments_AIA = c.query(ds, key=drms.const.all, seg='image')
    # The all aia urls
    urls_aia = 'http://jsoc.stanford.edu' + segments_AIA['image'] 
    xi, eta, lat, lon = find_cea_coord(keys_AIA, phi_c, lambda_c,
                                       nx, ny, dx, dy)
    aia_data_dict = dict()
    for chnl in range(len(wavelength)):
        url = urls_aia[chnl]
        image_file = download_file(url, cache=False)
        aia_image = aiaprep(Map(image_file))
        if doCut:
            aia_img_data = cut_AIA(xi, eta, aia_image.data)
        else:
            aia_img_data = aia_image.data
        wavelnth = str(int(aia_image.wavelength.value))
        aia_data_dict[wavelnth] = aia_img_data

    return aia_data_dict


def cut_AIA(xi, eta, aia_image_data):
    nx = xi.shape[0]
    ny = xi.shape[1]
    cropped_aia_image_data = np.zeros((nx, ny))
    for i in np.arange(0, nx):
        for j in np.arange(0, ny):
            cropped_aia_image_data[i, j] = aia_image_data[int(eta[i,j]), 
                                  int(xi[i, j])]
    
    return cropped_aia_image_data



def find_cea_coord(header, phi_c, lambda_c, nx, ny, dx, dy):
    '''
    Convert the cutout index to CCD coordinate [xi, eta]
    
    Parameters:
    -------------------------------
    header: FITS file header = fits.open(fname)[1].header
        The header of the AIA fits file, containing the basic info



    Returns:
    -------------------------------
    
    Notes: header maybe not a good value to be passed in, 
    
    
    '''
    dtor = 0.0174533
    nx = int(nx)
    ny = int(ny)

    # Array of CEA coordinates
    x = np.zeros((nx, ny))
    y = np.zeros((nx, ny))
    for i in np.arange(0, nx):
        x[i, :] = (i - (nx - 1.0) / 2.0) * dx * dtor # Stonyhurst rad 
    for j in np.arange(0, ny):
        y[:, j] = (j - (ny - 1.0) / 2.0) * dy * dtor

    # Temporary vars from AIA header file
    rSun = header['RSUN_OBS'][0] / header['CDELT1'][0]
    disk_latc = header['CRLT_OBS'][0] * dtor
    disk_lonc = header['CRLN_OBS'][0] * dtor
    disk_xc = header['CRPIX1'][0] - 1.0
    disk_yc = header['CRPIX2'][0] - 1.0
    pa = header['CROTA2'][0] * (-1.0) * dtor
    print('SanityCheck:', rSun, disk_latc/dtor, disk_lonc/dtor, disk_xc, disk_yc, pa/dtor)
    
    latc = lambda_c * dtor
    lonc = phi_c * dtor - disk_lonc

    # Convert coordinate
    lat = np.zeros((nx, ny))
    lon = np.zeros((nx, ny))
    for i in np.arange(0, nx):
        for j in np.arange(0, ny):
            lat[i, j], lon[i, j] = plane2sphere(x[i, j], y[i, j], latc, lonc)

    xi = np.zeros([nx, ny])
    eta = np.zeros([nx, ny])
    for i in np.arange(0, nx):                                                                                     
        for j in np.arange(0, ny):
            xi[i, j], eta[i, j] = sphere2img(lat[i, j], lon[i, j], disk_latc,
                                             0.0, disk_xc, disk_yc, rSun, pa)
    return xi, eta, lat, lon


def plane2sphere(x, y, latc, lonc):
    '''
    Convert (x, y) of CEA map to Stonyhurst/Carrington (lat, lon)

    Parameters:
    --------------------------------
    x: float
        Standard CEA coordinate x

    y: float
        Standard CEA coordinate y

    latc: float
        HMI patch center Heliographic latitude \lambda_c

    lonc: float
        HMI patch center Heliographic longitude \phi_c
    
    Returns:
    --------------------------------
    lat: float
        Corresponding Heliographic latitude \lambda
    
    lon: float
        Corresponding Heliographic latitude \phi
    '''

    if abs(y) > 1:
        raise ValueError('The y in CEA coordinate is larger than 1')

    coslatc = math.cos(latc)
    sinlatc = math.sin(latc)

    cosphi = math.sqrt(1.0 - y*y)

    lat = math.asin ((y * coslatc) + (cosphi * math.cos(x) * sinlatc))

    if math.cos(lat) == 0:
        tmp_var = 0
    else:
        tmp_var = cosphi * math.sin(x) / math.cos(lat)
    lon = math.asin(tmp_var) + lonc
    
    return lat, lon


def sphere2img(lat, lon, latc, lonc, xcenter, ycenter, rsun, peff):
    '''
    Convert Stonyhurst lat, lon to xi, eta in heliocentric-cartesian coord

    Parameters:
    -------------------------------
    lat, lon
        latitude and longitude of desired pixel, in radian
    latc, lonc
        latitude and longitude of disc center, in radian
    rsun
        rsun: radius of Sun, arbiturary unit

    Returns:
    -------------------------------
    xi, eta
        coordinate on image, in unit of rsun
    '''
    # correction of finite distance (1 AU)
    sin_asd = 0.004660
    cos_asd = 0.99998914
    last_latc = 0.0
    cos_latc = 1.0
    sin_latc = 0.0

    if (latc != last_latc):
        sin_latc = math.sin(latc)
        cos_latc = math.cos(latc)
        last_latc = latc

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    cos_lat_lon = cos_lat * math.cos(lon - lonc)

    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    r = rsun * cos_asd / (1.0 - cos_cang * sin_asd)
    xr = r * cos_lat * math.sin(lon - lonc)
    yr = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon)

    cospa = math.cos(peff)
    sinpa = math.sin(peff)
    xi = xr * cospa - yr * sinpa
    eta = xr * sinpa + yr * cospa

    xi = xi + xcenter
    eta = eta + ycenter

    return xi, eta

#The test code

aia_data_dict = download_AIA_data('2016.08.01_00:00:00_TAI', 
                                        doCut= True, wavelength=[94])

plt.imshow(aia_data_dict['94'], cmap='copper')
