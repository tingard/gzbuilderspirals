from tempfile import NamedTemporaryFile
import numpy as np
from astropy.wcs import WCS
from skimage.transform import rotate, rescale
import astropy.units as u
import re
import subprocess

def getFitsName(galaxy):
    # Lookup the source fits file (needed for the rotation matrix)
    return 'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'.format(
        int(galaxy['RUN']),
        int(galaxy['CAMCOL']),
        int(galaxy['FIELD'])
    )


def createWCSObject(galaxy, imageSize, fileName=None):
    # Load a WCS object from the FITS image
    wFits = WCS(fileName if fileName is not None else getFitsName(galaxy))

    # The SDSS pixel scale is 0.396 arc-seconds
    try:
        fits_cdelt = wFits.wcs.cdelt
    except AttributeError:
        fits_cdelt = [0.396 / 3600]*2

    # cutouts were chosen to be 4x Petrosean radius, and then scaled
    # (including interpolation) to be 512x512 pixels
    scale = 4 * (float(galaxy['PETRO_THETA']) / 3600) / 512

    # This should be obtained from the image, as some were not square.
    size_pix = np.array([512, 512])

    # Create a new WCS object
    w = WCS(naxis=2)
    w.wcs.crpix = size_pix / 2
    w.wcs.crval = np.array([float(galaxy['RA']), float(galaxy['DEC'])])
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']

    # Copy the rotation matrix from the source FITS file, adjusting the
    # scaling as needed
    print('[createWCSObject] Checking for transformation')
    if wFits.wcs.has_pc():
        print('[createWCSObject] Using PC')
        # great, we have a pc rotation matrix, which we don't need to change
        # edit the CDELT pixel scale values only
        w.wcs.cdelt[0] /= fits_cdelt[0] * scale
        w.wcs.cdelt[1] /= fits_cdelt[1] * scale
    elif wFits.wcs.has_cd():
        # fits is using the older standard of PCi_j rotation. Edid this 2d
        # matrix with the new scale values
        print('[createWCSObject] Using CD')
        w.wcs.cd = [
            wFits.wcs.cd[0] / fits_cdelt[0] * scale,
            wFits.wcs.cd[1] / fits_cdelt[1] * scale
        ]
    elif wFits.wcs.has_crota():
        print('[createWCSObject] changing cdelt')
        w.wcs.cdelt[0] /= fits_cdelt[0] * scale
        w.wcs.cdelt[1] /= fits_cdelt[1] * scale
    else:
        print('Could not edit pixel scale')
    return w


def getRotationMatrix(phi):
    return [
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]
    ]


def getAngle(gal, fitsName, imageSize=np.array([512, 512])):
    wFits = WCS(fitsName)
    # edit to center on the galaxy
    wFits.wcs.crval = [float(gal['RA']), float(gal['DEC'])]
    wFits.wcs.crpix = imageSize

    r = 4 * float(gal['PETRO_THETA']) / 3600
    ba = float(gal['SERSIC_BA'])
    phi = float(gal['SERSIC_PHI'])

    centerPix, decLine = np.array(wFits.all_world2pix(
        [gal['RA'].iloc[0], gal['RA'].iloc[0]],
        [gal['DEC'].iloc[0], gal['DEC'].iloc[0] + r],
        0
    )).T

    rot = getRotationMatrix(np.deg2rad(phi))

    vec = np.dot(rot, decLine - centerPix)
    galaxyAxis = vec + centerPix
    rotationAngle = 90 - np.rad2deg(np.arctan2(vec[1], vec[0])) - 90
    return rotationAngle


def deprojectArray(array, angle=0, ba=1):
    rotatedImage = rotate(array, angle)
    stretchedImage = rescale(rotatedImage, (1, 1 / ba))
    n = int((stretchedImage.shape[1] - array.shape[1]) / 2)

    if n > 0:
        deprojectedImage = stretchedImage[:, n:-n]
    else:
        deprojectedImage = stretchedImage.copy()
    return deprojectedImage
