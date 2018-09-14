import numpy as np
from astropy.wcs import WCS
from skimage.transform import rotate, rescale


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
    fits_cdelt = 0.396 / 3600

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
    w.wcs.cd = [
        wFits.wcs.cd[0] / fits_cdelt * scale,
        wFits.wcs.cd[1] / fits_cdelt * scale
    ]
    return w


def getAngle(galaxy, wcs, imageSize):
    r = float(galaxy['PETRO_THETA']) / 3600
    phi = float(galaxy['SERSIC_PHI'])

    cx, cy = imageSize / 2
    wCx, wCy = float(galaxy['RA']), float(galaxy['DEC'])

    # find our line in world coordinates
    x = r * np.sin(np.deg2rad(phi)) + wCx
    y = r * np.cos(np.deg2rad(-phi)) + wCy

    v = np.array([x - wCx, y - wCy])
    v /= np.linalg.norm(v)

    ra_line, dec_line = wcs.wcs_world2pix([wCx, x], [wCy, y], 0)

    axis_vector = np.subtract.reduce(
        np.stack(
            (ra_line, dec_line),
            axis=1
        )
    )

    angle = 180 * np.arccos(
        axis_vector[1] / np.linalg.norm(axis_vector)
    ) / np.pi
    return angle


def deprojectArray(array, angle=0, ba=1):
    rotatedImage = rotate(array, angle)
    stretchedImage = rescale(rotatedImage, (1, 1 / ba))
    n = int((stretchedImage.shape[1] - array.shape[1]) / 2)

    if n > 0:
        deprojectedImage = stretchedImage[:, n:-n]
    else:
        deprojectedImage = stretchedImage.copy()
    return deprojectedImage
