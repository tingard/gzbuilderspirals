import numpy as np
from astropy.wcs import WCS
from skimage.transform import rotate, rescale
from skimage.util import crop


def get_fits_name(galaxy):
    # Lookup the source fits file (needed for the rotation matrix)
    return 'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'.format(
        int(galaxy['RUN']),
        int(galaxy['CAMCOL']),
        int(galaxy['FIELD'])
    )


def create_WCS_object(galaxy, image_size, file_name=None):
    # Load a WCS object from the FITS image
    wcs_fits = WCS(
        file_name if file_name is not None else get_fits_name(galaxy)
    )

    # The SDSS pixel scale is 0.396 arc-seconds
    try:
        fits_cdelt = wcs_fits.wcs.cdelt
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
    if wcs_fits.wcs.has_pc():
        print('[createWCSObject] Using PC')
        # great, we have a pc rotation matrix, which we don't need to change
        # edit the CDELT pixel scale values only
        w.wcs.cdelt[0] /= fits_cdelt[0] * scale
        w.wcs.cdelt[1] /= fits_cdelt[1] * scale
    elif wcs_fits.wcs.has_cd():
        # fits is using the older standard of PCi_j rotation. Edid this 2d
        # matrix with the new scale values
        print('[createWCSObject] Using CD')
        w.wcs.cd = [
            wcs_fits.wcs.cd[0] / fits_cdelt[0] * scale,
            wcs_fits.wcs.cd[1] / fits_cdelt[1] * scale
        ]
    elif wcs_fits.wcs.has_crota():
        print('[createWCSObject] changing cdelt')
        w.wcs.cdelt[0] /= fits_cdelt[0] * scale
        w.wcs.cdelt[1] /= fits_cdelt[1] * scale
    else:
        print('Could not edit pixel scale')
    return w


def get_rotation_matrix(phi):
    return [
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]
    ]


def get_angle(gal, fits_name, image_size=np.array([512, 512])):
    wFits = WCS(fits_name)
    # edit to center on the galaxy
    wFits.wcs.crval = [float(gal['RA']), float(gal['DEC'])]
    wFits.wcs.crpix = image_size

    r = 4 * float(gal['PETRO_THETA']) / 3600
    phi = float(gal['PETRO_PHI90'])

    center_pix, dec_line = np.array(wFits.all_world2pix(
        [gal['RA'].iloc[0], gal['RA'].iloc[0]],
        [gal['DEC'].iloc[0], gal['DEC'].iloc[0] + r],
        0
    )).T

    rot = get_rotation_matrix(np.deg2rad(phi))
    vec = np.dot(rot, dec_line - center_pix)
    rotation_angle = 90 - np.rad2deg(np.arctan2(vec[1], vec[0])) - 90
    return rotation_angle


def deproject_array(arr, angle=0, ba=1):
    ba = min(ba, 1/ba)
    rotated_image = rotate(arr, angle)
    stretched_image = rescale(
        rotated_image,
        (1/ba, 1),
        mode='constant',
        anti_aliasing=True,
        multichannel=False
    )
    crop_amounts = np.repeat(
        np.subtract(stretched_image.shape, arr.shape),
        2
    ).reshape(2, 2) / 2

    return crop(stretched_image, crop_amounts)


def deproject_arm(arm, angle=0, ba=1):
    """Given an array of xy pairs, an axis ratio, and a rotation angle, rotate
    the points about the origin and scale outwards along the y axis
    """
    p = np.deg2rad(-angle)
    rotation_matrix = np.array(
        ((np.cos(p), -np.sin(p)), (np.sin(p), np.cos(p)))
    )
    rotated_arm = np.dot(rotation_matrix, arm.T)
    stretched_arm = rotated_arm.T * (1, 1/ba)
    return stretched_arm


def reproject_arm(arm, angle=0, ba=1):
    return deproject_arm(
        deproject_arm(arm, 0, 1/ba),
        -angle, 1
    )
