"""
----------------------------------------------------------------------------
crsutil.py    CRSUTIL Coordinate and Time Reference System Toolbox.
Version 1.1 (14 November 2020).
Created by: Hans van der Marel, Ullas Rajvanshi and Simon van Diepen
Date:       14 Nov 2020
Modified:   -

Copyright: Hans van der Marel, Ullas Rajvanshi, Simon van Diepen, Delft University of Technology
Email:     h.vandermarel@tudelft.nl
Github:    -
----------------------------------------------------------------------------
Functions:

Coordinate transformations (ECEF reference frame)

  xyz2plh     - Cartesian Coordinates to Ellipsoidal coordinates
  plh2xyz     - Ellipsoidal coordinates to Cartesian Coordinates
  inqell      - Semi-major axis, flattening and GM for various ellipsoids

UT1 to GMST, and ECI/ECEF, conversions

  ut2gmst    - Compute Greenwich Mean Siderial Time from UT1
  ecef2eci   - Convert position and velocity from ECEF to ECI reference frame
  eci2ecef   - Convert position and velocity from ECI to ECEF reference frame

Keplerian elements

  vec2orb     - Convert inertial state vector into Keplerian elements
  orb2vec     - Convert Keplerian elements into inertial state vector
  kepler      - Compute mean anomaly from eccentric anomaly (Kepler's equation)
  keplernu    - Compute mean anomaly from true anomaly (Kepler's equation)
  keplerm     - Compute eccentric/true from mean anomaly solving Kepler's eqn

Others

  datetime2num - Convert Python datetime to sequential datenumber

----------------------------------------------------------------------------
"""

"""
Planned ...

  xyz2neu     - North, East, Up (dN, dE, dU) to Cartesian delta's (dX, dY, dZ)
  neu2xyz     - Cartesian delta's (dX, dY, dZ) to North, East, Up (dN, dE, dU)
  plh2neu     - Ellipsoidal (Lat,Lon,Hgt) to North,East,Up (dN, dE, dU)
  xyz2zas     - Cartesian coordinates to Zenith angle, azimuth and distance
  zas2xyz     - Zenith angle, azimuth and distance to cartesian coordinates
"""

# Importing the Libraries
import numpy as np
from datetime import datetime
from dateutil.parser import parse as parsedate


def datetime2num(dt, matlab=False):
    ordnum = dt.toordinal()
    frac_seconds = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0 )
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    if matlab:
        # datenum is a Matlab datenumber (days since '00-Jan-0000')
        datenum = ordnum + frac_seconds + frac_microseconds + 366.0 
    else:
        # datenum is matplotlib datenumber (days since '01-Jan-1970')
        datenum = ordnum + frac_seconds + frac_microseconds - 719163.0
    return datenum


def num2datetime(datenum, matlab=False):

    if matlab:
        # datenum is a Matlab datenumber (days since '00-Jan-0000')
        ordnum = datenum - 366.0 
    else:
        # datenum is matplotlib datenumber (days since '01-Jan-1970')
        ordnum = datenum + 719163.0

    ordfloor = np.floor(ordnum)
    ordfrac = ordnum - ordfloor
    ordnum = int(ordfloor)

    microsecond = ordfrac * 24 * 60 * 60 * 1000000
    second = int(ordfrac * 24 * 3600) % 60

    hour = int(ordfrac * 24)
    minute = int((ordfrac * 24 - hour ) * 60)  
    second = int((ordfrac * 24 * 60 - hour * 60 - minute ) * 60)
    microsecond = int(np.round((ordfrac * 24 * 60 * 60 - hour * 3600 - minute * 60 - second) * 1000 ))
    if microsecond > 999:
        second = second + 1
        microsecond = microsecond - 1000
        if second > 59:
            minute = minute + 1
            second = second -60
            if minute > 59:
                hour = hour + 1
                minute = minute - 60
                if hour > 23:
                    ordnum = ordnum + 1
                    hour = hour - 24
    
    dt = datetime.fromordinal(ordnum)
    dt = dt.replace(hour=hour, minute=minute, second=second, microsecond=microsecond*1000)
                
    return dt

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# ----------------------------------------------------------------------------
#                      GEODETIC FROM/TO CARTESIAN COORDINATES 
# ----------------------------------------------------------------------------
#
#   xyz2plh     - Cartesian Coordinates to Ellipsoidal coordinates
#   plh2xyz     - Ellipsoidal coordinates to Cartesian Coordinates
#   inqell      - Semi-major axis, flattening and GM for various ellipsoids


def inqell():
    ell = ['AIRY', 'BESSEL', 'CLARKE', 'INTERNATIONAL', 'HAYFORD', 'GRS80', 'WGS-84']

    par = np.array([
        [6377563.396, 299.324964, np.NaN],
        [6377397.155, 299.1528128, np.NaN],
        [6378249.145, 293.465, np.NaN],
        [6378388.0, 297.00, np.NaN],
        [6378388.0, 297.00, 3.986329e14],
        [6378137.0, 298.257222101, 3.986005e14],
        [6378137.0, 298.257223563, 3.986005e14]])

    i = 0

    for j in range(0, par.shape[0]):
        i = j
    if i == 0:
        i = par.shape[0]
    # module only taking WGS-84 at the moment by default
    # need to add a string comparison and based on that add the current parameter value
    a = par[i, 0]
    f = 1 / par[i, 1]
    GM = par[i, 2]

    return a, f, GM


def xyz2plh(xyz, ellipse='WGS-84', method=0):
    """
    #(c) Hans van der Marel, Delft University of Technology, 1995,2013
    :param xyz:Nx3 matrix XYZ with in the rows cartesian coordinates X, Y and Z
    :param ellipse: allows to specify the ellipsoid. ELLIPS is a text
    is a text string with the name of the ellipsoid or a vector with the
    semi-major axis a and flattening 1/f. Default for ellips is 'WGS-84'
    :param method:  uses the more conventional iterative method
    instead of Bowring's method (the default method). Bowring's method is
    faster, but can only be used on the surface of the Earth. The iterative
    method is slower and less precise on the surface of the earth, but should
    be used above 10-20 km of altitude.
    :return:  Nx3 matrix PLH with ellipsoidal coordinates
    Phi, Lambda and h. Phi and Lambda are in radians, h is in meters
    """

    a, f, GM = inqell()
    # excentricity e(squared) and semi - minor axis
    e2 = 2 * f - f ** 2
    b = (1 - f) * a
    [m, n] = xyz.shape

    if n == 3 and m == 3:
        xyz = xyz.transpose()

    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2);

    if method == 1:
        # compute phi via iteration
        Np = xyz[:, 2]
        for i in range(0, 4):
            phi = np.arctan((xyz[:, 2] + e2 * Np) / r)
            N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
            Np = N * np.sin(phi)

    else:
        # compute phi using B.R.Bowring's equation (default method)
        u = np.arctan2(xyz[:, 2] * a, r * b)
        phi = np.arctan2(xyz[:, 2] + (e2 / (1 - e2) * b) * np.sin(u) ** 3, r - (e2 * a) * np.cos(u) ** 3)
        N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)

    plh = np.array([phi, np.arctan2(xyz[:, 1], xyz[:, 0]), r / np.cos(phi) - N])
    plh = plh.transpose()

    return plh


def plh2xyz(plh, ellipse='WGS-84'):
    """

    :param plh:Nx3 matrix PLH with in the rows ellipsoidal coordinates Phi, Lambda and h
    :param ellipse: specify the ellipsoid. ELLIPS is a text is a text string with the name of the ellipsoid or a vector with the  semi-major axis a and flattening 1/f. Default for ellips is 'WGS-84'.
    :return:Nx3 matrix XYZ with cartesian coordinates X, Y and Z.
    """
    a, f, GM = inqell()
    # excentricity e(squared)
    e2 = 2 * f - f ** 2
    [m, n] = plh.shape

    if n == 3 and m == 3:
        plh = plh.transpose()

    N = a / np.sqrt(1 - e2 * np.sin(plh[:, 0]) ** 2)
    xyz = np.array([
        (N + plh[:, 2]) * np.cos(plh[:, 0]) * np.cos(plh[:, 1]),
        (N + plh[:, 2]) * np.cos(plh[:, 0]) * np.sin(plh[:, 1]),
        (N - e2 * N + plh[:, 2]) * np.sin(plh[:, 0])
    ])
    xyz = xyz.transpose()

    return xyz


# ----------------------------------------------------------------------------
#                            KEPLERIAN ELEMENTS
# ----------------------------------------------------------------------------
#
#   vec2orb     - Convert inertial state vector into Keplerian elements
#   orb2vec     - Convert Keplerian elements into inertial state vector
#   kepler      - Compute mean anomaly from eccentric anomaly (Kepler's equation)
#   keplernu    - Compute mean anomaly from true anomaly (Kepler's equation)
#   keplerm     - Compute eccentric/true from mean anomaly solving Kepler's eqn


def vec2orb(svec, GM=3986004418e5):
    """
    VEC2ORB   Convert inertial state vector into Keplerian elements.
    ORB=VEC2ORB(SVEC) converts a 6-element inertial state vector SVEC
    with cartesian position and velocity (X, Y, Z, Xdot, Ydot, Zdot) into
    a vector ORB with 6 Keplerian elements, with
          ORB(:,1)    Semi-major axis (meters),
          ORB(:,2)    Eccentricity (unity),
          ORB(:,3)    Inclination (radians),
          ORB(:,4)    Right ascension of the ascending node (radians),
          ORB(:,5)    Argument of the pericenter (radians),
          ORB(:,6)    True anomaly (radians).
    This routine is fully vectorized, if SVEC is a matrix then ORB will be
    a matrix of the same size. One of the dimensions of the input matrix must
    be 6. Units are meter, meter/sec or radians.

    ORB=VEC2ORB(SVEC,GM) provides an optional gravitational parameter of the
    central body. Default for GM [meter**3/sec**2] is the IERS 1996 standard
    value for the Earth (GM=3986004418e5)

    [ORB,CORBTYPE]=VEC2ORB(SVEC) also outputs a character array CORBTYPE
    with the orbit type, with
        'ei'   elliptical inclined     (all Kepler elements defined)
        'ci'   circular inclined       (w =0, nu=arglat)
        'ee'   elliptical equatorial   (w=lonper, omega=0)
        'ce'   circular equatorial     (w=0, omega=0, nu=truelon)
    orbits are "circular" if the eccentricity < eps and "equatorial" if
    the inclination < eps, with eps=1e-8.
    """
    if type(svec == list):
        svec = np.array([svec])
    [m, n] = svec.shape

    if n != 6 and m == 6:
        svec = svec.transpose()

    # Inner products (rrdot = R.V , r=sqrt(R.R) , vsq = V.V )

    rrdot = svec[:, 0] * svec[:, 3] + svec[:, 1] * svec[:, 4] + svec[:, 2] * svec[:, 5]
    r = np.sqrt(svec[:, 0] * svec[:, 0] + svec[:, 1] * svec[:, 1] + svec[:, 2] * svec[:, 2])
    vsq = svec[:, 3] * svec[:, 3] + svec[:, 4] * svec[:, 4] + svec[:, 5] * svec[:, 5]

    # Angular momentum vector (H = R x V)

    hx = svec[:, 1] * svec[:, 5] - svec[:, 2] * svec[:, 4]
    hy = svec[:, 2] * svec[:, 3] - svec[:, 0] * svec[:, 5]
    hz = svec[:, 0] * svec[:, 4] - svec[:, 1] * svec[:, 3]

    hsini2 = hx * hx + hy * hy
    hsq = hsini2 + hz * hz
    h = np.sqrt(hsq)

    # Semi-major axis

    ainv = 2 / r - vsq / GM
    a = 1. / ainv

    # Eccentricity

    ome2 = hsq * ainv / GM
    ecc = np.sqrt(1.0 - ome2)
    ecc[ome2 > 1] = 0  # special handling of negative values

    # Inclination (0...pi)

    incl = np.arccos(hz / h)
    # Determine orbit type (for handling of special cases)
    small = 1e-8
    # idxecc = ecc < small
    # idxincl = np.logical_or((incl < small), (abs(incl - np.pi) < small))
    # Appendinng this in future version in python :
    # idx_ce = ( idxecc & idxincl );      % circular equatorial => w=0, omega=0, nu=truelon
    # idx_ci = ( idxecc & ~idxincl );     % circular inclined => w =0, nu=arglat
    # idx_ee = ( ~idxecc & idxincl );     % elliptical equatorial => w=lonper, omega=0
    # idx_ei = ( ~idxecc & ~idxincl );    % elliptical inclined
    #
    # orbtype(idx_ei)=0;
    # orbtype(idx_ee)=1;
    # orbtype(idx_ci)=2;
    # orbtype(idx_ce)=3;
    #
    # corbdef=['ei';'ee';'ci';'ce'];
    # corbtype=corbdef(orbtype+1,:);

    # Standard handling of elliptical inclined orbits...

    # The computations below do not do special hanling of circular or equatorial
    # orbits. This is possible because atan2(0,0)=0 is defined in Matlab, however,
    # the some of the angles will actually be undefined.

    # Longitude of ascending node (0...2*pi)
    omega = np.arctan2(hx, -hy)
    idx = (omega < 0).nonzero()
    omega[idx] = omega[idx] + 2 * np.pi

    # True anomaly (0...2*pi)

    resinf = a * ome2 * rrdot / h
    recosf = a * ome2 - r
    nu = np.arctan2(resinf, recosf)
    idx = (nu < 0).nonzero()
    nu[idx] = nu[idx] + 2 * np.pi

    # Argument of perigee (0...2*pi)

    suprod = -hz * (svec[:, 0] * hx + svec[:, 1] * hy) + svec[:, 2] * hsini2
    cuprod = h * (-svec[:, 0] * hy + svec[:, 1] * hx)
    w = np.arctan2(suprod * recosf - cuprod * resinf, cuprod * recosf + suprod * resinf)
    idx = (w < 0).nonzero()
    w[idx] = w[idx] + 2 * np.pi

    orb = np.array([a, ecc, incl, omega, w, nu])
    orb = orb.transpose()

    return orb


def orb2vec(orb, GM=3986004418e5):
    """
    ORB2VEC   Convert Keplerian elements into inertial state vector.
    SVEC=ORB2VEC(ORB) converts the vector ORB with 6 Keplerian elements
    into the 6-element inertial state vector SVEC with cartesian position and
    velocity (X, Y, Z, Xdot, Ydot, Zdot). The Keplerian elements are
          ORB(:,1)    Semi-major axis (meters),
          ORB(:,2)    Eccentricity (unity),
          ORB(:,3)    Inclination (radians),
          ORB(:,4)    Right ascension of the ascending node (radians),
          ORB(:,5)    Argument of the pericenter (radians),
          ORB(:,6)    True anomaly (radians).
    This routine is fully vectorized, if ORB is a matrix then SVEC
    will be a matrix of the same size. One of the dimensions of the input
    matrix must be 6. The units are meter, meter/sec or radians.

    SVEC=ORB2VEC(ORB,GM) provides an optional gravitational parameter of the
    central body. Default for GM [meter**3/sec**2] is the IERS 1996 standard
    value for the Earth (GM=3986004418e5)

    See also VEC2ORB, KEPLER, KEPLERINV

    :param orb:
    :param GM:
    :return:
    """

    if type(orb) == list:
        orb = np.array([orb])

    m, n = orb.shape

    if n != 6 and m == 6:
        orb = orb.T
        m, n = n, m
        transpose = True
    else:
        transpose = False

    if n != 6:
        raise ValueError("Input must be a vector with 6 Keplerian elements, received {}!".format(n))

    # Compute position (rx,ry) and velocity (vx,vy) in orbital plane (perifocal system)

    ecc = orb[:, 1]                   # Eccentricity
    cosnu = np.cos(orb[:, 5])         # Cosine and sine of true anomaly (nu)
    sinnu = np.sin(orb[:, 5])

    p = orb[:, 0] * (1.0 - ecc ** 2)  # Parameter of the ellipse p=a*(1-e^2)

    r = p / (1.0 + ecc * cosnu)       # Length of position vector

    rx = r * cosnu                    # Position (rx,ry) in orbital plane
    ry = r * sinnu

    p[abs(p) < 0.0001] = 0.0001       # Protect against division by zero
    tmp = np.sqrt(GM / p)

    vx = -tmp * sinnu                 # Velocity (vx,vy) in orbital plane
    vy = tmp * (ecc + cosnu)

    # Convert into inertial frame (3-1-3 Euler rotations)

    cosincl = np.cos(orb[:, 2])       # Cosine and sine of inclination (incl)
    sinincl = np.sin(orb[:, 2])
    cosomega = np.cos(orb[:, 3])      # Cosine and sine of longitude of ascending node (omega)
    sinomega = np.sin(orb[:, 3])
    cosw = np.cos(orb[:, 4])          # Cosine and sine of argument of perigee (w)
    sinw = np.sin(orb[:, 4])

    rx0 = cosw * rx - sinw * ry       # Cosine and sine of argument of latitude u=w+nu
    ry0 = cosw * ry + sinw * rx

    vx0 = cosw * vx - sinw * vy
    vy0 = cosw * vy + sinw * vx

    #svec = np.vstack([rx0 * cosomega - ry0 * cosincl * sinomega,   # Simon   
    #svec = np.array([rx0 * cosomega - ry0 * cosincl * sinomega,    # Ullas
    svec = np.column_stack([rx0 * cosomega - ry0 * cosincl * sinomega,
                     rx0 * sinomega + ry0 * cosincl * cosomega,
                     ry0 * sinincl,
                     vx0 * cosomega - vy0 * cosincl * sinomega,
                     vx0 * sinomega + vy0 * cosincl * cosomega,
                     vy0 * sinincl])

    if transpose:
        svec = svec.T

    return svec


def kepler(E, ecc):
    """Compute mean anomaly from eccentric anomaly (Kepler's equation).
    
	M=kepler(E,ecc) computes the mean anomaly M [rad] from the eccentric
    anomaly E [rad] and eccentricity ecc. This routine is fully vectorized,
    if E is a vector, M will be a vector with the same dimensions.

    This routine should only be used for elliptical orbits. Parabolic and
    hyperbolic orbits are not supported and give false results (this is
    nowhere checked for).
    """

    M = E - ecc * np.sin(E)

    return M


def keplernu(nu, ecc):
    """Compute mean anomaly from true anomaly (Kepler's equation).

    M=keplernu(nu,ecc) computes the mean anomaly M [rad] from the true
    anomaly nu [rad] and eccentricity ecc. This routine is fully vectorized,
    if nu is a vector, M will be a vector with the same dimensions.

    [M,E]=keplernu(nu,ecc) returns also the eccentric anomaly E [rad].

    This routine should only be used for elliptical orbits. Parabolic and
    hyperbolic orbits are not supported and give false results (this is
    nowhere checked for).
    """

    denom = 1.0 + ecc * np.cos(nu)
    sine = (np.sqrt(1.0 - ecc * ecc) * np.sin(nu)) / denom
    cose = (ecc + np.cos(nu)) / denom
    E = np.arctan2(sine, cose)

    # Compute mean anomaly
    M = E - ecc * np.sin(E)

    return M, E


def keplerm(M, ecc, TOL=1e-10):
    """Compute eccentric/true from mean anomaly solving Kepler's eqn.
    
	E=KEPLERM(M,ECC) computes the eccentric anomaly E [rad] from the mean
    anomaly M [rad] and eccentricity ECC by solving Kepler's equation
    M=E-ECC*sin(E) iteratively using Newton's method. This routine is fully
    vectorized, if M is a vector, E will be a vector with the same dimensions.

    [E,NU]=KEPLERM(M,ECC) returns also the true anomaly NU [rad].

    E=KEPLERM(M,ECC,TOL) uses TOL as the cutoff criterion for the iterations,
    the default value is 1e-10.

    This routine should only be used for elliptical orbits. Parabolic and
    hyperbolic orbits are not supported and give false results (this is
    nowhere checked for).

    See also keplernu and kepler.
	
    :param M:
    :param ecc:
    :param T0L:
    :return:
    """
    
    E = M                            # Use M for the first value of E
    # [m, n] = E.shape
    f = np.ones(E.shape)              # Newton's method for root finding
    while max(abs(f)) > TOL:
        f = M - E + ecc * np.sin(E)  # Kepler's Equation
        fdot = -1 + ecc * np.cos(E)  # Derivative of Kepler's equation
        E = E - f / fdot

    sinnu = -1 * np.sqrt(1 - ecc ** 2) * np.sin(E) / fdot
    cosnu = (ecc - np.cos(E)) / fdot

    nu = np.arctan2(sinnu, cosnu)    # True anomaly

    return E, nu


# ----------------------------------------------------------------------------
#                 UT1 to GMST, and ECI/ECEF, conversions
# ----------------------------------------------------------------------------
#
#   ut2gmst    - Compute Greenwich Mean Siderial Time from UT1
#   ecef2eci   - Convert position and velocity from ECEF to ECI reference frame
#   eci2ecef   - Convert position and velocity from ECI to ECEF reference frame


def ut2gmst(ut1, model="IAU-82"):
    """Compute Greenwich Mean Siderial Time from UT1

    GMST=UT2GMST(UT1) returns the Greenwich Mean Siderial Time GMST
    [0-2pi rad] for UT1. UT1 is a matlab date number or matlab date string
    representing UT1.

    [GMST,OMEGAE]=UT2GMST(UT1) returns also the rotation rate of the
    Earth [rev/day].

    [...]=UT2GMST(...,MODEL) selects the computation method, possible choices
    for MODEL are APPROXIMATE and IAU-82. The default is the IAU-82 model.

    Examples:

      gmst=ut2gmst('2012-01-04 15:00:03')
      [gmst0,omegae]=ut2gmst('2012-01-04')
      gmst=ut2gmst(datenum)

    :param ut1:
    :param model:
    :return:
    """

    if type(ut1) == str:
        ut1 = datetime2num(parsedate(ut1))

    if model.upper() == "APPROXIMATE":
        t0 = datetime2num(datetime(2000, 1, 1, 12, 0, 0))
        gmst = (18.697374558 + 24.06570982441908 * (ut1 - t0)) % 24
        gmst *= 3600
    elif model.upper() == "IAU-82":
        j2000 = (ut1 - datetime2num(datetime(2000, 1, 1, 12, 0, 0)) ) / 36525.0
        gmst = ((- 6.2e-6 * j2000 + 0.093104) * j2000 + (876600.0 * 3600.0 + 8640184.812866)) * j2000 + 67310.54841
        gmst %= 86400
    else:
        raise ValueError("Unknown model!")

    gmst *= np.pi / 43200

    omegae = 1.0027379093

    return gmst, omegae


def ecef2eci(t, xsate, vsate=[]):
    """Convert position and velocity from ECEF to ECI reference frame.

    xsat, vsat = ecef2eci(t,xsate,vsate) converts cartesian coordinates xsate
    and velocities vsate in Earth Centered Earth Fixed (ECEF) reference frame,
    with t the time (UT1) given as Matlab datenumbers, into cartesian
    coordinates xsat and velocities vsat in Earth Centered Inertial (ECI)
    reference frame. t is a vector with length n, the number of epochs, and
    xsate and vsate are n-by-3 matrices.

    xsat, vsat = ecef2eci(t,xsate) assumes the velocity in ECEF is zero.

    See also ut2gmst and eci2ecef.
    """
    
    """
    (c) Hans van der Marel, Delft University of Technology, 2016-2020.

    Created:  26 November 2016 by Hans van der Marel
    Modified: 18 November 2020 by Simon van Diepen and Hans van der Marel
                - port to Python
    """
    
    # Convert input parameters to all numpy arrays
    
    if type(t) in [int,float]: 
        t = [t]
    t = np.array(t)
    if t.ndim == 0:
        t = np.array([t])
    xsate = np.array(xsate)
    vsate = np.array(vsate)

    # Check input arguments
    
    if xsate.shape[0] == 3 and t.shape[0] > 1:
        xsate = np.tile(xsate,[t.shape[0],1])

    if xsate.shape[0] != t.shape[0]:
        raise ValueError("Size of t does not match size of xsate")

    if xsate.shape[1] != 3:
        raise ValueError("xsate incorrect size")

    if vsate.size == 0:
        vsate = np.zeros(xsate.shape)

    if vsate.shape != xsate.shape:
        raise ValueError("vsate and xsate do not have the same shape")

    if vsate.shape[1] != 3:
        raise ValueError("vsate incorrect size")

    # Compute rotation angle (GMST) around z-axis

    gst0, omegae = ut2gmst(t[0])
    gst = gst0 + 2*np.pi*omegae*(t-t[0])
    gst = -1*gst

    # Rotate satellite positions around z-axis (ECEF -> ECI)

    xsat = np.zeros(xsate.shape)
    xsat[:, 0] = np.cos(gst)*xsate[:, 0] + np.sin(gst) * xsate[:, 1]
    xsat[:, 1] = -np.sin(gst) * xsate[:, 0] + np.cos(gst) * xsate[:, 1]
    xsat[:, 2] = xsate[:, 2]

    """
    To convert the velocity is more complicated. The velocity in ECEF
    consists of two parts. We find this by differentiating the transformation
    formula for the positions
 
       xsat = R * xsate

    This gives (product rule, and some rewriting), with |_dot| the derivatives

       xsat_dot = R * xsate_dot + R_dot * xsate    <=>
       vsat = R * ( vsate + inv(R)*R_dot * xsate ) <=>
       vsat = R * ( vsate + W * xsate )
 
    with |W = inv( R )*R_dot = [ 0 -w(3) w(2) ; w(3) 0 -w(1) ; -w(2) w(1) 0 ]| 
    and with |w| the angular velocity vector of the ECI frame with respect to
    the ECEF frame, expressed in the ECEF frame.
    """
 
    # The velocity vector in the ECI is computed as follows

    w = np.array([[0],
                  [0],
                  [-2*np.pi*omegae/86400]])
    W = np.array([[0, -w[2, 0], w[1, 0]],
                  [w[2, 0], 0, -w[0, 0]],
                  [-w[1, 0], w[0, 0], 0]])

    vsatw = vsate - np.matmul(xsate, W.T)

    vsat = np.zeros(vsate.shape)

    vsat[:, 0] = np.cos(gst)*vsatw[:, 0] + np.sin(gst) * vsatw[:, 1]
    vsat[:, 1] = -np.sin(gst) * vsatw[:, 0] + np.cos(gst) * vsatw[:, 1]
    vsat[:, 2] = vsatw[:, 2]

    return xsat, vsat


def eci2ecef(t, xsat, vsat=[]):
    """Convert position and velocity from ECI to ECEF reference frame.

    xsate, vsate = eci2ecef(t, xsat, vsat) converts cartesian coordinates xsat
    and velocities vsat in Earth Centered Inertial (ECI) reference frame,
    with t the time (UT1) given as Matlab datenumbers, into cartesian
    coordinates xsate and velocities vsate in Earth Centered Earth Fixed (ECEF)
    reference frame. t is a vector with length n, the number of epochs, and
    xsat and vsat are n-by-3 matrices.

    xsate, _ = eci2ecef(t, xsat) only transforms the positions.

    See also ut2gmst and ecef2eci.
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2016-2020.

    Created:  26 November 2016 by Hans van der Marel
    Modified: 18 November 2020 by Simon van Diepen and Hans van der Marel
                - port to Python
    """

    # Convert input parameters to all numpy arrays
    
    if type(t) in [int,float]: 
        t = [t]
    t = np.array(t)
    if t.ndim == 0:
        t = np.array([t])
    xsat = np.array(xsat)
    vsat = np.array(vsat)

    # Check input arguments

    if vsat.size == 0:
        vsat = np.zeros(xsat.shape) * np.nan

    if t.shape[0] != xsat.shape[0]:
        raise ValueError("Shape of xsat does not match T")

    if xsat.shape[1] != 3:
        raise ValueError("Incorrect xsat shape")

    if vsat.shape != xsat.shape:
        raise ValueError("xsat and vsat shape do not match")

    if vsat.shape[1] != 3:
        raise ValueError("vsat has incorrect shape")

    # Compute rotation angle (GMST) around Z-axis
   
    gst0, omegae = ut2gmst(t[0])
    gst = gst0 + 2*np.pi*omegae*(t-t[0])

    # Rotate satellite positions around z-axis (ECI -> ECEF)

    xsate = np.zeros(xsat.shape)
    xsate[:, 0] = np.cos(gst)*xsat[:, 0] + np.sin(gst) * xsat[:, 1]
    xsate[:, 1] = -np.sin(gst) * xsat[:, 0] + np.cos(gst) * xsat[:, 1]
    xsate[:, 2] = xsat[:, 2]

    """
    To convert the velocity is more complicated. The velocity in ECEF
    consists of two parts. We find this by differentiating the transformation
    formula for the positions

       xsate = R * xsat

    This gives (product rule, and some rewriting), with |_dot| the derivatives

       xsate_dot = R * xsat_dot + R_dot * xsat    <=>
       vsate = R * ( vsat + inv(R)*R_dot * xsat ) <=>
       vsate = R * ( vsat + W * xsat )

    with |W = inv( R )*R_dot = [ 0 -w(3) w(2) ; w(3) 0 -w(1) ; -w(2) w(1) 0 ]| 
    and with |w| the angular velocity vector of the ECEF frame with respect to
    the ECU frame, expressed in the ECI frame.
    """
    
    # The velocity vector in the ECEF is computed as follows

    w = np.array([[0],
                  [0],
                  [2*np.pi*omegae/86400]])
    W = np.array([[0, -w[2, 0], w[1, 0]],
                  [w[2, 0], 0, -w[0, 0]],
                  [-w[1, 0], w[0, 0], 0]])

    vsatw = vsat - np.matmul(xsat, W.T)

    vsate = np.zeros(vsat.shape)

    vsate[:, 0] = np.cos(gst)*vsatw[:, 0] + np.sin(gst) * vsatw[:, 1]
    vsate[:, 1] = -np.sin(gst) * vsatw[:, 0] + np.cos(gst) * vsatw[:, 1]
    vsate[:, 2] = vsatw[:, 2]

    return xsate, vsate
 
    
def satlookanglesp(t, xsat, xobj, verbose=0, swathdef=['VIS', 0, 80, '']):
    """Compute table with satellite look angles.
    
    lookangles, flags = satlookanglessp(t, xsat, xobj) computes look angles 
    from the object with coordinates xobj to satellite with coordinates
    xsat at time t, and vice versa, and returns them in the array lookangles.
    The input parameters t is n-by-1 vector with date numbers representing 
    time, xsat is a n-by-6 state-vector with the satellite position and 
    velocity at time T in a ECEF frame, and  xobj is either a vector of 
    length 3 with the position of the object, or a n-by-6 matrix with the 
    position and velocity of the object. xobj  and xsat must be in the same
    reference frame.
   
    The function returns two output parameters, lookangles and flags.
    The array lookangles has shape n by 8, with in its columns

      1  incidence angle at the object, which is identical to the zenith angle,
      2  azimuth angle from object to the satellite,
      3  off-nadir angle at the satellite to the direction of the object,
      4  azimuth angle at the satellite in the direction of the object,
      5  look angle in the direction of the object with respect to the flight 
         direction of the satellite, 
      6  azimuth angle of the flight direction of the satellite
      7  range between satellite and object
      8  range rate in the line of sight.

    The units are radians and meters. The lookangles are computed assuming 
    a spherical Earth.

    flags is  with in column 1 the ascending/descending flag ['ASC'|'DSC'], 
    in column 2 the left- or right-looking flag ['LL'|'RL'], and in column 
    3 the visibility flag ['VIS|''|<swath>], whereby the swath name can be 
    set in the SWATHDEF option.

    The function has two optional input parameters   

        verbose    0,1        Verbosity level [0]
        swathdef   list       Visibility or swath definition ['VIS'' 0 80 '']

    swathdef is a list with four items, a label, the incidence angle range, 
    and the look direction (right look'RL', left look 'LL' or both '')

        swathdef= ['VIS', 0.00, 80.00, '']       # Default with 10 deg elevation mask
 
        swathdef=[['IW1', 29.16, 36.59, 'RL' ],  # Example for SENTINEL-1, with
                   'IW2', 34.77, 41.85, 'RL' ],  # swath name, minimum and maximum
                   'IW3', 40.04, 46.00, 'RL' ]]  # incidence angle, only right looking

    Note that xsat and xobj must be in the same reference frame: so if xobj is a
    vector of length 3, in ECEF, also xsat must be in the ECEF reference frame.
    If xobj is a n-by-6 matrix then both ECI and ECEF are possible, but the
    reference frame for xsat must be the same as for xobj.

    See also prtlookangle, eci2ecef and ecef2eci.
    """
    
    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020

    Created:    26 July 2017 by Hans van der Marel for Matlab
    Modified:   27 Nov 2017 by Hans van der Marel
                   - version for spherical angles (SP)
                18 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """
    # Check input arguments

    if t.shape[0] != xsat.shape[0]:
        raise ValueError("Time argument t and satellite state vector xsat length mismatch")

    if xsat.shape[1] != 6:
        raise ValueError("Satellite state vector xsat does not have 6 columns")

    if xobj.shape[-1] not in [3, xsat.shape[-1]]:
        raise ValueError("xobj must be a vector of length 3 or have the same time length as xsat.")

    # Compute position vector from object to satellite, range and rangerate

    if max(xobj.shape) == 3:
        xobj2sat = np.zeros(xsat.shape)
        xobj2sat[:, :3] = xsat[:, :3] - np.tile(xobj,[t.shape[0],1])
        xobj2sat[:, 3:] = xsat[:, 3:]
    else:
        xobj2sat = xsat - xobj

    robj2sat = np.sqrt(np.sum(xobj2sat[:, :3]**2, axis=1))
    rrobj2sat = np.sum(xobj2sat[:, 3:] * xobj2sat[:, :3], axis=1) / robj2sat

    # Compute azimuth and zenith angle from object point of view

    if max(xobj.shape) == 3:
        xobj_ = np.zeros(xsat.shape)
        xobj_[:, :3] = xobj
        xobj = xobj_[:, :]
        

    robj = np.sqrt(np.sum(xobj[:, :3]**2, axis=1))  # range to object (observer)
    n0 = xobj[:, :3] / robj[:, np.newaxis]  # normal vector from object (observer)
    ers = xobj2sat[:, :3] / robj2sat[:, np.newaxis]  # init direction vector from observer to satellite

    ip = np.sum(n0 * ers, axis=1)
    z1 = np.arccos(ip)
    a1 = np.arctan2(-n0[:, 1] * ers[:, 0] + n0[:, 0] * ers[:, 1], -ip * n0[:, 2] + ers[:, 2])
    a1 += 2*np.pi
    a1 %= 2*np.pi

    # Compute azimuth and nadir angle from satellite point of view

    rsat = np.sqrt(np.sum(xsat[:, :3]**2, axis=1))
    n0sat = xsat[:, :3]/rsat[:, np.newaxis]

    ipsat = np.sum(n0sat*ers, axis=1)
    z2 = np.arccos(ipsat)
    a2 = np.arctan2(n0sat[:, 1] * ers[:, 0] - n0sat[:, 0] * ers[:, 1], ipsat*n0sat[:, 2] - ers[:, 2])
    a2 += 2*np.pi
    a2 %= 2*np.pi

    # compute the heading and look angle from the satellite point of view

    velsat = np.sqrt(np.sum(xsat[:, 3:]**2, axis=1))
    evsat = xsat[:, 3:]/velsat[:, np.newaxis]

    ipvel = np.sum(n0sat*evsat, axis=1)
    heading = np.arctan2(-n0sat[:, 1] * evsat[:, 0] + n0sat[:, 0] * evsat[:, 1], -ipvel*n0sat[:, 2] + evsat[:, 2])
    heading += 2*np.pi
    heading %= 2*np.pi

    lookangle = (4*np.pi + a2 - heading) % (2*np.pi)

    """
    Set ascending/descending and look direction flags

    ASC:  Heading -90 ...  90       RL:  lookangle   0 ... 180     
    DSC:  Heading  90 ... 270       LL:  lookangle 180 ... 360 
    """
    ASCDSC = np.array(['DSC', 'ASC'], dtype=object)
    ascdsc = ASCDSC[ np.array(np.floor((heading-np.pi/2)/np.pi) % 2, int) ]

    LOOKDIR = np.array(['RL', 'LL'], dtype=object)
    lookdir = LOOKDIR[ np.array(np.floor(lookangle/np.pi) % 2, int) ]
   
    # Convert swatchdef from list to 2D numpy array
    swathdef = np.array(swathdef)
    swathdef = swathdef.reshape(-1,swathdef.shape[-1])

    # Set swath definition flag
    IW = np.empty([z1.shape[0],], dtype="U12")
    for sw in swathdef:
        swbool = ( z1*180/np.pi >= np.float(sw[1]) ) & ( z1*180/np.pi <= np.float(sw[2]) ) 
        if sw[3] != '':
            swbool = swbool & lookdir == sw[3] 
        IW[swbool] = np.char.add(IW[swbool],'&' + sw[0])
    IW=np.char.strip(IW,'&')

    # Collect output
    
    lookangles = np.zeros((z1.shape[0], 8))
    lookangles[:, 0] = z1
    lookangles[:, 1] = a1
    lookangles[:, 2] = z2
    lookangles[:, 3] = a2
    lookangles[:, 4] = lookangle
    lookangles[:, 5] = heading
    lookangles[:, 6] = robj2sat
    lookangles[:, 7] = rrobj2sat

    flags = np.zeros((z1.shape[0], 3), dtype=object)
    flags[:, 0] = ascdsc
    flags[:, 1] = lookdir
    flags[:, 2] = IW

    if verbose:
        prtlookangle(t, lookangles, flags)

    return lookangles, flags


def prtlookangle(t, lookangles, flags, titlestr='', tableformat='default'):
    """Print a table with satellite look angles.
 
    prtlookangle(t,lookangles,flags) prints a table with the look angles
    from the n-by-8 matrix lookangles and n-by-3 cell-array FLAGS, which
    are output of the function SATLOOKANGLE.  t is n-by-1 vector with
    date numbers representing time.

    The array lookangles has shape n by 8, with in its columns

      1  incidence angle at the object, which is identical to the zenith angle,
      2  azimuth angle from object to the satellite,
      3  off-nadir angle at the satellite to the direction of the object,
      4  azimuth angle at the satellite in the direction of the object,
      5  look angle in the direction of the object with respect to the flight 
         direction of the satellite, 
      6  azimuth angle of the flight direction of the satellite
      7  range between satellite and object
      8  range rate in the line of sight.

    The units are radians and meters. The lookangles are computed assuming 
    a spherical Earth.

    flags is  with in column 1 the ascending/descending flag ['ASC'|'DSC'], 
    in column 2 the left- or right-looking flag ['LL'|'RL'], and in column 
    3 the visibility flag ['VIS|''|<swath>], whereby the swath name can be 
    set in the SWATHDEF option.

    prtlookangle has as optional input parameters

        title    string   Title string
        format   keyword  Print format ['default']

    See also satlookangle.
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020.

    Created:    26 July 2017 by Hans van der Marel for Matlab
    Modified:    1 Nov 2017 by Hans van der Marel
                   - split off from satlookangle 
                18 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """

    # Check input arguments

    if t.shape[0] != lookangles.shape[0] or lookangles.shape[0] != flags.shape[0]:
        raise ValueError("All input arrays must have the same number of rows")
    if lookangles.dtype not in [float, int, np.float, np.int]:
        raise ValueError("Lookangles must be a numeric array!")
    if lookangles.shape[1] != 8:
        raise ValueError("Lookangles must have 8 columns!")
    if type(flags) != np.ndarray:
        raise ValueError("Flags must be a numpy array!")
    if flags.shape[1] != 3:
        raise ValueError("Flags must have 3 columns!")

    # Print optional title

    print("\n{}{}\n".format(titlestr, "" if titlestr == "" else "\n"))
    
    # Print table
    
    if tableformat in ['default', 'ers']:
        print('                        Incidence Satellite  Off-Nadir LookAngle LookAngle')
        print('         Satellite Pass     Angle   Azimuth      Angle   Azimuth FlightDir   Heading     Range Rangerate'
              '   Flags')
        print('                            (deg)     (deg)      (deg)     (deg)     (deg)     (deg)      (km)    (km/s)'
              )
        for k in range(t.shape[0]):
            dt = num2datetime(t[k])
            isodate = dt.isoformat(timespec='milliseconds')
            print("{} {:>9.3f} {:>9.3f}  {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}   {} {} {}".format(
                isodate, lookangles[k, 0] * 180/np.pi, lookangles[k, 1] * 180 / np.pi, lookangles[k, 2] * 180 / np.pi,
                lookangles[k, 3] * 180 / np.pi, lookangles[k, 4] * 180 / np.pi, lookangles[k, 5] * 180 / np.pi,
                lookangles[k, 6] / 1000, lookangles[k, 7] / 1000, flags[k, 0], flags[k, 1], flags[k, 2]))
    else:
        raise ValueError("Unsupported table format!")
        
        