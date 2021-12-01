"""
----------------------------------------------------------------------------
tleplot.py    Compute satellite position and velocity from two-line elements.
Version 1.0 (12 November 2020).
Created by: Hans van der Marel and Simon van Diepen
Date:       12 Nov 2020
Modified:   -

Copyright: Hans van der Marel, Simon van Diepen, Delft University of Technology, 2020
Email:     h.vandermarel@tudelft.nl
Github:    -
----------------------------------------------------------------------------
Functions:

Get and read NORAD Two Line Elements in TLE structure array
  tleget      - Retrieve NORAD Two Line Elements from www.celestrak.com
  tleread     - Read NORAD Two Line Elements from file.

Plotting of satellite positions, elevation, azimuth, visibility, ...
  tleplot1    - Several plots of satellite position and velocity from NORAD Two Line Elements
  tle2kml1    - Google Earth KML file with satellite position and optional ground station links
  skyplot     - Polar plot with azimuth and elevation
  pltgroundtrack - Plot satellite ground track 

Find satellites and select dates 
  tlefind     - Find named satellites in the NORAD Two Line Elements.
  tledatenum  - Compute Matlab datenumbers from a date range.

Compute satellite positions and orbit propagation
  tle2vec     - Satellite position and velocity from NORAD Two Line Elements.
  tle2orb     - Compute orbital elements from NORAD Two Line Elements

Compute and print satellite look angles and times
  satlookanglesp - Create a table with satellite look angles
  prtlookangle   - Print a table with satellite look angles


Examples:
  tle=tleread('resource-10-oct-2017.tle')
  tlefind(tle,'SENTINEL')
  tleplot1(tle,{'2017-10-10 0:00', 24*60 ,1},'SENTINEL-1A',[ 52 4.8  0 ])

(c) Hans van der Marel, Delft University of Technology, 2012-2020.
"""

"""
This toolbox imports the following functions from the crsutil.py

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
"""

import os
import numpy as np

from collections import namedtuple
import urllib.request as url
from datetime import datetime
from dateutil.parser import parse as parsedate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
from scipy.io import loadmat
import ssl
from crsutil import keplerm, ut2gmst, orb2vec, datetime2num, num2datetime, pol2cart


ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def tleget(tleset, tlefile=""):
    """Retrieve NORAD Two Line Elements from www.celestrak.com.

    tlefile = tleget(tleset) download current NORAD two line elements 
    from Celestrak (www.celestrack.com) and saves the two line elements to
    the file tlefile. tleset is the name of the two-line element
    set, or a list with the names of two-line element sets. This can be 
    the name of a file on Celestrak, or, a family of satellites such as "GNSS". 
    The function returns tlefile with the name of the file with two-line 
    elements on the local disk,

    tlefile = tleget(tleset, tlefile=...) saves the download NORAD two 
    line elements to a file with the name given with the optional tlefile
    argument.
    
    The mean orbital elements contained in TLE are Earth-centered-inertial (ECI)
    coordinates with respect to the true equator of date and the mean equinox
    of date. They do not include the effect of nutation. The TLE are 
    compatible with the SGP4 and SDP4 orbit propagators of NORAD. The 
    two line elements are read with the tleread() function.
    
    Examples:
       tlefile = tleget('gnss')
       tlefile = tleget('gps','gps-20131102.tle')
       tleGPS = tleread(tlefile)

    Common sets of two line elements are 'GPS' ('gps-ops'), 'GLONASS'
    ('glo-ops'), 'GALILEO', 'BEIDOU', 'SBAS'; 'GNSS' or 'SATNAV' to do
    all satellite navigation systems; 'resource' for Earth resource
    satellites, etc. For a full list see the Celestrack website.
    
    See also tleread, tle2orb and tle2vec.
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2013-2020.

    Created:     2 November 2013 by Hans van der Marel for Matlab
    Modified:    3 August 2015 by Hans van der Marel
                12 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """
    
    if type(tleset) not in [np.ndarray, list, str]:
        raise ValueError("tleset is not a list, string or array!")

    celestrakurl = 'http://celestrak.com/NORAD/elements/'
    if tlefile == "":
        print("Saving TLE set to default name")
        if type(tleset) == str:
            tlefile = tleset + ".txt"
        else:
            raise ValueError("tlefile must be defined when using multiple tleset!")

    if type(tleset) == str:
        tleset = [tleset]
    elif type(tleset) == np.ndarray:
        tleset = list(tleset)

    tleset2 = []
    for k in range(len(tleset)):
        tlesetk = tleset[k].lower()
        if tlesetk in ['gnss', 'satnav']:
            tleset2.append('gps-ops')
            tleset2.append('glo-ops')
            tleset2.append('galileo')
            tleset2.append('beidou')
            tleset2.append('sbas')
        elif tlesetk == 'gps':
            tleset2.append('gps-ops')
        elif tlesetk in ['glonass', 'glo']:
            tleset2.append('glo-ops')
        else:
            tleset2.append(tlesetk)

        s = ''
        fname = []
        for tlelink in tleset2:
            fullurl = celestrakurl + tlelink + '.txt'
            try:
                sk = str(url.urlopen(fullurl, context=ctx).read())
            except IOError:
                print("TLEGET: Could not retrieve {}.txt from {}".format(tlelink, celestrakurl))
                continue
            s += sk
            fname.append(tlelink)
            print("TLEGET: Downloaded {}.txt from {}".format(tlelink, celestrakurl))

        f = open(tlefile, "w")
        if "\\r" in s:
            s = s.replace("\\r", "")
        if "\\n" in s:
            s = s.replace("\\n", "\n")
		
        f.write(str(s)[2:-1])
        f.close()

        print("Saved TLE to {}".format(tlefile))

        return tlefile

def tleread(fname, verbose=1):
    """Read NORAD Two Line Elements from file.

    tle = tleread(fname) reads NORAD two line elements from file with name 
    fname and returns the two-line elements (TLE's) in a list tle of named tuples.

    The mean orbital elements contained in tle are Earth-centered-inertial (ECI)
    coordinates with respect to the true equator of date and the mean equinox
    of date. They do not include the effect of nutation. The tle are 
    compatible with the SGP4 and SDP4 orbit propagators of NORAD.

    Files with TLE's can be obtained from www.celestrak.com. You may use
    the function tleget to do this. The TLE files can have an optional line 
    with a twenty-four character name before the traditional Two Line Element 
    format (Three-Line Elemement Set).
    
    The function takes an optional argument verbose, if verbose=1 (default) 
    an overview of the TLE's is printed, if verbose=0 the function is quiet.
    
    Example:
      tle=tleread('gps-ops.tle')

    See also tleget, tlefind, tle2orb and tle2vec.
    """
    
    """
    (c) Hans van der Marel, Delft Universtiy of Technology, 2012-2013.

    Created:    30 Dec 2012 by Hans van der Marel for Matlab
    Modified:   12 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """

    # Constants (WGS-84)
    mu = 398600.5  # km^3/s^2
    Re = 6378.137  # km (WGS84 earth radius)
    d2r = np.pi / 180
    
    
    # Define internal function to compute eccentric anomaly
    def eanomaly(m0, ecc0, T0L=1e-10):
        E = m0
        f = [1, 1]
        fdot = 1
        while abs(np.array([f])).max() > T0L:

            f = m0 - E + ecc0 * np.sin(E)
            fdot = -1 + ecc0 * np.cos(E)
            E -= f / fdot

        return E


    # Define named tuple TLE (similar to Matlab stuct)
    tlestruct = namedtuple("TLE",
                           "name satid ephtype year epoch t0 ecc0 inc0 raan0 argp0 m0 n0 ndot nddot bstar revnum a0 e0")

    # Open the file with TLEs'
    if not os.path.exists(fname):
        raise ValueError("Filename {} with TLE elements not found!".format(fname))

    f = open(fname, "r")
    lines = f.read().split("\n")
    f.close()
    
    if verbose:
        print('\nSatellite              Reference_Epoch    a [km]    ecc [-]  inc [deg] RAAN [deg] argp [deg]    '
              'E [deg]    Period\n')

    # Read TLE elements

    """
    Data for each satellite consists of three lines in the following format:

             1         2         3         4         5         6         7
    1234567890123456789012345678901234567890123456789012345678901234567890

    AAAAAAAAAAAAAAAAAAAAAAAA
    1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
    2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN

    Line 0 is a twenty-four character name (to be consistent with the name 
    length in the NORAD SATCAT). Line 0 is optional and may be preceeded with
    a zero.  

    Lines 1 and 2 are the standard Two-Line Orbital Element Set Format 
    identical to that used by NORAD and NASA. The format description is:

    Line 1

    Column 	Description
    01      Line Number of Element Data
    03-07 	Satellite Number
    08      Classification (U=Unclassified)
    10-11 	International Designator (Last two digits of launch year)
    12-14 	International Designator (Launch number of the year)
    15-17 	International Designator (Piece of the launch)
    19-20 	Epoch Year (Last two digits of year)
    21-32 	Epoch (Day of the year and fractional portion of the day)
    34-43 	First Time Derivative of the Mean Motion
    45-52 	Second Time Derivative of Mean Motion (decimal point assumed)
    54-61 	BSTAR drag term (decimal point assumed)
    63      Ephemeris type
    65-68 	Element number
    69      Checksum (Modulo 10)

    Line 2

    Column 	Description
    01      Line Number of Element Data
    03-07   Satellite Number
    09-16 	Inclination [Degrees]
    18-25 	Right Ascension of the Ascending Node [Degrees]
    27-33 	Eccentricity (decimal point assumed)
    35-42 	Argument of Perigee [Degrees]
    44-51 	Mean Anomaly [Degrees]
    53-63 	Mean Motion [Revs per day]
    64-68 	Revolution number at epoch [Revs]
    69      Checksum (Modulo 10)
 
    All other columns are blank or fixed.

    The mean orbital elements in TLE are Earth-centered-inertial (ECI)
    coordinates with respect to the true equator of date and the mean equinox 
    of date. They do not include the effect of nutation.
    """
   
    tlelist = []
    ntle = 0
    line0 = ""
    line1 = ""
    line2 = ""
    for line in lines:
        # Skip blank lines
        if line == "":
            continue
        # Decode the line with the satellite name (optional) and read next two lines
        if line[0:2] == "1 ":     # Line 1
            if line1 != "":
                raise ValueError("Invalid line format, received 2 line1: {}, {}.".format(line1, line))
            line1 = line
        elif line[0:2] == "2 ":   # Line 2
            if line1 == "":
                raise ValueError("Received line2 when line1 not yet read! {}".format(line))
            if line2 != "":
                raise ValueError("Invalid line format, received 2 line2: {}, {}.".format(line2, line))
            line2 = line
        else:
            if line0 != "":
                raise ValueError("Invalid line format, received 2 line0: {}, {}.".format(line0, line))
            line0 = line

        if line2 != "":
            # Now we have the second line we can start the decoding
            
            # Decode the zero line (if available)
            if line0 != "":
                if line0[:2] == "0 ":
                    satname = line0[2:]
                else:
                    satname = line0
            else:
                satname = "Undefined"

            # Decode the first line with TLE data

            satid = eval(line1[2:7].lstrip("0"))
            classification = line1[7]
            intldesg = line1[9:17]
            epochyr = eval(line1[18:20].lstrip("0"))
            if epochyr < 57:
                epochyr += 2000
            else:
                epochyr += 1900

            epochdays = eval(line1[20:32].lstrip("0"))
            ndot = eval(line1[33:43].lstrip("0"))
            nddot = eval(line1[44:50].lstrip("0"))
            nexp = eval(line1[50:52].lstrip("0"))
            nddot *= 1e-5 * 10**nexp
            bstar = eval(line1[53:59].lstrip("0"))
            ibexp = eval(line1[59:61].lstrip("0"))
            bstar *= 1e-5 * 10**ibexp
            ephtype = line1[62]
            elnum = eval(line1[64:68].lstrip("0"))

            # Decode the second line with TLE data

            if eval(line2[2:7].lstrip("0")) != satid:
                raise ValueError("satid from line 1 ({}) does not match satid from line 2 ({})!".format(satid,
                                                                                                        eval(line2[2:7])
                                                                                                        ))
            inc0 = eval(line2[7:16].lstrip("0"))
            raan0 = eval(line2[16:25].lstrip("0"))
            ecc0 = eval(line2[26:33].lstrip("0")) * 1e-7
            argp0 = eval(line2[33:42].lstrip("0"))
            m0 = eval(line2[42:51].lstrip("0"))
            n0 = eval(line2[51:63].lstrip("0"))
            revnum = eval(line2[63:68].lstrip("0"))

            # Complete orbital elements
            t0 = datetime2num(datetime(year=int(epochyr), month=1, day=1)) + epochdays - 1
            a0 = (mu/(n0*2*np.pi/(24*3600))**2)**(1/3)
            e0 = eanomaly(m0, ecc0)
            OE = [a0, ecc0, inc0, raan0, argp0, e0]

            # Print some data from the TLE's
            if verbose:
                ihour = np.floor(24/n0)
                imin = np.floor(24*60/n0 - ihour*60)
                isec = round(24*3600/n0-ihour*3600-imin*60)
                if isec == 60:
                    isec = 0
                    imin += 1
                    if imin == 60:
                        imin = 0
                        ihour += 1
                tt = '{:0>2.0f}:{:0>2.0f}:{:0>2.0f}'.format(ihour, imin, isec)
                print("{:<24s}{:4d}-{:>3.5f}".format(satname, epochyr, epochdays) +
                      "  {:>8.2f}  {:>9.7f}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {}".format(*OE, tt))

            # Fill output structure           
            ntle += 1
            tlelist.append(tlestruct(satname.strip(" "), (satid, classification, intldesg.strip(" ")), (ephtype, elnum),
                                     epochyr, epochdays, t0, ecc0, inc0 * d2r, raan0 * d2r, argp0 * d2r, m0*d2r,
                                     n0 * 2 * np.pi, ndot * 2 * np.pi, nddot * 2 * np.pi, bstar, revnum, a0 * 1000,
                                     e0 * d2r))
            # Clear lines
            line0 = ""
            line1 = ""
            line2 = ""

    return tlelist


def tlefind(tle, satid, verbose=1):
    """Find named satellites in the NORAD Two Line Elements.

    isat, satids = tlefind(tle,satid) finds the satellite(s) with name satid
    in the list of named tuples tle with NORAD Two Line Elelements, and returns
    in the element numbers in the list of named tuples tle. The element
    numbers and names of the selected satellites are returned in isat and 
    isatids. tle is a list of named tuples that was read by tleread. satid
    is either a string, a list or numpy array of strings or numeric.

    Examples:

      tleERS=tleread('resource.txt')       # read two-line elements
      isat, satids = tlefind(tleERS, 'RADAR')
      isat, *_ = tlefind(tleERS, 'RADARSAT-2')
      isat, satids = tlefind(tleERS, [ 'RADARSAT-2', 'SENTINEL-1', 'TERRASAR' ])
      isat, satids = tlefind(tleERS, isat)

    See also tleget, tleread, tle2orb and tle2vec.
    """
    
    """
    (c) Hans van der Marel, Delft Universtiy of Technology, 2012-2020  
    
    Created:    30 Dec 2012 by Hans van der Marel for Matlab
    Modified:    3 August 2015 by Hans van der Marel for Matlab
                12 November 2020 by Hans van der Marel and Simon van Diepen
                   - port to Python
    """

    # Check the format of satid
    if type(satid) in [str, int, np.int64]:
        satids = [satid]
    elif type(satid) in [np.ndarray, list]:
        satids = list(satid)
    else:
        raise ValueError("satid is not a string, int or list of both!")

    # Extract satellite names from tle
    satnames = np.array(list(map(lambda s: s.name,tle)))

    # Loop through the satids and find the satellite indices
    isat = np.empty((0,1),int)
    for i in range(len(satids)):

        if isinstance(satids[i],str):
           if "GLONASS" in satids[i].upper():
              satids[i] = 'COSMOS'
           elif "GALILEO" in satids[i].upper():
              satids[i] = 'GSAT'

           l=list(map(lambda s: satids[i] in s,satnames))
           isati = np.argwhere(l)

           if np.size(isati) == 0:
              raise ValueError("Satellite {} not in tle!".format(satids[i]))

           isat = np.concatenate(([isat, isati]))

        else:
           isat = np.append(isat, satids[i])
                  
    isat = np.unique(isat)
    satids = satnames[isat]

    if verbose > 0:
       print("Found {} satellites:".format(len(satids)))
       for i in range(len(satids)):
          print(" {}  ({})".format(tle[isat[i]].name, isat[i]))

    return isat, satids


def tledatenum(daterange):
    """Computes Matlab datenumbers from a date range. 
    
    t = tledatenum(daterange) computes the Matlab datenumbers from the
    input parameter daterange and returns the Matlab datenumbers in t.

    The input parameter daterange is either 
    - a three element list with start date, end date (or duration in minutes) 
      and data interval in minutes, 
    - a string with the date or a list with date strings, or,
    - a numpy array with Matlab datenumbers. 
    
    Examples:
       t=tledatenum(['2013-9-13 0:00:00', 24*60 ,1])   -> returns array with 
            datenum values from the first date for a time range of entry 2 in 
            minutes, with  a time step of entry 3 
            
       t=tledatenum(['2013-9-13 0:00:00', '2013-9-14 0:00:00', 1]) -> returns 
            array with datenum values from the first date to the second date, 
            with an interval in minutes of entry 3 (1 in this case)
       t=tledatenum('2013-9-13')  -> returns datenum for this date
       t=tledatenum('2013-9-13 0:00:00') -> returns datenum for this string
       t=tledatenum(np.array(['2013-9-13 0:00:00', '2013-9-14'])) -> returns 
           numpy array of datenum of these strings
       t=tledatenum(['2013-9-13 0:00:00', '2013-9-14']) -> returns array of 
           datenum of these strings
       t=tledatenum(t) -> returns array of datenums in t

    See also tle2orb, tle2vec and tleplot.
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2012-2020

    Created:    30 Dec 2012 by Hans van der Marel
    Modified:   12 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """

    if type(daterange) in [np.ndarray, list]:

        if len(list(daterange)) == 3 and type(daterange[0]) == str:
            if type(daterange[2]) != str:
                t0 = datetime2num(parsedate(daterange[0]))
                if type(daterange[1]) == str:
                    t1 = datetime2num(parsedate(daterange[1]))
                    t = np.arange(t0, t1 + daterange[2]/(24*60), daterange[2]/(24*60))
                else:
                    t = np.arange(t0, t0 + daterange[1]/(24*60) + daterange[2]/(24*60), daterange[2]/(24*60))
            else:
                t = np.array([datetime2num(parsedate(datestamp)) for datestamp in daterange])
        elif type(daterange[0]) == str:
            t = np.array([datetime2num(parsedate(datestamp)) if type(datestamp) == str else datestamp
                                 for datestamp in daterange])
        else:
            t = daterange
            
    elif type(daterange) == str:
        t = np.array([ datetime2num(parsedate(daterange)) ])
    elif type(daterange) in [int, float, np.float, np.int]:
        t = np.array([daterange])
    else:
        raise ValueError("Unknown input type {}".format(type(daterange)))

    return t


def tle2orb(tle, t, propagation="J2"):
    """Compute satellite orbital elements from NORAD Two Line Elements.

    orb = tle2orb(tle, t) computes the satellite orbital elements from NORAD 
    Two Line Elements from NORAD two line elements in the named tuple tle at
    times t, and returns the orbital elements at times t on orb. t is an
    numpy 1D array with date numbers in UT1. The output orb is an array with 
    a row for each element of t, and on row k the six orbital elements 
    corresponding to the time t[k]: semi-major axis [m], eccentricity [-], 
    inclination [rad], right ascension of ascending node [rad], argument of 
    periapsis [rad], and true anomaly [rad]. 
    
    The optional input parameter propagation is the propagation method to 
    compute orb. Valid propagation methods are:

       J2    Include secular effects of J2, but nothing else. (default) This gives
             acceptable results for plotting purposes and planning computations

       NOJ2  Ignores effect of J2 on orbit propagation. Should only be used for 
             educational purposes

       SGP4  SGP4 orbit propagator of NORAD. Not implemented yet
    
    See also tleget, tleread, tlefind and tle2vec.
    """
    
    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020

    Created:    13 September 2017 by Hans van der Marel for Matlab
    Modified:   12 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """
    
    # Define constants
    J2 = 0.00108262998905
    Re = 6378136  # m, radius of Earth
    mu = 3986004418e5  # m^2/s^2 , gravitational constant of Earth

    # Prepare the array with times
    t = tledatenum(t)
    n_epoch = t.shape[0]
    t0 = datetime2num(datetime(year=tle.year, month=1, day=1)) + tle.epoch - 1

    # Orbit propagation
    if propagation.upper() == "J2":
        
       # Compute rate of change of orbital elements
       #
       #   draan/dt = s.*cos(inclination)
       #   dargp/dt = -0.5*s.*(5*cos(inclination-1).^2)
       #   dM/dt = -0.5*s.*sqrt(1-e.^2).*(3*cos(inclination).^2 -1)
       #
       # with s=-J2*3/2*sqrt(mu/a^3)*(Re/p)^2
       #
       # dM/dt is not needed for two line element propagation, but computed nevertheless. 

        p = tle.a0*(1-tle.ecc0**2)
        s = -J2 * 3 / 2 * np.sqrt(mu / tle.a0**3) * (Re / p)**2
        odot = s * np.cos(tle.inc0) * 86400
        wdot = -0.5 * s * (5*np.cos(tle.inc0)**2 - 1) * 86400
        # mdot = -0.5 * s * np.sqrt(1-tle.ecc0**2) * (3*np.cos(tle.inc0)**2 - 1) * 86400

        raan = tle.raan0 + odot * (t-t0)
        argp = tle.argp0 + wdot * (t-t0)

        m = tle.m0 + tle.n0 * (t-t0)
        ignore, nu = keplerm(m + argp, tle.ecc0)
        nu -= argp

        orb = np.hstack([np.array([tle.a0, tle.ecc0, tle.inc0] * n_epoch).reshape((n_epoch, 3)),
                         raan.T[:, np.newaxis], argp.T[:, np.newaxis], nu.T[:, np.newaxis]])

    elif propagation.upper() == "NOJ2":
        
        # Very simple orbit propagation ignoring effect of J2 (use with
        # extreme caution, only for educational purposes)

        m = tle.m0 + tle.n0 * (t - t0)
        ignore, nu = keplerm(m, tle.ecc0)
        orb = np.hstack([np.array([tle.a0, tle.ecc0, tle.inc0, tle.raan0, tle.argp0] * n_epoch).reshape((n_epoch, 5)),
                         nu.T[:, np.newaxis]])

    elif propagation.upper() == "SGP4":
        raise ValueError("SGP4 propagation method not implemented yet!")
    else:
        raise ValueError("Unknown propagation method {}! Please use J2, NOJ2, (or SGP4)!".format(propagation))

    return orb


def tle2vec(tle, t, satid, propagation="J2", verbose=0):
    """Compute satellite position and velocity in ECI from NORAD Two Line Elements.
    
    xsat, vsat = tle2vec(tle,t,satid) computes the satellite position and 
    velocity in ECI coordinates from NORAD two line elements, for satellite 
    satid at times t. tle is a list with named tuples that is read by tleread,
    t must be a 1D numpy array with date numbers in UT1, a character string
    with a data or a list with a date range. For more details on t see
    tledatenum. The function returns the satellite position and velocity
    in numpy arrays xsat and vsat.

    The shape of xsat and vsat is (n, 3), with n the length of t. The first 
    axis is time t, the second axis are the coordinates X, Y and Z, 
    respectively velocities VX, VY and VA, in ECI reference frame.

    ... = tle2vec(...,propagation='...') selects an alternative propagation
    method. The default propagation method is J2 which includes the secular 
    effects of J2. This gives acceptable results for plotting purposes and 
    planning computations. Formally, TLE are only compatible with the SGP4 
    orbit propagator of NORAD, but this is not (yet) supported by this 
    function. The other available option is NOJ2, which ignores the effect 
    of J2 on the orbit propagation and should only be used for educational 
    purposes.

    Examples:

       tleERS=tleread('resource.txt')     # read two-line elements
       xsat, vsat = tle2vec(tleERS,'2013-9-13 0:00','RADARSAT-2')

       t=tledatenum('2013-9-13 0:00', 24*60, 1)    # array with date numbers ...
       xsat, vsat = tle2vec(tle,t,'RADARSAT-2')

       xsat, vsat = tle2vec(tle,['2013-9-13 0:00', 24*60, 1],'RADARSAT-2')

    See also tleget, tleread, tle2orb and orb2vec.
    """
    
    """
    (c) Hans van der Marel, Delft Universtiy of Technology, 2012-2020

    Created:    30 Dec 2012 by Hans van der Marel for Matlab
    Modified:   12 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
    """
   
    # Prepare the dates
    t = tledatenum(t)
    
    # Find the satellite and compute the position and velocity in ECI frame
    isat, satids = tlefind(tle, satid, verbose)
    if len(isat) == 0:
        # No satellite found, return empty without complaining
        xsat = []
        vsat = []
    elif len(isat) > 1:
        # More than one satellite found, raise error
        raise ValueError("Satellite {} not unique in provided TLE!".format(satid))
    else:
        # Compute position and velocity in ECI
        orb = tle2orb(tle[isat[0]], t, propagation)
        vec = orb2vec(orb)

        xsat = vec[:,:3]
        vsat = vec[:,3:]

        #print('vec',vec.shape)
        #print('orb',orb.shape)
        #print('xsat',xsat.shape)
        #print('vsat',vsat.shape)

    return xsat, vsat


def tleplot1(tle, daterange, satid, objcrd=None, figsize=(10,6)):
    """Plot satellite position and velocity from NORAD Two Line Elements.

    tleplot1(tle,daterange,satid) plots the satellite position and 
    velocities in various ways. tle is a list with named tuples containing
    two line elements read by tleread. daterange is a list with the start date, 
    end date (or duration) and data interval. The duration and data interval 
    are in minutes. satid is a character string with the name of the satellite 
    to plot. 

    tleplot1(tle,daterange,satid,objcrd) add's the position of an observer 
    with coordinates objcrd to the plots. objcrd are the geographical 
    coordinates with latitude and longitude (in degrees), and height (in meters) 
    of the observer or object on Earth.
 
    Example:
       tleERS=tleread('resource.txt')
       tleplot1(tleERS,{'2013-9-13 0:00', 24*60 ,1},'RADARSAT-2')
       tleplot1(tleERS,{'2013-9-13 0:00', 24*60 ,1},'RADARSAT-2',[ 52 4.8  0 ])
 
    Files with TLE's can be obtained from www.celestrak.com
 
    See also tleget, tleread, tlefind, tledatenum, tle2vec, keplerm and orb2vec.
    """
    
    """
    (c) Hans van der Marel, Delft University of Technology, 2012-2020

    Created:    30 Dec 2012 by Hans van der Marel for Matlab
    Modified:   12 November 2020 by Simon van Diepen and Hans van der Marel
                   - port to Python
                15 November 2021 by Hans van der Marel
                   - plots wthout object coordinates possible
    """
    
    # Constants

    Re = 6378136
    # mu = 3986004418e5
    Me = 7.2921151467e-5  # rad/s , rotational velocity of Ewarth
    # c = 299792458

    # Compute the epoch times t from the list with start, end and data interval

    t = tledatenum(daterange)
    nepoch = t.shape[0]
    
    # Prepare x-axis date ticks

    kmid=int(t.shape[0]/2)
    if ( t[-1] - t[0] ) < 1/120:
       xtimefmt = '%H:%M:%S'
       xlabelfmt = '%Y-%m-%d'
    elif ( t[-1] - t[0] ) < 3: 
       xtimefmt = '%H:%M'
       xlabelfmt = '%Y-%m-%d'
    elif ( t[-1] - t[0] ) < 6: 
       xtimefmt = '%m-%d %Hh'
       xlabelfmt = '%Y-%m-%d'
    else: 
       xtimefmt = '%m-%d'
       xlabelfmt = '%Y'

    xlabeldate = num2datetime(t[kmid]).strftime(xlabelfmt)
	
	# ----------------------------------------------------------------------------
    # Compute and plot satellite position and velocity in ECI
    # ----------------------------------------------------------------------------

    # Compute satellite state vectors (position and velocity in ECI) using tle2vec.m

    xsat, vsat = tle2vec(tle, t, satid)
    if not xsat.all():
        raise ValueError("xsat is empty!")

    rsat = np.sqrt(np.sum(xsat**2, axis=1))
    velsat = np.sqrt(np.sum(vsat**2, axis=1))

    plt.figure("ECI Position and Velocity",figsize=figsize)
    #plt.subplots(2,2,sharex=True, tight_layout=True)
    #plt.suptitle("{} position and velocity in ECI ({})".format(satid, xlabeldate))
    
    plt.subplot(211)
    #plt.figure("Position",figsize=figsize)
    plt.plot(t, np.array(xsat[:, 0])/1000, linewidth=2, label="X")
    plt.plot(t, np.array(xsat[:, 1])/1000, linewidth=2, label="Y")
    plt.plot(t, np.array(xsat[:, 2])/1000, linewidth=2, label="Z")
    plt.plot(t, rsat/1000, linewidth=2, color='k', label="r")
    plt.title("{} position and velocity in ECI ({})".format(satid, xlabeldate))
    plt.ylabel("Position [km]")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    #plt.xlabel("Date {}".format(xlabeldate))
    plt.legend()

    plt.subplot(212)
    #plt.figure("Velocity",figsize=figsize)
    plt.plot(t, np.array(vsat[:, 0])/1000, linewidth=2, label="V_X")
    plt.plot(t, np.array(vsat[:, 1])/1000, linewidth=2, label="V_Y")
    plt.plot(t, np.array(vsat[:, 2])/1000, linewidth=2, label="V_Z")
    plt.plot(t, velsat/1000, linewidth=2, color='k', label="v")
    plt.ylabel("Velocity [km/s]")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xlabel("Date {}".format(xlabeldate))
    plt.legend()

    # ----------------------------------------------------------------------------
    # Compute position and velocity of the observer in ECI
    # ----------------------------------------------------------------------------

    if objcrd != None:
        
        # The position of the observer (latitude, longitude and height) is given in ECEF
        # using the input array objcrd

        lat = objcrd[0]*np.pi/180
        lon = objcrd[1]*np.pi/180
        Rs = Re + objcrd[2]
	
        # Position of the observer in ECEF (assume latitude and longitude are for spherical Earth)

        xobjECEF = [Rs*np.cos(lat)*np.cos(lon),
                    Rs*np.cos(lat)*np.sin(lon),
                    Rs*np.sin(lat)]
        vobjECEF = [0, 0, 0]
	
   	    # The transformation from an ECEF to ECI is a simple rotation around the z-axis
        # a. the rotation angle is GMST (Greenwhich Mean Stellar Time) 
        # b. the rotation around the z-axis can be implemented by replacing the
        #    longitude (in ECEF) by local stellar time (lst) in the ECI
        #
        # The times given in t are in UTC, which is close to UT1 (max 0.9 s difference),
        # which is not important for a plotting application

        # Compute GMST from UT1, for the first epoch in t, using the Matlab function 
        # ut2gmst. The second output returned by ut2gmst is the rotational velocity
        # omegae of the Earth in rev/day

        gst0, omegae = ut2gmst(t[0])

        # Compute local stellar time (in radians) from the longitude, GMST at the initial
        # epoch and the rotational velocity of the Earth (times elapsed time). Note that
        # lst is an array, while lon is a scalar)

        lst = lon+gst0+2*np.pi*omegae*(t-t[0])

        # Compute position and velocity of the observer in ECI using lst (position and 
        # velocity in an ECI change all the time, unlike in a ECEF)

        xobj = np.zeros((nepoch, 3))   # pre-allocate memory, makes it run faster
        vobj = np.zeros((nepoch, 3))

        xobj[:, 0] = Rs*np.cos(lat)*np.cos(lst)
        xobj[:, 1] = Rs*np.cos(lat)*np.sin(lst)
        xobj[:, 2] = Rs*np.sin(lat)
        vobj[:, 0] = -Rs*np.cos(lat)*Me*np.sin(lst)
        vobj[:, 1] = Rs*np.cos(lat)*Me*np.cos(lst)
    
	
    	    # Plot observer position (Not really interesting, but easy to do, use previous
        # plots as example)


    	    # ----------------------------------------------------------------------------
        # Satellite position and velocity from the observer (object) point of view
        # ----------------------------------------------------------------------------
    
        # position and velocity vectors, range and range rates, from object to
        # satellites in ECI
    
        xobj2sat = xsat-xobj
        vobj2sat = vsat-vobj
        robj2sat = np.sqrt(np.sum(xobj2sat**2, axis=1))
        rrobj2sat = np.sum(vobj2sat * xobj2sat/robj2sat[:, np.newaxis], axis=1)
    
        # Note that the range rate rrobj2sat is not the same as the relative velocity
        # sqrt(sum(vobj2sat.^2,2)), these are different things
    
        # normal vector (vertical) and unit direction vector to satellite from observer
    
        robj = np.sqrt(np.sum(xobj**2, axis=1))
        n0 = xobj / robj[:, np.newaxis]
        ers = xobj2sat / robj2sat[:, np.newaxis]
    	
        # zenith angle and azimuth of satellite (as seen from object wrt to radial direction)
    
        ip = np.sum(n0 * ers, axis=1)
        zenith = np.arccos(ip)
        azi = np.arctan2(-n0[:, 1]*ers[:, 0] + n0[:, 0]*ers[:, 1], ip*-1*n0[:, 2] + ers[:, 2])
        azi += 2*np.pi
        azi %= 2*np.pi
    
        # Elevation angle and satellite visibility (if elevation angle > 0))
    
        elevation = np.pi/2 - zenith
    
        cutoff = 0                       # cutoff angle in degrees
        visible = elevation > np.radians(cutoff)
    	
        # Plot elevation, azimuth, range and range rate
    
        plt.figure("Viewing angles",figsize=figsize, tight_layout=True)
        #plt.subplots(2,2,sharex=True, tight_layout=True)
        plt.suptitle("{} ({})".format(satid, xlabeldate))
        
        plt.subplot(221)
        plt.plot(t, elevation * 180/np.pi, linestyle='--', color='k', label='Elevation')
        plt.scatter(t[visible], elevation[visible] * 180/np.pi, color='g', s=15, label='Visible')
        plt.title('Elevation angle [deg]')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.legend()
    
        plt.subplot(223)
        plt.plot(t, azi * 180/np.pi, linestyle='--', color='k', label='Azimuth')
        plt.scatter(t[visible], azi[visible] * 180/np.pi, color='g', s=15, label='Visible')
        plt.title('Azimuth angle [deg]')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.legend()
    
        plt.subplot(222)
        plt.plot(t, robj2sat / 1000, linestyle='--', color='k', label='Range')
        plt.scatter(t[visible], robj2sat[visible] / 1000, color='g', s=15, label='Visible')
        plt.title('Range [km]')
        plt.gca().yaxis.tick_right()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.legend()
    
        plt.subplot(224)
        plt.plot(t, rrobj2sat / 1000, linestyle='--', color='k', label='Range rate')
        plt.scatter(t[visible], rrobj2sat[visible] / 1000, color='g', s=15, label='Visible')
        plt.title('Range rate [km/s]')
        plt.gca().yaxis.tick_right()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.legend()
    
        # Make a skyplot (polarplot of azimuth and zenith angle)
    
        plt.figure("Skyplot", figsize=figsize)
        skyplot(t, azi, zenith, cutoff=cutoff, satnames=satid)

    # ----------------------------------------------------------------------------
    # Plot satellite orbit tracks in ECI (right-ascension and declination)
    # ----------------------------------------------------------------------------

    # Compute right-ascension and declination of the observer and satellite

    if objcrd != None:
        aobj = np.degrees(np.arctan2(xobj[:, 1], xobj[:, 0]))
        dobj = np.degrees(np.arctan(xobj[:, 2]/np.sqrt(xobj[:, 1]**2+xobj[:, 0]**2)))
    asat = np.degrees(np.arctan2(xsat[:, 1], xsat[:, 0]))
    dsat = np.degrees(np.arctan(xsat[:, 2]/np.sqrt(xsat[:, 1]**2+xsat[:, 0]**2)))

    # Plot right ascension and declination 

    plt.figure("ECI tracks",figsize=figsize)
    if objcrd != None:
        plt.scatter(aobj, dobj, color="g", marker=".", s=1, label="observer")
    plt.scatter(asat, dsat, color="b", marker=".", s=1, label=satid)
    if objcrd != None:
        plt.scatter(aobj[visible], dobj[visible], marker="+", color="g", s=30, label="visible observer")
        plt.scatter(asat[visible], dsat[visible], marker="+", color="b", s=30, label="visible {}".format(satid))
    # plot time stamps
    stepsize = int(t.shape[0]/15)
    for k in range(0,stepsize,3):
        plt.scatter(asat[k],dsat[k], marker=".", color='k')
        #plt.text(asat[k]+5,dsat[k],"{:>.2f}h".format((t[k]-np.floor(t[0]))*24))
        plt.text(asat[k]+5,dsat[k], num2datetime(t[k]).strftime(xtimefmt))       
    for k in range(stepsize,t.shape[0],stepsize):
        plt.scatter(asat[k],dsat[k], marker=".", color='k')
        #plt.text(asat[k]+5,dsat[k],"{:>.2f}h".format((t[k]-np.floor(t[0]))*24))
        plt.text(asat[k]-5,dsat[k], num2datetime(t[k]).strftime(xtimefmt),horizontalalignment='right')       
    plt.axis([-180, 180, -90, 90])
    plt.legend()
    plt.xlabel("Right ascension")
    plt.ylabel("Declination")
    plt.title("{} orbit track in ECI ({})".format(satid,xlabeldate))

    # ----------------------------------------------------------------------------
    # Satellite ground tracks
    # ----------------------------------------------------------------------------

    # Substract GMST from right-ascension of observer and object in ECI to get the 
    # longitude in ECEF

    gst0, omegae = ut2gmst(t[0])
    gst = gst0*180/np.pi+360*omegae*(t-t[0])

    if objcrd != None:
        lobj = aobj-gst-360*np.round((aobj-gst)/360., 0)   # must be in the range [-180,+180]
    lsat = asat-gst-360*np.round((asat-gst)/360, 0)
    
    # Plot the ground tracks using the pltgroundtrack function

    plt.figure("Ground tracks", figsize=figsize)
    if objcrd != None:
        pltgroundtrack((lsat, dsat), visible=visible, satid=satid)
        plt.scatter(lobj, dobj, color="g", marker="*", s=49, label="observer")
    else:
        pltgroundtrack((lsat, dsat), satid=satid)
    plt.title("{} ground tracks ({} - {})".format(satid, num2datetime(t[0]).isoformat(), num2datetime(t[-1]).isoformat()))
    plt.legend() 

    # ----------------------------------------------------------------------------
    # 3D plot (ECI)
    # ----------------------------------------------------------------------------

    fig = plt.figure("3D satellite orbit (ECI)",figsize=figsize, tight_layout=True)
    plt.suptitle("{} 3D orbit ({} - {})".format(satid, num2datetime(t[0]).isoformat(), num2datetime(t[-1]).isoformat()))
    ax = fig.add_subplot(121, projection='3d')
    #ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    if objcrd != None:
        plot_orbit_3D(ax, xsat, xobj)
    else:
        plot_orbit_3D(ax, xsat)
    ax.view_init(30, 30)   # defaults to -60 (azimuth) and 30 (elevation)
    ax.set_title('Inertial (ECI)')
    
    # ----------------------------------------------------------------------------
    # 3D plot (ECEF)
    # ----------------------------------------------------------------------------

    # Compute rotation angle (GMST) around Z-axis

    gst0, omegae = ut2gmst(t[0])
    gst = gst0 + 2*np.pi*omegae*(t-t[0])
    if objcrd != None:
        lst = lon+gst

    # Rotate observer positions round z-axis (ECI->ECEF) 

    xsate = np.zeros(xsat.shape)
    xsate[:, 0] = np.cos(gst)*xsat[:, 0] + np.sin(gst)*xsat[:, 1]
    xsate[:, 1] = -np.sin(gst)*xsat[:, 0] + np.cos(gst)*xsat[:, 1]
    xsate[:, 2] = xsat[:, 2]

    if objcrd != None:
        xobje = np.zeros(xobj.shape)
        xobje[:, 0] = np.cos(gst) * xobj[:, 0] + np.sin(gst) * xobj[:, 1]
        xobje[:, 1] = -np.sin(gst) * xobj[:, 0] + np.cos(gst) * xobj[:, 1]
        xobje[:, 2] = xobj[:, 2]

    #fig = plt.figure("3D satellite orbit (ECEF)",figsize=figsize)
    ax = fig.add_subplot(122, projection='3d')
    #ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    if objcrd != None:
        plot_orbit_3D(ax, xsate, xobje)
        ax.scatter(xobje[:, 0]/1000, xobje[:, 1]/1000, xobje[:, 2]/1000, marker="*", s=49, color="r")
    else:
        plot_orbit_3D(ax, xsate)        
    ax.view_init(30, 30)   # defaults to -60 (azimuth) and 30 (elevation), rotate by 90 degree to make Delft visible
    ax.set_title('Earth Fixed (ECEF)')

def pltgroundtrack(xsate, satid="", visible=[], **kwargs):
    """ Plot satellite ground track(s).
    
    pltgroundtrack(xsate, ...) plot the satellite ground track for a satellite
    with in xsate the Cartesian ECEF coordinates. xsate must be a ndarray with 
    three columns, containing the X, Y, Z Cartesian ECEF coordinates in [m].
    
    pltgroundtrack((lon,lat), ...) plot the satellite ground track for a 
    satellite with the longitude and latitude given in the tuple (lon, lat).
    lon and lat must be ndarray with the ECEF longitude and latitude [degrees].
    
    The function takes optional parameters, satid, visible and/or any matplotlib, 
    argument, with satid the name of the satellite, and visible an optional logical array
    (of the same shape as lon and lat) with True for the elements where the
    satellite is visible.

    Coast lines are plotted when the file coast.mat is your working directory.

    If you want multiple satellites to be plotted call this function 
    repeatedly for the different satellites, like in the example below.
    
    Example:

       plt.figure("Ground tracks")
       pltgroundtrack((sat1, dsat), visible=visible, satid=satid)
       pltgroundtrack(xsate, satid=satid,linewidth=0.5)
       plt.title("Ground tracks for two satellites.")
       plt.legend() 

    See also tleread, tle2vec and eci2ecef.
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020
    
    Created:  30 November 2017 by Hans van der Marel for Matlab
    Modified: 19 November 2020 by Hans van der Marel and Simon van Diepen
                 - port to Python
    """

    # Check if we have Cartesian or Spherical coordinates

    if isinstance(xsate,tuple):
        # we hape a tuple with longitude and latitude
        lsat = xsate[0]
        dsat = xsate[1]
    elif xsate.shape[1] == 3:
        # we have Cartesian coordinates -> compute longitude and latitude
        lsat = np.arctan2(xsate[:,1], xsate[:,0]) * 180 / np.pi
        dsat = np.arctan(xsate[:,2] / np.sqrt(xsate[:,1]**2 + xsate[:,0]**2) )  * 180 / np.pi
    else:
        raise ValueError("xsate must be a tuple with (lon,lat) or ndarray with 3 columns with Cartesian coordinates!")
                
    # find the longitude and latitude roll over, and use masking to interupt the plotted line
    
    lsatmask = np.ma.array(lsat)
    dsatmask = np.ma.array(dsat)
    lsatmask[np.hstack([False,np.abs(np.diff(lsat)) > 180])] = np.ma.masked 
    dsatmask[np.hstack([False,np.abs(np.diff(dsat)) > 90])]= np.ma.masked 

    # Plot the axis and base map (only first time)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if not labels:
        plt.axis([-180, 180, -90, 90])
        if os.path.exists("coast.mat"):
            coast = loadmat("coast.mat")
            plt.plot(coast["long"], coast["lat"], color=(0.35, 0.35, 0.35), linewidth=0.5)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
    # Plot the ground tracks
    
    plt.plot(lsatmask, dsatmask, label=satid, **kwargs)
    #plt.scatter(lsat, dsat, s=5, label=satid)
    visible = np.array(visible)
    if visible.size > 0:
        plt.scatter(lsat[visible], dsat[visible], s=20, marker="*", label="visible {}".format(satid))


def skyplot(t, azi, zen, cutoff=0, satnames=[]):
    """ Skyplot (polar plot) with elevation and azimuth of satellite(s).
    
    skyplot(t,azi,zen) creates a polar plot with the elevation and 
    azimuth of a satellite. t is an array with the time as datenumbers, azi 
    and zen are arrays with the azimuth (radians) and zenith zenith angle 
    (radians). The first axis of azi and zen must have the same length as
    the array with time t. The second axis represents different satellites,
    i.e.  ntimes, nsat = zen.shape . The optional parameter cutoff is the 
    cutoff elevation (degrees) for plotting the trajectories, the default is
    0 degrees (the local horizon). 

    Example:

       tlegps = tleread('gps.txt','verbose',0)
       t = tledatenum(['2017-11-30',24*60,1])

       latlon = np.radians([ 52, 4.8 ]);
       xobje = 6378136 * [ np.cos(latlon[1]) * np.cos(latlon[2]) , 
                           np.cos(latlon[1]) * np.sin(latlon[2]) ,
                           np.sin(latlon[1]) ]

       nepo = t.shape[0]
       ngps = tlegps.shape[0]
       zgps = np.full([nepo,ngps],np.nan)
       agps = np.full([nepo,ngps],np.nan)
       for k in range(ngps)
          xgpsi, vgpsi = tle2vec(tlegps, t, k)
          xgpse, vgpse = eci2ecef(t, xgpsi, vgpsi)
          lookangles, _ = satlookanglesp(t, [xgpse, vgpse], xobje)
          zgps[:,k] = lookangles[:,1]
          agps[:,k] = lookangles[:,2]
          
       plt.figure(title)
       skyplot(t, agps, zgps, cutoff=10)

    See also tleread, tle2vec, eci2ecef and satlookanglessp. 
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020
    
    Created:   5 December 2017 by Hans van der Marel for Matlab
    Modified: 19 November 2020 by Hans van der Marel and Simon van Diepen
                 - port to Python
    """

    # reshape input arrays zen and azi from 1D to 2D with one column, or keep as 2D 
    zen = zen.reshape(zen.shape[0],-1)
    azi = azi.reshape(azi.shape[0],-1)

    # check the sizes of t, zen and azi
    if  zen.shape[0] != azi.shape[0] or zen.shape[1] != azi.shape[1]  : 
        raise ValueError("zen and azi input parameters must be of the same shape!") 
    elif t.shape[-1] != zen.shape[0]:
        raise ValueError("t must match the first dimension of zen and azi!") 
    
    # Check the optional satnames input
    if type(satnames) == str:
        satnames = [ satnames ]
    satnames = np.array(satnames)
    if satnames.size > 0:                # have optional satnames, check if size matches
        if satnames.shape[0] != zen.shape[1]:
            raise ValueError("the number of satellite names must match the number of columns in zen and azi!")

    # Prepare arrow tip

    xx = np.array([[-1, 0, -1]]).T
    yy = np.array([[0.4, 0, -0.4]]).T
    arrow = xx + 1j * yy
    
    # Plot the base figure
    
    #plt.figure(title)
    plt.axis([-90, 90, -90, 90])
    plt.axis("equal")
    lcol = "bgrcmk"
    for i in range(0, 360, 30):
        x, y = pol2cart(-np.radians(i-90), 90)
        plt.plot([0, x], [0, y], color='gray', linewidth=1)
        x, y = pol2cart(-np.radians(i-90), 94)
        plt.text(x, y+1, "{}".format(i), fontsize=8, horizontalalignment='center', verticalalignment='center', rotation=-i)

    i_values = np.append(np.arange(0, 91, 15), cutoff)
    for i in i_values:
        az = np.arange(361)
        el = (90-i)*np.ones(az.shape)
        x, y = pol2cart(np.radians(az), el)
        plt.plot(x, y, color='gray', linestyle='-' if i != cutoff else '--', linewidth=1)
        if i not in [cutoff, 0]:
            x, y = pol2cart(-np.radians(-90), 90 - i)
            plt.text(x, y, "{}".format(i), fontsize=8, horizontalalignment='center', verticalalignment='bottom')
            #plt.text(0, 90 - i, "{}".format(i),horizontalalignment='center', verticalalignment='bottom')

    plt.axis('off')
    
    # Plot the tracks
    
    nsat = zen.shape[1]
    for k in range(nsat):

        dt = np.diff(t).min()
        idx1 = np.arange(zen.shape[0])[zen[:,k] < np.radians(90-cutoff) ]
       
        if idx1.shape[0] > 0:
            idx2 = np.append(0, np.arange(idx1.shape[0]-1)[np.diff(t[idx1]) > 3*dt])
            idx2 = np.append(idx2, idx1.shape[0]-1)
           
            for j in range(idx2.shape[0]-1):
    
                idx3 = idx1[idx2[j]+1:idx2[j+1]]
    
                x, y = pol2cart(-azi[idx3,k] + np.pi/2, np.degrees(zen[idx3,k]))
                plt.plot(x, y, linewidth=2, color=lcol[k % len(lcol)])
    
                if idx3.shape[0] > 1:
                    tx, ty = pol2cart(-azi[idx3[-1],k] + np.pi / 2, np.degrees(zen[idx3[-1],k]))
                    txx, tyy = pol2cart(-azi[idx3[-2],k] + np.pi/2, np.degrees(zen[idx3[-2],k]))
                    dd = np.sqrt((tx-txx)**2+(ty-tyy)**2)
                    z = (tx-txx)/dd + 1j * (ty-tyy)/dd
                    a = arrow * z
                    plt.plot(tx + 3*a.real, ty + 3*a.imag, linewidth=2, color=lcol[k % len(lcol)])
                    tx += 6*(tx-txx)/dd
                    ty += 5*(ty-tyy)/dd
                else:
                    try:
                        tx += 5
                        ty += 2
                    except UnboundLocalError:
                        pass

                if satnames.size > 1:
                    plt.text(tx, ty, "{}".format(satnames[k]), fontsize=8, horizontalalignment='center', verticalalignment='bottom', color=lcol[k % len(lcol)])

    # add title
   
    if satnames.size == 1:
        plt.title("{} Skyplot ({} - {})".format(satnames[0], num2datetime(t[0]).isoformat(), num2datetime(t[-1]).isoformat()))
    else:
        plt.title("Skyplot ({} - {})".format(num2datetime(t[0]).isoformat(), num2datetime(t[-1]).isoformat()))


def plot_orbit_3D(ax, xsat, xobj=None):
    """
    Plots 3d orbits of satellites arount the Earth
    :param xsat:
    :param xobj:
    :param title:
    :return:
    """

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    Re = 6378136

    # Make data
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = Re * np.outer(np.cos(u), np.sin(v)) / 1000
    y = Re * np.outer(np.sin(u), np.sin(v)) / 1000
    z = Re * np.outer(np.ones(np.size(u)), np.cos(v)) / 1000

    # Plot the surface
    ax.plot_surface(x, y, z, shade=True, edgecolor="gray", linewidth=0.5, color=[.9, .9, .9], alpha=.2)
    #ax.plot_surface(x, y, z, color=[.9, .9, .9], alpha=0.2)
    #ax.plot_wireframe(x, y, z, color='gray', linewidth=0.5)
    if type(xobj) in [list, np.array]:
        ax.plot(xobj[:, 0]/1000, xobj[:, 1]/1000, xobj[:, 2]/1000, color='r', linewidth=2, alpha=1)
    ax.plot(xsat[:, 0]/1000, xsat[:, 1]/1000, xsat[:, 2]/1000, color='b', linewidth=2, alpha=1)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    
    set_axes_equal(ax)


if __name__ == "__main__":
    satid = 'gps'
    satellite = 'GPS BIIR-9  (PRN 21)'
    fname = tleget(satid)
    tlelist = tleread(fname)
    xsat, vsat = tle2vec(tlelist, ['2020-11-8 0:00:00', 24 * 60, 1], satellite)
    tleplot1(tlelist, ['2020-11-8 0:00:00', 24 * 60, 1], satellite, [52, 4.8, 0])

    plt.show()
