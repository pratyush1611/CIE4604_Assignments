"""
Function to compute specular points with test script

By Hans van der Marel, Delft Universtiy of Technology, developed for CIE4604
Simulation and Visualization course, assignment 2.
"""

import numpy as np
from scipy.optimize import root

def specularpoint(xr, xt, verbose=False):
    """Compute position vector of specular reflection point

    xp=specularpoint(xr,xt) computes the cartesian coordinate vector 
    xp of the specular reflection point from the cartesian coordinate
    vector xt of the transmitting station and xr of the receiving station.

    Instead of cartesian coordinates, xr and xt may also be given as 
    a triplet with [ latitude, longitude, height ]. The latitude and longitude
    are in degrees, the height is in kilometers. When xr or xt is given
    as spherical latitude and longitude coordinates the option verbose is
    automatically turned on.
    
    When using Cartesian coordinates the verbose option is by default False,
    as this is the case for production runs.

    Examples
   
      xp=specularpoint(xr,xt)
      xp=specularpoint([51, 4, 700 ], [53,  20, 20000 ])

    (c) Hans van der Marel, Delft University of Technology, 2018-2020.
    """

    # Constants
    Re = 6378136

    # Convert to numpy arrays
    xr = np.array(xr)
    xt = np.array(xt)
    
    # Check input format
    #
    # 1. [ lat lon H ] in [ deg , deg, km]
    # 2. [ X Y Z ] in m

    if np.sqrt(np.sum(xr**2)) < 6000000: 
        xr = sphere2cart(xr[0], xr[1], xr[2]*1000)
        verbose = True
        
    if np.sqrt(np.sum(xt**2)) < 6000000: 
        xt = sphere2cart(xt[0], xt[1], xt[2]*1000)
        verbose = True

    # Compute radius
    Rt = np.sqrt(np.sum(xt**2))
    Rr = np.sqrt(np.sum(xr**2))

    # Compute zenith and theta angles
    zr, ar, _ = xyz2zas(xr,xt)
    zt, at, _ = xyz2zas(xt,xr)

    theta = zr + zt - np.pi

    if verbose:
        print('Zenith/azimuth angle transmitter  {:>12.2f}{:>12.2f} [deg]'.format(zt*180/np.pi,at*180/np.pi))
        print('Zenith/azimuth angle receiver     {:>12.2f}{:>12.2f} [deg]'.format(zr*180/np.pi,ar*180/np.pi))
        print('Theta angle (total)               {:>12.2f} [deg]'.format(theta*180/np.pi))

    # Compute incidence angle
    #
    # This is the business end of this function. In order to find the 
    # incidence angle we have to solve an equation. We use for this
    # the Scipy root function (equivalnet of Matlab function fzero for 
    # single-variable nonlinear zero finding), with two inputs
    #
    # * lambda function for which we want to find a zero, with x the incidence angle,
    # * start value for the zero finding (theta)
    #
    # The function root returns an object, called fzero, with the solution 
    # (the incidence angle we are after) and other performance paramters
    
    # Legacy Matlab code ...
    # fun=@(x)asin(Re/Rr*sin(x))+asin(Re/Rt*sin(x))+theta-2*x; 
    # incidence=fzero(fun,theta)

    fzero = root(lambda x: np.arcsin(Re/Rr*np.sin(x))+np.arcsin(Re/Rt*np.sin(x))+theta-2*x , theta)
    incidence = fzero.x[0]
    
    if verbose:
        print('\nIncidence angle                   {:>12.2f} [deg]'.format(incidence*180/np.pi))
        print(fzero)

    # Compute theta angles for receiver and transmitter
    thetar = incidence - np.arcsin(Re/Rr*np.sin(incidence))
    thetat = incidence - np.arcsin(Re/Rt*np.sin(incidence))

    if verbose:
        print('Theta angle transmitter           {:>12.2f} [deg]'.format(thetat*180/np.pi))
        print('Theta angle receiver              {:>12.2f} [deg]'.format(thetar*180/np.pi))

    # Print nadir angles for receiver and transmitter
    if verbose:
        print('Nadir angle transmitter           {:>12.2f} [deg]'.format((incidence-thetat)*180/np.pi))
        print('Nadir angle receiver              {:>12.2f} [deg]'.format((incidence-thetar)*180/np.pi))

    # Reflection point ... this gives the final result xp
    #
    # 1. Compute rotation axis
    n = np.cross(xr,xt)
    n = n / np.sqrt(np.sum(n**2))

    # 2. Rotate xr over thetar using Rodriques rotation formula and scale to Earth radius
    xp = Re/Rr * ( np.cos(thetar) * xr + np.sin(thetar) * np.cross(n,xr) + 
                  ( 1 - np.cos(thetar)) * np.dot(n,xr) * n )
    
    # This is our final result, the rest is just printing ...

    # Print positions in latitude, longitude and height (see also cart2sphere)
    if verbose:
        print('\nTransmitter position          {:>8.2f}{:>8.2f}    {:>8.2f} [deg,deg,km]'
              .format(np.arcsin(xt[2]/Rt)*180/np.pi,np.arctan2(xt[1],xt[0])*180/np.pi,(Rt-Re)/1000))
        print('Receiver position             {:>8.2f}{:>8.2f}    {:>8.2f} [deg,deg,km]'
              .format(np.arcsin(xr[2]/Rr)*180/np.pi,np.arctan2(xr[1],xr[0])*180/np.pi,(Rr-Re)/1000))
        print('Specular Reflection Point     {:>8.2f}{:>8.2f}    {:>8.2f} [deg,deg,km]'
              .format(np.arcsin(xp[2]/Re)*180/np.pi,np.arctan2(xp[1],xp[0])*180/np.pi,
                      (np.sqrt(np.sum(xp**2))-Re)/1000))

    # Print nadir angles and azimuth angles for receiver and transmitter to reflection point
    if verbose:
        zrp, arp, _ = xyz2zas(xr,xp)
        ztp, atp, _ = xyz2zas(xt,xp)
        print('\nNadir/azimuth angle transmitter   {:>12.2f}{:>12.2f} [deg]'.format(180-ztp*180/np.pi, atp*180/np.pi))
        print('Nadir/azimuth angle receiver      {:>12.2f}{:>12.2f} [deg]'.format(180-zrp*180/np.pi, arp*180/np.pi))

    # do internal check 
    if verbose:
        n=-n
        xp2 = Re/Rt * ( np.cos(thetat) * xt + np.sin(thetat) * np.cross(n,xt) + 
                       ( 1 - np.cos(thetat)) * np.dot(n,xt) * n )
        print('Checks: theta {},  xp  {}'.format(theta-thetar-thetat,np.sqrt(np.sum((xp-xp2)**2))))

    return xp

def xyz2zas(xfrom, xto):
    """Compute zenith angle [rad], azimuth [rad] and range [m] between points.
    
    z, a, s = xyz2zas(xfrom, xto) compute the zenith angle z [rad], azimuth
    angle a [rad] and range s [m] from point xfrom to xto, with xfrom and xto
    given by their Cartesian coordinates [X, Y, Z] in [m].
    """
    
    rfrom = np.sqrt(np.sum(xfrom**2))
    n0 = xfrom / rfrom
    xrs = xto - xfrom
    s = np.sqrt(np.sum(xrs**2)) 
    ers = xrs / s
    
    ip = np.sum(n0 * ers)
    z = np.arccos(ip)
    a = np.arctan2( -n0[1] * ers[0] + n0[0] * ers[1], -ip * n0[2] + ers[2])
    a += 2*np.pi
    a %= 2*np.pi
    
    return z, a, s

def sphere2cart(lat,lon,height):
    """Compute Cartesian coordinates from spherical latitude, longitude and height.

    x = sphere2cart(lat, lon, height) computes the Cartesian coordinates [X,Y,Z]
    from the spherical latitude lat [deg], longitude lon [deg] and height [m].
    It returns the cartesian coordinates in an array x with the X, Y and Z
    coordinates in [m].
    """

    Re = 6378136
    
    tmpb = [ np.cos(lat*np.pi/180) * np.cos(lon*np.pi/180), 
             np.cos(lat*np.pi/180) * np.sin(lon*np.pi/180), 
             np.sin(lat*np.pi/180) ]
    tmpa = Re + height 
    x = tmpa * np.array(tmpb)
    
    return x

def cart2sphere(x):
    """Compute spherical latitude, longitude and height from Cartesian coordinates.

    lat, lon, height = cart2sphere(x) computes the spherical latitude lat [deg], 
    longitude lon [deg] and height [m] from Cartesian coordinates [X,Y,Z].
    The cartesian coordinates are given in an array x with the X, Y and Z
    coordinates in [m].
    """

    Re = 6378136

    R = np.sqrt(np.sum(x**2))

    lat = np.arcsin(x[2]/R) * 180/np.pi
    lon = np.arctan2(x[1], x[0]) * 180/np.pi
    height = R-Re
    
    return lat, lon, height

if __name__ == "__main__":

    # Compute Cartesian coordinates of specular point from given input in
    # latitude, longitude and height for a LEO and GPS satellite
    xsp = specularpoint([51., 4., 700. ], [53.,  20., 20000. ]) 
    latsp, lonsp, heightsp = cart2sphere(xsp)
    print('\nCartesian coordinates [km] specular point  ',xsp/1000)
    print('Lat, lon and height [deg/km] specular point',[latsp, lonsp, heightsp/1000])

    # Do the same, but with Cartesian coordinates as input (verbose is off now)
    xr = sphere2cart(51., 4., 700.*1000)
    xt = sphere2cart(53.,  20., 20000.*1000) 
    print('\nCartesian coordinates [km] LEO receiver    ',xr/1000)
    print('Cartesian coordinates [km] GPS transmitter ',xt/1000)
    xsp =specularpoint(xr,xt) 
    latsp, lonsp, heightsp = cart2sphere(xsp)
    print('Cartesian coordinates [km] specular point  ',xsp/1000)
    print('Lat, lon and height [deg/km] specular point',[latsp, lonsp, heightsp/1000])
