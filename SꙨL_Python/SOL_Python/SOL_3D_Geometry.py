# ╭──────────────────────╮
# │  SOL_3D_Geometry.py  │ 3D geometry drawing with Python
# ╰──────────────────────╯
#  © SꙨL, SpaceOrbitLAB 
"""
# (Header to include here)

(main).py

SOL_Tools/
  ├─ AstroConstants.py
  ├─ Orbit_tools.py
  ├─ Math_tools.py
  └─ Graph_tools.py

  Visual Studio Code: Color Theme: Light Modern
"""

# General Python tools
import numpy as np
import pyvista as pv  # https://docs.pyvista.org/getting-started/
                      # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_title
import os
import sys

# SOL specific tools
from SOL_Tools.AstroConstants import *  # all variales
from SOL_Tools.Orbit_tools import *  # OrbitPropagation_2BN, OrbitPropagation_2BK, COE_to_SV, ...
from SOL_Tools.Graph_tools import *  # DrawEarth3D, DrawAngularMomentumVector
from SOL_Tools.Math_tools import * 

from datetime import datetime, timezone
from astropy.time import Time


if __name__ ==  "__main__":

    if False:  # (debugging)
        print("Earth.mu = ", Earth.mu)
        print("Earth.ecc = ", Earth.ecc)
        print('')
        print(os.path.dirname(__file__))
    print('')

    # Initialize the 3D PyVista Scene
    pl = pv.Plotter(lighting="none", window_size = (1200, 900))  # kill the default three-light kit, we supply our own
    pl.render_window.SetPosition(100, 50)  # Position on compute screen, from top left origin
    # pl.set_background("black")  → nice, but impractical for course notes and printing

    front_light = pv.Light(light_type = "headlight", intensity = 0.9)
    pl.add_light(front_light)

    #--- Defines Epoch
    #Date_UTC = datetime.now(timezone.utc)
    # Reference epoch:
    Date_UTC = datetime(2026,9,3, 12,0,0)  # → GMST = 162.637288334314° at this date; At Noon, the Sun will be at Greenwich meridian
    #Date_UTC = datetime(2026,6,21, 8,24,50)  # Spring Equinox when Sun is at it's highest declination
    #Date_UTC = datetime(2026,9,23, 00,5,13) # Autumn Equinox when Sun is at 
    if False:
        Date_UTC -= timedelta(hours=24 * -71.2778252/360)  # → brings the Sun exactly in Quebec's meridian
        print(Date_UTC)
        
    # Computes the Greenwich Mean Sidereal Time corresponding to this Epoch
    GMST = ComputeGMST(Date_UTC)  # [°]

    str = f'Epoch = {Date_UTC} → {Date_UTC.strftime('%Y-%m-%d %H:%M:%S')} UTC → GMST = {GMST:.4f}°'
    print(str)
    txt = pl.add_text(str, position = "upper_left", font_size = 12, font = 'courier')
    #txt = pl.add_text(str, position = "upper_left", font_size = 8, font_file='C:/Windows/Fonts/Calibri.ttf')
    #txt.prop.bold = False

    t = Time.now()  # Astropy: Computes all key time components
    #print(t.mjd, t.jd, t.iso, t.tt, t.tai)

    Sun = SunEphemerides(Date_UTC)
    #Sun.RA=35
    #Sun.Dec=25
    Sun.Lon = Sun.RA - GMST  # [°] Sun's Longitude (ECEF)
    str = f'Sun RA = {Sun.RA:.4f}, Dec = {Sun.Dec:.4f}, Lon = {Sun.Lon:.4f}°, Dist = {Sun.Dist_AU:.4f} AU = {Sun.Dist_km:.0f} km'
    print(str)
    pl.add_text(str, position = "lower_left", font_size = 13, font='courier')

    
    #ViewMode = 'ECI'    # Earth-Centered, Intertial
    ViewMode = 'ECEF'   # Earth-Centered, Earth-Fixed

    pl = DrawEarth3D(pl, ViewMode)

    # initial view, will be re-adjusted later
    view(pl, -35, 40)
    pl.camera.zoom(1.2)  # useful to zoom/unzoom


    # (0) Defines locations
    # TODO  use DefineObserverLocation() 

    class Greenwich:
        Name = 'Greenwich'
        Lat_geod = +51.4934  # [°N]
        Lon      = 0.0       # [°] ≡ 0° by definition as being the Prime Meridian
        pos, Lat_geoc, R_e, R_N, R_M, x0,z0 = \
            geod_to_pos(Lat_geod, Lon, 0.0, Earth.r1_km,Earth.r2_km)
    print('Greenwich: ϕ =', Greenwich.Lat_geod, ' → φ ', Greenwich.Lat_geoc)
                
    class Quebec:
        Name = 'Québec'
        Lat_geod = +46.78095  # [°N]
        Lon      = -71.27783  # [°W]
        pos, Lat_geoc, R_e, R_N, R_M, x0,z0 = \
            geod_to_pos(Lat_geod, Lon, 0.0, Earth.r1_km,Earth.r2_km)
    print('Québec: ϕ =', Quebec.Lat_geod, ' → φ ', Quebec.Lat_geoc)

    # (1) Starting point at λ = 0°, φ = 0°
    pos0 = Sph2Cart(0, 0, Earth.r1_km)  # OK to use on an ellipsoid as at ϕ = 0°
    pl.add_points(np.column_stack(pos0), color = 'red', point_size = 8, render_points_as_spheres=True, lighting=False)
    # also:
    #  point = pv.Sphere(radius = Earth.r1_km / 50, center = np.column_stack(pos0))
    #  pl.add_mesh(point, color = 'red', show_edges = False, ambient = 1)

    # (2) Computes point location from equator up to Greenwich
    if False:  # Spherical Earth
        pos1 = RotY(-Greenwich.Lat_geoc) @ pos0;  # ← requires a negative sign, as latitudes are measured clockwise, 
                                                  #   while the RotY rotation operator is defined counter-clockwise
    else:
        pos1, *rest = geod_to_pos(Greenwich.Lat_geod, 0, 0, Earth.r1_km,Earth.r2_km)
    pl.add_points(np.column_stack(pos1), color = 'red', point_size = 8, render_points_as_spheres = True, lighting = False)

    pos3, *rest = geod_to_pos(Quebec.Lat_geod, Quebec.Lon, 0, Earth.r1_km,Earth.r2_km)
    pl.add_points(pos3, color = 'green', point_size = 8, render_points_as_spheres=True, lighting=False) 

    if False:  # (temporary exit)
        pl.show()

    # (3) Draws arc line up to Greenwich
    phi_geoc_vect = np.linspace(0, Greenwich.Lat_geoc, 100)
    if False:
        arc_geoc = Sph2Cart(np.zeros_like(phi_geoc_vect), phi_geoc_vect, Earth.r1_km)  # Arc on a sphere
    else:  # Arc on geoid
        arc_geod, _ = geoc_to_pos(phi_geoc_vect, np.zeros_like(phi_geoc_vect), 0.0, Earth.r1_km,Earth.r2_km)  
    if False:
        print('arc shape =', arc_geod.shape) 
        print('arc[start]  =', arc_geod[:,0])
        print('arc[end]=', arc_geod[:,-1])
    Plot3D(pl, arc_geod, color = 'red', line_width = 2.5)  


    # (4a) Draw Earth Parallel at Greenwich latitude
    theta = np.linspace(0.0, 360, 100)  # [°]
    if False:
        radius_at_lat = Greenwich.R_N * cosd(Greenwich.Lat_geod)  # = r_δ
        r_K = Greenwich.R_N * sind(Greenwich.Lat_geod)
        C = Sph2Cart(theta, np.zeros_like(theta), radius_at_lat)  # disc at equator
        C[2, :] += r_K + Greenwich.z0
    else:
        C, *rest = geod_to_pos(Greenwich.Lat_geod * np.ones_like(theta), theta, 0, Earth.r1_km,Earth.r2_km)
    Plot3D(pl, EXPAND_FACTOR * C, color = 'cyan', line_width = 2.0)

    # (4b) Using Small Circle
    phi_ = atand(tand(Greenwich.Lat_geoc) / Earth.sf)  # == atand(pos1[2]/Earth.sf / pos1[0])
    alpha = 90 - phi_
    lat, lon, xyz_SC = SmallCircle(0, 90, alpha)  # unitary SC
    xyz_SC *= Earth.r1_km
    xyz_SC[2,:] *= Earth.sf  # ← OK in this special case
    if False:
        print('φ = ', lat[0], lat[-1])
        print('λ = ', lon[0], lon[-1])
        print('ρ[0] = ', xyz_[:,1])
    Plot3D(pl, EXPAND_FACTOR * xyz_SC, color = 'green')


    # (5) Computes point location westward to longitude λ = 71.2°W (Quebec City longitude)
    pos2 = RotZ(+Quebec.Lon) @ pos0   # ← positive here, as longitude is computed
                                      #   counter-clockwise, the same as RotZ operator
    pl.add_points(np.column_stack(pos2), color = 'lime_green', point_size = 8, render_points_as_spheres=True, lighting=False)


    # (6&7) Draws arc from Equator
    phi_geoc_vect = np.linspace(0, Quebec.Lat_geoc, 100)
    arc_geod, _ = geoc_to_pos(phi_geoc_vect, Quebec.Lon + np.zeros_like(phi_geoc_vect), 0.0, Earth.r1_km,Earth.r2_km)  # Arc on a geoid
    Plot3D(pl, arc_geod, color = 'green', line_width = 2.5) 
    pl.add_points(arc_geod[:,-1], color = 'green', point_size = 8, render_points_as_spheres=True, lighting=False)  # (7) Draws last point of arc


    #--- Draws Sun vector pointing to Earth Center and add lighting in this direction

    # (8) Draw Sun vector pointing to Earth center, knowing Sun longitude λ_⊙ and Sun declination δ_⊙
    DrawSunVector(pl, Sun, ViewMode)

    # (9) Draws Ecliptic line knowing Earth obliquity 𝜖 and GMST angle at epoch
    DrawEclipticLine(pl, GMST, Earth)

    # (10) Draws the Great Circle arc between Miami & Greenwich
    city1 = 'Miami, USA'
    lat1 =  25.7636  # [°]
    lon1 = -80.1296  # [°]

    city2 = 'Greenwich'
    lat2 = Greenwich.Lat_geod  # [°]
    lon2 = 0.0  # [°]

    GC = GreatCircle2(lat1, lon1, lat2, lon2)  # geodetic coordinates on sphere

    # NO: Plot3D(pl, EXPAND_FACTOR * GC.xyz * Earth.r1_km, color = 'red')  
    GC_path, phi_geoc,*rest = geod_to_pos(GC.lat, GC.lon, 0.0, Earth.r1_km, Earth.r2_km)
    Plot3D(pl, EXPAND_FACTOR * GC_path, color = 'blue', line_width = 2)  


    # (11) Draws topocentric SEZ plane at observer's position
    SEZ = DrawSEZPlane(pl, Quebec, 1000)

    # to view perpendicular to Terminator
    # view(pl, Sun.Lon - 90, 0)

    print('Done.')  # appears when all computations done

    #--- Shows the final figure
    pl.show(title = '© SpaceOrbitLAB')

    print('Exit.')  # appears when closing the window

# end __main__