# © 2026 Thierry Marcoux et Zacharie Robitaille Jean.
import numpy as np
from pathlib import Path
import sys, os
import math
import pyvista as pv
from datetime import datetime, timedelta, timezone

from astropy.time import Time
from astropy.coordinates import get_body, ITRS
from astropy import units as u

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from SOL_Tools.Graph_tools import *
from SOL_Tools.Math_tools import *
from SOL_Tools.Orbit_tools import *
from SOL_Tools.RSO import RSO


def setup_plotter():
    pl = pv.Plotter(lighting="none", window_size=(1200, 900))  # kill the default three-light kit, we supply our own
    pl.render_window.SetPosition(100, 50)  # Position on compute screen, from top left origin
    return pl

def get_prime_vertical(phi_geod, a=1, e=0):
    """
    Calculates the prime vertical radius

    :param phi_geod: Geodetic latitude
    :param a: Semi major axis
    :param b: Semi minor axis
    """
    R_N = a / np.sqrt(1 - (e*np.sin(phi_geod))**2)

    return R_N

def geod_2_cart(phi_geod, lon, height_km):
    R_N = get_prime_vertical(phi_geod, a=Earth.r1_km, e=Earth.ecc)

    x = (R_N + height_km)*np.cos(phi_geod)*np.cos(lon)
    y = (R_N + height_km)*np.cos(phi_geod) * np.sin(lon)
    z = ((1 - Earth.ecc**2)*R_N + height_km)*np.sin(phi_geod)

    return x, y, z

def phi_geoc_2_geod(phi_geoc, e=Earth.ecc):
    phi_geod = np.arctan(np.tan(phi_geoc)/(1 - e**2))
    return phi_geod

def get_z0(x, y, z, R_N):
    z0 = np.sqrt(R_N**2 - x**2 - y**2) - z
    return z0

    
def compute_altitudes(positions):
    """Altitudes au-dessus de l’ellipsoïde, en km.
    positions : tableau ECEF de forme (3, N), en km.
    """
    x, y, z = np.asarray(positions, dtype=float)

    a = Earth.r1_km
    e2 = Earth.ecc**2

    # Distance à l’axe de rotation terrestre
    p = np.hypot(x, y)

    # Première estimation de la latitude géodésique [rad]
    phi = np.arctan2(z, p * (1 - e2))

    # Affinement de la latitude pour tous les points
    for _ in range(15):
        R_N = a / np.sqrt(1 - e2 * np.sin(phi)**2)
        phi = np.arctan2(z + e2 * R_N * np.sin(phi), p)

    # Hauteur suivant la normale à l’ellipsoïde
    h = (
        p * np.cos(phi)
        + z * np.sin(phi)
        - a * np.sqrt(1 - e2 * np.sin(phi)**2)
    )

    return h

def GetMoonPositionECEF(epoch):
    """
    Position apparente géocentrique de la Lune à une date UTC.

    Parameters
    ----------
    epoch : datetime
        Date avec fuseau horaire, par exemple timezone.utc.

    Returns
    -------
    position_km : ndarray, shape (3,)
        Coordonnées terrestres ITRS [x, y, z], en kilomètres.
    """

    t = Time(epoch, scale="utc")

    # Éphéméride JPL : position apparente depuis le centre terrestre. 
    # GCRS is distinct from ICRS mainly in that it is relative to the Earth’s center-of-mass rather than the solar system Barycenter.

    moon_gcrs = get_body( # -> https://docs.astropy.org/en/stable/api/astropy.coordinates.get_body.html
        "moon",
        t,
        ephemeris="de440s",
    )

    # Transformation vers le repère terrestre.
    moon_ecef = moon_gcrs.transform_to(ITRS(obstime=t))

    return moon_ecef.cartesian.xyz.to_value(u.km)





if __name__ == "__main__":
    rso = RSO(
        filename=PROJECT_ROOT / "SOL_Tools" / "TLE" / "TLE.txt"
    )

    # sat_name = "ISS (ZARYA)"
    sat_name = "MOLNIYA 3-50"
    rso.print_sat_info(sat_name=sat_name)

    # ------------------------------------------------------
    # Époque du satellite, extraite du TLE : YYDDD.dddddddd
    # ------------------------------------------------------
    sat_index = rso.name.index(sat_name)
    epoch_tle = rso.epoch[sat_index].strip()

    year_short = int(epoch_tle[:2])
    year = (
        2000 + year_short
        if year_short < 57
        else 1900 + year_short
    )

    day_of_year = float(epoch_tle[2:])
    date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)

    t_span = np.arange(120000, dtype=float)

    # ------------------------------------------------------
    # Conditions initiales et propagation ECI
    # ------------------------------------------------------
    coe = rso.get_coe(sat_name)
    r, v = COE_to_SV(COE=coe)
    sv = np.concatenate((r, v))

    # positions, velocities = OrbitPropagation_2BN(sv, t_span)
    positions, velocities, nus, Ms, Es = OrbitPropagation_2BK(coe, t_span)
    nus_deg = np.degrees(nus)
    Ms_deg = np.degrees(Ms)
    Es_deg = np.degrees(Es)
    # positions et velocities : tableaux de forme (3, N)

    # ------------------------------------------------------
    # Conversion ECI -> ECEF à chaque instant
    # ------------------------------------------------------
    dates = [
        date + timedelta(seconds=float(t))
        for t in t_span
    ]

    GMSTs = np.array([ComputeGMST(epoch=d)for d in dates])

    # RotZ attend un angle scalaire en degrés.
    # Avec la définition actuelle de RotZ : ECI -> ECEF = -GMST.
    positions_ECEF = np.empty_like(positions)
    velocities_ECEF = np.empty_like(velocities)

    # Vitesse angulaire terrestre [rad/s].
    omega_earth = np.array([0.0, 0.0, 7.292115e-5])

    for k, gmst in enumerate(GMSTs):
        rotation = RotZ(-gmst)

        positions_ECEF[:, k] = rotation @ positions[:, k]

        # Vitesse relative au repère terrestre en rotation.
        velocities_ECEF[:, k] = (
            rotation @ velocities[:, k]
            - np.cross(omega_earth, positions_ECEF[:, k])
        )

    # Le satellite affiché utilise déjà positions[:, -1].
    date_affichage = dates[-1]

    position_lune_ECEF = GetMoonPositionECEF(date_affichage)

    # tracé d'une révoluetion lunaire autour de la terre : 
    duree_lune_jours = 1
    nombre_points = 2000

    # Trjectoire pour une date précise :
    dates_lune = (
        Time(date_affichage, scale="utc")
        + np.linspace(
            -duree_lune_jours,
            0.0,
            nombre_points,
        ) * u.day
    )
    positions_lune_ECEF = GetMoonPositionECEF(dates_lune)

    print("Date d'affichage :", date_affichage.isoformat())
    print("Position lunaire ECEF [km] :", position_lune_ECEF)
    print(
        "Distance entre les centres Terre–Lune [km] :",
        np.linalg.norm(position_lune_ECEF),
    )




    # ------------------------------------------------------
    # Affichage ECI
    # ------------------------------------------------------
    pl = setup_plotter()
    pl = DrawEarth3D(pl=pl, view_mode="ECI")
    pl = setup_plotter()
    pl = DrawEarth3D(pl=pl, view_mode="ECEF")

    


    pl.add_points(
        positions.T,
        color="red",
        point_size=2,
    )
    
    vitesse = velocities[:, -1]

    # satellite = pv.Sphere(
    #     radius=100,  # km : taille exagérée pour être visible
    #     center=position,
    # )
    # pl.add_mesh(satellite, color="green")

    # direction = vitesse / np.linalg.norm(vitesse)

    # pl.add_arrows(
    #     position.reshape(1, 3),
    #     direction.reshape(1, 3),
    #     mag=1500,  # longueur visuelle de la flèche [km]
    #     color="yellow",
    # )

    pl.show(title="Orbite du satellite (ECI)")
    

    # ------------------------------------------------------
    # Affichage ECEF
    # ------------------------------------------------------
    pl = setup_plotter()
    pl = DrawEarth3D(pl=pl, view_mode="ECEF")
    pl = DrawMoon3D(
            pl=pl,
            position_km=position_lune_ECEF,
        )   
    
    trajectoire_lune = pv.lines_from_points(
            positions_lune_ECEF.T,
            close=False,
        )
    
    pl.add_mesh(
            trajectoire_lune,
            name="Trajectoire_Lune",
            color="red",
            line_width=2,
            lighting=False,
        )

    pl.add_points(
        positions_ECEF.T,
        color="red",
        point_size=2,
    )

    position_ECEF = positions_ECEF[:, -1]
    vitesse_ECEF = velocities_ECEF[:, -1]

    satellite_ECEF = pv.Sphere(
        radius=100,
        center=position_ECEF,
    )
    pl.add_mesh(satellite_ECEF, color="green")

    direction_ECEF = vitesse_ECEF / np.linalg.norm(vitesse_ECEF)

    pl.add_arrows(
        position_ECEF.reshape(1, 3),
        direction_ECEF.reshape(1, 3),
        mag=1500,
        color="yellow",
    )
    pl.reset_camera()


    # Deux dernières positions de la trajectoire. -> Tracer la flèche de la trajectoire 
    position_actuelle = positions_lune_ECEF[:, -1]
    position_precedente = positions_lune_ECEF[:, -2]

    # Temps écoulé entre les deux positions [s].
    dt = (dates_lune[-1] - dates_lune[-2]).to_value(u.s)

    # Estimation de la vitesse [km/s].
    vitesse_lune = (
        position_actuelle - position_precedente
    ) / dt

    # Norme de la vitesse [km/s].
    norme_vitesse = np.linalg.norm(vitesse_lune)

    if norme_vitesse > 0:
        # Direction unitaire du mouvement.
        direction_lune = vitesse_lune / norme_vitesse

        pl.add_arrows(
            position_actuelle.reshape(1, 3),
            direction_lune.reshape(1, 3),
            mag=30000,  # Longueur visuelle de la flèche [km].
            color="yellow",
            lighting=False,
        )
    pl.show(title="Terre, Lune et satellite — ECEF")


    lat_ulaval = 46.7809
    lon_ulaval = -71.2778
        # Latitude géodésique et longitude en radians
    phi = np.radians(lat_ulaval)
    lam = np.radians(lon_ulaval)
    rayon = 10
    coord_ulaval = Sph2Cart(lon_ulaval, lat_ulaval)

    ### Altitudes et latitudes et longitudes

    lons_sat, lats_geoc_sat, rs_sat = Cart2Sph(positions_ECEF)
    lats_geod_sat = phi_geoc_2_geod(lats_geoc_sat)

    altitudes = compute_altitudes(positions_ECEF)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(t_span, np.degrees(lats_geod_sat))
    ax[0].set_ylabel("Latitude géodésique [°]")
    ax[0].set_xlabel("Temps [s]")
    ax[0].set_title("Latitude géodésique")
    ax[0].set_xlim([0, t_span[int(len(t_span)/10)]])
    ax[1].plot(t_span, np.degrees(lons_sat))
    ax[1].set_ylabel("Longitude [°]")
    ax[1].set_xlabel("Temps [s]")
    ax[1].set_title("Longitude")
    ax[2].plot(t_span, altitudes)
    ax[2].set_ylabel("Altitude [km]")
    ax[2].set_xlabel("Temps [s]")
    ax[2].set_title("Altitude")

    plt.show()

    ### 

    lat_cercle, lon_cercle, xyz = SmallCircle(
        az=lon_ulaval,
        el=lat_ulaval,
        alpha=rayon,
        t=np.linspace(0, 360, 360, endpoint=False)
    )

    points_cercle = xyz.T * (Earth.r1_km + 5.0)
    cercle = pv.lines_from_points(points_cercle, close=True)
    
    position_ulaval = np.array(
        geod_2_cart(
            phi_geod=phi,
            lon=lam,
            height_km=0
        )
    )
    
    normale_geodetic = Sph2Cart(lon_ulaval, phi_geoc_2_geod(lat_ulaval))
    
        # Direction normale à l’ellipsoïde : vecteur unitaire
    normale_geodetic = np.array([
        np.cos(phi) * np.cos(lam),
        np.cos(phi) * np.sin(lam),
        np.sin(phi)
    ])
        # Flèche partant de l’Université Laval
    vecteur_normal = pv.Arrow(
        start=position_ulaval,
        direction=normale_geodetic,
        scale=1500,        # longueur visuelle [km]
        tip_length=0.15,
        tip_radius=0.04,
        shaft_radius=0.01
    )

    # Utiliser le même plotter que pour la Terre et le satellite
    pl.add_mesh(vecteur_normal, color="blue", lighting=False)
    pl.add_mesh(cercle, color="gray", line_width=4, lighting=False)

    # pl.show(title="Normale à l’Université Laval — ECEF")
    
    # Calcul des anomalies 

    plt.plot(t_span / 60, Ms_deg % 360, label="Anomalie moyenne")
    plt.plot(t_span / 60, Es_deg % 360, label="Anomalie excentrique")
    plt.plot(t_span / 60, nus_deg % 360, label="Anomalie vraie")

    plt.xlabel("Temps [min]")
    plt.ylabel("Anomalie [°]")
    plt.legend()
    plt.grid()
    plt.show()





















