"""Visualiseur interactif de la position géocentrique de la Lune."""

from datetime import datetime, timedelta, timezone
import csv
import json
import os
from pathlib import Path
import re
from queue import Empty, Queue
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError
from threading import Lock, Thread
import tempfile
from time import monotonic
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkCellPicker,
    vtkPolyDataMapper2D,
)
from SOL_Tools.AstroConstants import Earth, Moon
from SOL_Tools.Graph_tools import geodetic_ellipsoid_mesh
from SOL_Tools.Orbit_tools import ComputeGMST


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "SOL_Tools" / "DATA" / "Images"
DISPLAY_RADIUS_SCALE = 8.0
SUN_VISUAL_DISTANCE_KM = 4_500_000.0
SKY_SPHERE_RADIUS_KM = 5_000_000.0
MAX_CAMERA_RADIUS_FRACTION = 0.95
MOON_LINK_DASH_COUNT = 72
MOON_LINK_DASH_FRACTION = 0.38
TIMER_INTERVAL_MS = 100
DEFAULT_SIMULATION_DAYS_PER_SECOND = 1.0
WINDOW_SIZE = (1440, 900)
SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
# The lunar FK supplies the generic MOON_ME name for the DE440 orientation PCK.
SPICE_KERNELS = (
    ("lsk/naif0012.tls", 1_000),
    ("pck/pck00011.tpc", 10_000),
    ("fk/satellites/moon_de440_250416.tf", 10_000),
    ("pck/moon_pa_de440_200625.bpc", 1_000_000),
    ("spk/planets/de440s.bsp", 10_000_000),
)
_SPICE_KERNELS_LOADED = False
_SPICE_LOADED_KERNEL_PATHS = set()
# Seuil provisoire pour l'estimation de disponibilité; à adapter à l'antenne.
MIN_EARTH_ELEVATION_DEG = 10.0
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_EDGE_TOLERANCE = timedelta(minutes=1)
CISLUNAR_SATELLITES = {
    "lro": {
        "name": "LRO",
        "description": "orbiteur lunaire polaire",
        "horizons_id": "-85",
        "color": "#FFD166",
        "half_window_days": 3,
        "step_size": "10m",
    },
    "capstone": {
        "name": "CAPSTONE",
        "description": "orbiteur NRHO",
        "horizons_id": "-1176",
        "color": "#7DD3FC",
        "half_window_days": 30,
        "step_size": "2h",
    },
}
DSN_STATIONS = (
    {
        "name": "Goldstone",
        "antenna": "DSS-14",
        "latitude_deg": 35.4259,
        "longitude_deg": 243.11046,
        "color": "#F3B45B",
    },
    {
        "name": "Madrid",
        "antenna": "DSS-63",
        "latitude_deg": 40.43121,
        "longitude_deg": 355.75198,
        "color": "#58C4DD",
    },
    {
        "name": "Canberra",
        "antenna": "DSS-43",
        "latitude_deg": -35.402424,
        "longitude_deg": 148.98126,
        "color": "#B49BFF",
    },
)


def _parse_horizons_datetime(value):
    """Convertit une date Horizons anglaise en datetime UTC."""
    match = re.search(
        r"(?P<year>\d{4})-(?P<month>[A-Za-z]{3})-(?P<day>\d{2})\s+"
        r"(?P<hour>\d{2}):(?P<minute>\d{2})"
        r"(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?",
        value,
    )
    if match is None:
        raise ValueError(f"Date Horizons illisible : {value}")

    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    values = match.groupdict()
    fraction = (values["fraction"] or "")[:6].ljust(6, "0")
    return datetime(
        int(values["year"]),
        months[values["month"].lower()],
        int(values["day"]),
        int(values["hour"]),
        int(values["minute"]),
        int(values["second"] or 0),
        int(fraction or 0),
        tzinfo=timezone.utc,
    )


def _horizons_result(parameters):
    """Exécute une requête JSON à l'API JPL Horizons."""
    query = urlencode(parameters)
    request = Request(
        f"{HORIZONS_API_URL}?{query}",
        headers={"User-Agent": "SOL-Python lunar visualization"},
    )
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload.get("result", "")


def _horizons_trajectory_coverage(spacecraft_id):
    """Récupère la période de trajectoire annoncée par Horizons."""
    result = _horizons_result(
        {
            "format": "json",
            "COMMAND": f"'{spacecraft_id}'",
            "OBJ_DATA": "'YES'",
            "MAKE_EPHEM": "'NO'",
        }
    )
    section = result.partition("SPACECRAFT TRAJECTORY:")[2]
    if not section:
        raise RuntimeError("Horizons ne fournit pas de période de trajectoire.")

    date_pattern = (
        r"\d{4}-[A-Za-z]{3}-\d{2}\s+\d{2}:\d{2}"
        r"(?::\d{2})?"
    )
    row_pattern = re.compile(
        rf"(?P<start>{date_pattern})\s+(?P<stop>{date_pattern})\s*$",
        re.MULTILINE,
    )
    intervals = [
        (_parse_horizons_datetime(match["start"]),
         _parse_horizons_datetime(match["stop"]))
        for match in row_pattern.finditer(section)
    ]
    if not intervals:
        raise RuntimeError("Horizons n'a indiqué aucune fenêtre de données.")
    return min(start for start, _ in intervals), max(
        stop for _, stop in intervals
    )


def _load_horizons_satellite_track(satellite, selected_date, coverage=None):
    """Charge un court segment détaillé de trajectoire géocentrique."""
    spacecraft_id = satellite["horizons_id"]
    if coverage is None:
        coverage = _horizons_trajectory_coverage(spacecraft_id)
    coverage_start, coverage_stop = coverage
    selected_date = selected_date.astimezone(timezone.utc)
    if not (
        coverage_start - HORIZONS_EDGE_TOLERANCE
        <= selected_date
        <= coverage_stop + HORIZONS_EDGE_TOLERANCE
    ):
        return {
            "coverage": coverage,
            "unavailable": True,
            "error": "Aucune donnée Horizons pour cette date.",
        }

    half_window = timedelta(days=satellite["half_window_days"])
    start = max(selected_date - half_window, coverage_start)
    stop = min(selected_date + half_window, coverage_stop)
    result = _horizons_result(
        {
            "format": "json",
            "COMMAND": f"'{spacecraft_id}'",
            "OBJ_DATA": "'NO'",
            "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'VECTORS'",
            "CENTER": "'500@399'",
            "START_TIME": f"'{start:%Y-%m-%d %H:%M:%S}'",
            "STOP_TIME": f"'{stop:%Y-%m-%d %H:%M:%S}'",
            "STEP_SIZE": f"'{satellite['step_size']}'",
            "REF_PLANE": "'FRAME'",
            "REF_SYSTEM": "'ICRF'",
            "OUT_UNITS": "'KM-S'",
            "VEC_TABLE": "'1'",
            "VEC_CORR": "'NONE'",
            "CSV_FORMAT": "'YES'",
            "VEC_LABELS": "'NO'",
            "TIME_TYPE": "'UT'",
            "TIME_DIGITS": "'SECONDS'",
        }
    )
    try:
        table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    except IndexError as error:
        raise RuntimeError("Horizons n'a renvoyé aucun vecteur.") from error

    times = []
    positions = []
    for row in csv.reader(table.splitlines()):
        if len(row) < 5:
            continue
        try:
            times.append(_parse_horizons_datetime(row[1].replace("A.D.", "")))
            positions.append([float(value) for value in row[2:5]])
        except (ValueError, KeyError):
            continue
    if len(positions) < 2:
        raise RuntimeError("La trajectoire reçue contient trop peu de points.")
    return {
        "coverage": coverage,
        "start": times[0],
        "stop": times[-1],
        "times": np.asarray([value.timestamp() for value in times]),
        "positions": np.asarray(positions, dtype=float),
    }


def _spice_kernel_cache_dir():
    """Retourne le cache local des kernels SPICE, hors du dossier du projet."""
    cache_root = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(cache_root) / "SOL" / "spice-kernels"


def _download_spice_kernel(relative_path, minimum_bytes, cache_dir):
    """Télécharge atomiquement un kernel générique depuis le serveur NAIF."""
    url = f"{SPICE_KERNEL_BASE_URL}/{relative_path}"
    destination = cache_dir / Path(relative_path).name
    temporary_path = None
    try:
        request = Request(url, headers={"User-Agent": "SOL-Python/1.0"})
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary_file.write(chunk)
        if temporary_path.stat().st_size < minimum_bytes:
            raise OSError("le fichier reçu est incomplet")
        temporary_path.replace(destination)
        print(f"Kernel SPICE téléchargé : {destination}")
        return destination
    except OSError as error:
        raise RuntimeError(
            f"Impossible de récupérer le kernel SPICE {relative_path} depuis NAIF. "
            f"Vérifiez la connexion Internet ou placez le fichier dans {cache_dir}."
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _ensure_spice_kernels():
    """Charge DE440s et les kernels de temps et d'orientation lunaire."""
    global _SPICE_KERNELS_LOADED
    if _SPICE_KERNELS_LOADED:
        return

    cache_dir = _spice_kernel_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, minimum_bytes in SPICE_KERNELS:
        kernel_path = cache_dir / Path(relative_path).name
        if not kernel_path.is_file():
            kernel_path = _download_spice_kernel(
                relative_path,
                minimum_bytes,
                cache_dir,
            )
        kernel_key = str(kernel_path.resolve())
        if kernel_key in _SPICE_LOADED_KERNEL_PATHS:
            continue
        try:
            spice.furnsh(kernel_key)
        except SpiceyError as error:
            raise RuntimeError(
                f"Le kernel SPICE {kernel_path} ne peut pas être chargé. "
                "Supprimez-le du cache pour qu'il soit téléchargé de nouveau."
            ) from error
        _SPICE_LOADED_KERNEL_PATHS.add(kernel_key)

    if spice.namfrm("MOON_ME") == 0:
        raise RuntimeError(
            "Les kernels chargés ne définissent pas le repère lunaire MOON_ME."
        )
    _SPICE_KERNELS_LOADED = True


def _spice_et(epoch):
    """Convertit une date UTC Python ou textuelle en secondes SPICE (ET)."""
    _ensure_spice_kernels()
    if isinstance(epoch, datetime):
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        epoch_text = epoch.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f UTC"
        )
    else:
        epoch_text = str(epoch).strip()
        iso_text = (
            epoch_text[:-1] + "+00:00"
            if epoch_text.endswith("Z")
            else epoch_text
        )
        try:
            parsed_epoch = datetime.fromisoformat(iso_text)
        except ValueError:
            parsed_epoch = None
        if parsed_epoch is not None:
            if parsed_epoch.tzinfo is None:
                parsed_epoch = parsed_epoch.replace(tzinfo=timezone.utc)
            epoch_text = parsed_epoch.astimezone(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S.%f UTC"
            )
        else:
            epoch_text = epoch_text.replace("T", " ")
            if not re.search(
                r"(?:UTC|[+-]\d{2}:?\d{2})$",
                epoch_text,
                re.IGNORECASE,
            ):
                epoch_text += " UTC"
    try:
        return float(spice.str2et(epoch_text))
    except SpiceyError as error:
        raise ValueError(f"Date UTC invalide pour SPICE : {epoch!r}") from error


def _spice_coverage_error(operation, epoch, error):
    return ValueError(
        f"Calcul SPICE impossible pour {epoch!r} ({operation}). La date est peut-être "
        "hors de la couverture des kernels chargés; vérifiez la couverture DE440s "
        f"et du kernel d'orientation lunaire. Détail SPICE : {error}"
    )


def _spk_position_j2000(target, observer, et):
    """Retourne un vecteur géométrique en J2000, en km, via DE440s."""
    _ensure_spice_kernels()
    try:
        position, _ = spice.spkpos(target, float(et), "J2000", "NONE", observer)
    except SpiceyError as error:
        raise _spice_coverage_error(
            f"position {target} relative à {observer}", et, error
        ) from error
    return np.asarray(position, dtype=float)


def _spk_position_series_j2000(target, observer, et_values):
    """Calcule une série de vecteurs géométriques J2000 en kilomètres."""
    _ensure_spice_kernels()
    positions = []
    try:
        for et in np.asarray(et_values, dtype=float):
            position, _ = spice.spkpos(
                target,
                float(et),
                "J2000",
                "NONE",
                observer,
            )
            positions.append(position)
    except SpiceyError as error:
        raise _spice_coverage_error(
            f"trajectoire de {target} relative à {observer}", et, error
        ) from error
    return np.asarray(positions, dtype=float)


def _moon_orientation_matrix(epoch):
    """Rotation du repère lunaire MOON_ME vers J2000 à la date UTC donnée."""
    et = _spice_et(epoch)
    try:
        return np.asarray(spice.pxform("MOON_ME", "J2000", et), dtype=float)
    except SpiceyError as error:
        raise _spice_coverage_error("orientation de la Lune", epoch, error) from error


def _moon_link_dash_points(position):
    """Crée les extrémités des tirets blancs entre la Terre et la Lune."""
    position = np.asarray(position, dtype=float)
    dash_index = np.arange(MOON_LINK_DASH_COUNT, dtype=float)
    dash_starts = (
        dash_index + (1.0 - MOON_LINK_DASH_FRACTION) / 2.0
    ) / MOON_LINK_DASH_COUNT
    dash_ends = dash_starts + MOON_LINK_DASH_FRACTION / MOON_LINK_DASH_COUNT
    return np.stack(
        (dash_starts[:, np.newaxis] * position,
         dash_ends[:, np.newaxis] * position),
        axis=1,
    ).reshape(-1, 3)


def earth_elevation_moon(
    date_utc: str,
    lat_deg: float,
    lon_deg: float,
    alt_km: float = 0.0,
    moon_radius_km: float = 1737.4,
    frame: str = "MOON_ME",
) -> float:
    """
    Calcule l'élévation géométrique du centre de la Terre
    par rapport à l'horizon local d'un site lunaire.

    Paramètres
    ----------
    date_utc : str
        Date UTC, ex. "2026-10-23T12:00:00".
    lat_deg : float
        Latitude sélénocentrique [-90, 90], nord positif.
    lon_deg : float
        Longitude est-positive, en degrés.
    alt_km : float
        Altitude au-dessus de la sphère de référence (km).
    moon_radius_km : float
        Rayon de la sphère lunaire (km).
    frame : str
        Repère lunaire fixé au corps, par défaut MOON_ME.

    Retour
    ------
    float
        Élévation du centre de la Terre en degrés.
        0° = horizon local
        90° = zénith
        -90° = nadir

    Notes
    -----
    Charge au premier appel les kernels NAIF requis dans le cache utilisateur.
    Le calcul est géométrique (sans aberration lumineuse).
    L'horizon est celui d'une sphère, sans relief.
    """

    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError("Latitude hors de [-90, 90].")
    if moon_radius_km + alt_km <= 0:
        raise ValueError("Rayon local invalide.")

    et = _spice_et(date_utc)
    # Position géométrique du centre de la Terre
    # par rapport au centre de la Lune (km)
    try:
        r_earth, _ = spice.spkpos("EARTH", et, frame, "NONE", "MOON")
    except SpiceyError as error:
        raise _spice_coverage_error(
            "position de la Terre vue de la Lune",
            date_utc,
            error,
        ) from error
    return _earth_elevation_from_vector(
        r_earth,
        lat_deg,
        lon_deg,
        alt_km=alt_km,
        moon_radius_km=moon_radius_km,
    )


def _earth_elevation_from_vector(
    earth_vector_km,
    lat_deg: float,
    lon_deg: float,
    alt_km: float = 0.0,
    moon_radius_km: float = 1737.4,
) -> float | np.ndarray:
    """Calcule l'élévation terrestre depuis un ou plusieurs vecteurs relatifs à la Lune."""
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError("Latitude hors de [-90, 90].")
    if moon_radius_km + alt_km <= 0:
        raise ValueError("Rayon local invalide.")

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    up = np.array(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )
    site_position_km = (moon_radius_km + alt_km) * up

    # Terre vue depuis le site lunaire
    los = np.asarray(earth_vector_km, dtype=float) - site_position_km
    los /= np.linalg.norm(los, axis=-1, keepdims=True)

    # Projection sur la verticale locale
    sin_elevation = np.clip(np.sum(los * up, axis=-1), -1.0, 1.0)
    elevation = np.rad2deg(np.arcsin(sin_elevation))
    if np.ndim(elevation) == 0:
        return float(elevation)
    return np.asarray(elevation, dtype=float)


def _earth_elevation_moon_j2000(
    moon_position_j2000,
    epoch,
    lat_deg: float,
    lon_deg: float,
    alt_km: float = 0.0,
    moon_radius_km: float = 1737.4,
) -> float:
    """Évalue l'élévation terrestre avec l'orientation MOON_ME de la date."""
    moon_position_j2000 = np.asarray(moon_position_j2000, dtype=float)
    moon_rotation = _moon_orientation_matrix(epoch)
    earth_vector_local = moon_rotation.T @ (-moon_position_j2000)
    return _earth_elevation_from_vector(
        earth_vector_local,
        lat_deg,
        lon_deg,
        alt_km=alt_km,
        moon_radius_km=moon_radius_km,
    )


def _earth_elevation_moon_series_j2000(
    moon_positions_j2000,
    moon_rotations,
    lat_deg: float,
    lon_deg: float,
    alt_km: float = 0.0,
    moon_radius_km: float = 1737.4,
) -> np.ndarray:
    """Calcule l'élévation d'un site avec l'orientation réelle à chaque date."""
    positions = np.asarray(moon_positions_j2000, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Les positions lunaires doivent avoir la forme (n, 3).")
    rotations = np.asarray(moon_rotations, dtype=float)
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("Les rotations lunaires doivent avoir la forme (n, 3, 3).")

    earth_vectors_local = np.einsum("nji,nj->ni", rotations, -positions)
    return _earth_elevation_from_vector(
        earth_vectors_local,
        lat_deg,
        lon_deg,
        alt_km=alt_km,
        moon_radius_km=moon_radius_km,
    )


def _solar_illumination_moon_j2000(
    moon_position_j2000,
    sun_position_j2000,
    epoch,
    lat_deg: float,
    lon_deg: float,
    moon_radius_km: float = 1737.4,
) -> tuple[float, float]:
    """Retourne l'élévation solaire et l'éclairement horizontal relatif (%)"""
    moon_position = np.asarray(moon_position_j2000, dtype=float)
    sun_position = np.asarray(sun_position_j2000, dtype=float)
    moon_rotation = _moon_orientation_matrix(epoch)
    sun_vector_local = moon_rotation.T @ (sun_position - moon_position)
    sun_elevation_deg = _earth_elevation_from_vector(
        sun_vector_local,
        lat_deg,
        lon_deg,
        moon_radius_km=moon_radius_km,
    )
    sun_distance_km = np.linalg.norm(sun_position - moon_position)
    direct_illumination_percent = (
        100.0
        * (Earth.AU / sun_distance_km) ** 2
        * max(0.0, np.sin(np.deg2rad(sun_elevation_deg)))
    )
    return sun_elevation_deg, float(direct_illumination_percent)


def _solar_illumination_moon_series_j2000(
    moon_positions_j2000,
    sun_positions_j2000,
    moon_rotations,
    lat_deg: float,
    lon_deg: float,
    moon_radius_km: float = 1737.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Évalue l'élévation et l'éclairement direct cohérents sur une série."""
    moon_positions = np.asarray(moon_positions_j2000, dtype=float)
    sun_positions = np.asarray(sun_positions_j2000, dtype=float)
    rotations = np.asarray(moon_rotations, dtype=float)
    if (
        moon_positions.ndim != 2
        or moon_positions.shape[1] != 3
        or sun_positions.shape != moon_positions.shape
        or rotations.shape != (len(moon_positions), 3, 3)
    ):
        raise ValueError(
            "Les positions et rotations doivent avoir les formes (n, 3) et (n, 3, 3)."
        )

    sun_vectors = sun_positions - moon_positions
    sun_vectors_local = np.einsum("nji,nj->ni", rotations, sun_vectors)
    sun_elevation_deg = _earth_elevation_from_vector(
        sun_vectors_local,
        lat_deg,
        lon_deg,
        moon_radius_km=moon_radius_km,
    )
    sun_distance_km = np.linalg.norm(sun_vectors, axis=1)
    direct_illumination_percent = (
        100.0
        * (Earth.AU / sun_distance_km) ** 2
        * np.maximum(0.0, np.sin(np.deg2rad(sun_elevation_deg)))
    )
    return sun_elevation_deg, direct_illumination_percent


def _earth_surface_position_km(latitude_deg, longitude_deg, equatorial_radius_km, polar_radius_km):
    """Convertit une position géodésique en coordonnées cartésiennes WGS-84."""
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    eccentricity_squared = 1.0 - (polar_radius_km / equatorial_radius_km) ** 2
    prime_vertical_radius = equatorial_radius_km / np.sqrt(
        1.0 - eccentricity_squared * np.sin(latitude) ** 2
    )
    return np.array(
        [
            prime_vertical_radius * np.cos(latitude) * np.cos(longitude),
            prime_vertical_radius * np.cos(latitude) * np.sin(longitude),
            prime_vertical_radius
            * (1.0 - eccentricity_squared)
            * np.sin(latitude),
        ],
        dtype=float,
    )


def _earth_rotation_matrix_series_j2000(et_values):
    """Retourne les rotations terrestres IAU_EARTH vers J2000 pour une série."""
    _ensure_spice_kernels()
    rotations = []
    try:
        for et in np.asarray(et_values, dtype=float):
            rotations.append(spice.pxform("IAU_EARTH", "J2000", float(et)))
    except SpiceyError as error:
        raise _spice_coverage_error("orientation de la Terre", et, error) from error
    return np.asarray(rotations, dtype=float)


def _earth_fixed_vector_series_j2000(vector_fixed, earth_rotations):
    """Transforme une position ou normale terrestre fixe en coordonnées J2000."""
    vector_fixed = np.asarray(vector_fixed, dtype=float)
    rotations = np.asarray(earth_rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("Les rotations terrestres doivent avoir la forme (n, 3, 3).")
    return np.einsum("nij,j->ni", rotations, vector_fixed)


def _dsn_station_geometry(track_ets):
    """Retourne positions et normales DSN dans le même repère J2000 que la Lune."""
    earth_rotations = _earth_rotation_matrix_series_j2000(track_ets)
    station_geometry = {}
    for station in DSN_STATIONS:
        latitude = np.deg2rad(station["latitude_deg"])
        longitude = np.deg2rad(station["longitude_deg"])
        fixed_position = _earth_surface_position_km(
            station["latitude_deg"],
            station["longitude_deg"],
            Earth.r1_km,
            Earth.r2_km,
        )
        fixed_normal = np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ],
            dtype=float,
        )
        station_geometry[station["name"]] = {
            "positions": _earth_fixed_vector_series_j2000(
                fixed_position,
                earth_rotations,
            ),
            "normals": _earth_fixed_vector_series_j2000(
                fixed_normal,
                earth_rotations,
            ),
        }
    return station_geometry


def _moon_orientation_matrix_series(et_values):
    """Construit les rotations MOON_ME vers J2000 pour une série de temps ET."""
    _ensure_spice_kernels()
    rotations = []
    try:
        for et in np.asarray(et_values, dtype=float):
            rotations.append(spice.pxform("MOON_ME", "J2000", float(et)))
    except SpiceyError as error:
        raise _spice_coverage_error("orientation lunaire de la série", et, error) from error
    return np.asarray(rotations, dtype=float)


def _dsn_visibility_for_site_series(
    moon_positions_j2000,
    moon_rotations,
    latitude_deg,
    longitude_deg,
    station_geometry,
    moon_radius_km,
):
    """Choisit la meilleure élévation DSN avec visibilité aux deux horizons."""
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    moon_up_local = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        dtype=float,
    )
    moon_up_j2000 = np.einsum("nij,j->ni", moon_rotations, moon_up_local)
    moon_site_positions = moon_positions_j2000 + moon_up_j2000 * moon_radius_km

    station_elevations = []
    for station in DSN_STATIONS:
        station_state = station_geometry[station["name"]]
        line_of_sight = station_state["positions"] - moon_site_positions
        distance = np.linalg.norm(line_of_sight, axis=1)
        line_of_sight_unit = line_of_sight / np.maximum(
            distance[:, np.newaxis], 1.0e-12
        )
        lunar_elevation = np.rad2deg(
            np.arcsin(
                np.clip(
                    np.sum(line_of_sight_unit * moon_up_j2000, axis=1),
                    -1.0,
                    1.0,
                )
            )
        )
        station_elevation = np.rad2deg(
            np.arcsin(
                np.clip(
                    np.sum(
                        -line_of_sight_unit * station_state["normals"],
                        axis=1,
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        visible_from_both = (lunar_elevation > 0.0) & (station_elevation > 0.0)
        station_elevations.append(
            np.where(visible_from_both, lunar_elevation, np.nan)
        )

    station_elevations = np.column_stack(station_elevations)
    visible = np.isfinite(station_elevations)
    best_station_indices = np.argmax(
        np.where(visible, station_elevations, -np.inf),
        axis=1,
    )
    has_visibility = np.any(visible, axis=1)
    best_elevations = station_elevations[
        np.arange(len(station_elevations)),
        best_station_indices,
    ]
    best_elevations[~has_visibility] = np.nan
    best_station_indices[~has_visibility] = -1
    return best_elevations, best_station_indices, station_elevations


def _interactive_chart_window(initial_chart, chart_updates, window_state, lock):
    """Affiche les graphiques séparément avec zoom et panoramique."""
    root = None
    try:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )

        root = tk.Tk()
        root.title("Élévation de la Terre depuis les sites lunaires")
        root.geometry("1400x900")
        root.minsize(900, 600)
        container = tk.Frame(root, background="#101722")
        container.pack(fill=tk.BOTH, expand=True)
        chart_controls = tk.Frame(container, background="#101722")
        chart_controls.pack(side=tk.TOP, fill=tk.X)
        hint = tk.Label(
            container,
            text=(
                "Zoom : choisir l'outil Zoom et encadrer une zone. "
                "Pan : déplacer la vue. Home : réinitialiser."
            ),
            background="#101722",
            foreground="#B9C8D8",
            anchor="w",
            padx=10,
            pady=4,
        )
        hint.pack(side=tk.BOTTOM, fill=tk.X)
        canvas = None
        toolbar = None
        current_chart = {"bundle": None}

        def set_site_selection(selected_indices, redraw=True):
            chart_bundle = current_chart["bundle"]
            if chart_bundle is None:
                return
            selected_indices = set(selected_indices)
            chart_bundle["selected_site_indices"] = selected_indices
            for site_index, artists in chart_bundle["site_artists"].items():
                is_selected = site_index in selected_indices
                for artist in artists:
                    artist.set_visible(is_selected)

            station_matrix = chart_bundle["station_matrix"]
            station_rows = chart_bundle["site_indices"]
            row_mask = np.broadcast_to(
                np.asarray(
                    [
                        site_index not in selected_indices
                        for site_index in station_rows
                    ],
                    dtype=bool,
                )[:, None],
                station_matrix.shape,
            )
            chart_bundle["station_mesh"].set_array(
                np.ma.masked_where(row_mask, station_matrix).ravel()
            )
            station_axis = chart_bundle["axes"]["stations"]
            station_axis.set_yticklabels(
                [
                    chart_bundle["site_labels"][row]
                    if site_index in selected_indices
                    else ""
                    for row, site_index in enumerate(station_rows)
                ]
            )

            for axis_name, title in (
                ("visibility", "Sites · accès DSN"),
                ("illumination", "Points lunaires"),
            ):
                axis = chart_bundle["axes"][axis_name]
                handles, labels = axis.get_legend_handles_labels()
                visible_entries = [
                    (handle, label)
                    for handle, label in zip(handles, labels)
                    if handle.get_visible()
                ]
                axis.legend(
                    [entry[0] for entry in visible_entries],
                    [entry[1] for entry in visible_entries],
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=7,
                    title=title,
                )

            if redraw and canvas is not None:
                canvas.draw_idle()

        def open_site_selector():
            chart_bundle = current_chart["bundle"]
            if chart_bundle is None:
                return
            dialog = tk.Toplevel(root)
            dialog.title("Sites affichés dans les graphiques")
            dialog.geometry("480x560")
            dialog.minsize(380, 360)
            dialog.configure(background="#101722")
            dialog.transient(root)
            tk.Label(
                dialog,
                text="Sélectionnez les points lunaires à tracer",
                background="#101722",
                foreground="#F4F8FC",
                font=("Segoe UI", 12, "bold"),
                anchor="w",
                padx=14,
                pady=12,
            ).pack(fill=tk.X)

            list_container = tk.Frame(dialog, background="#101722")
            list_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
            site_canvas = tk.Canvas(
                list_container,
                background="#101722",
                highlightthickness=0,
            )
            scrollbar = tk.Scrollbar(
                list_container,
                orient=tk.VERTICAL,
                command=site_canvas.yview,
            )
            site_list = tk.Frame(site_canvas, background="#101722")
            site_list.bind(
                "<Configure>",
                lambda _event: site_canvas.configure(
                    scrollregion=site_canvas.bbox("all")
                ),
            )
            site_canvas.create_window((0, 0), window=site_list, anchor="nw")
            site_canvas.configure(yscrollcommand=scrollbar.set)
            site_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            variables = {}
            for site_index, label in chart_bundle["site_options"]:
                variable = tk.BooleanVar(
                    master=dialog,
                    value=site_index in chart_bundle["selected_site_indices"],
                )
                variables[site_index] = variable
                tk.Checkbutton(
                    site_list,
                    text=label,
                    variable=variable,
                    background="#101722",
                    foreground="#E5EDF8",
                    activebackground="#101722",
                    activeforeground="#FFFFFF",
                    selectcolor="#243449",
                    anchor="w",
                    padx=8,
                    pady=4,
                ).pack(fill=tk.X, anchor="w")

            def set_all_sites(value):
                for variable in variables.values():
                    variable.set(value)

            quick_actions = tk.Frame(dialog, background="#101722")
            quick_actions.pack(fill=tk.X, padx=12, pady=(0, 8))
            for label, value in (("Tout sélectionner", True), ("Tout retirer", False)):
                tk.Button(
                    quick_actions,
                    text=label,
                    command=lambda value=value: set_all_sites(value),
                    background="#18263A",
                    foreground="#E5EDF8",
                    activebackground="#264158",
                    activeforeground="#FFFFFF",
                    relief=tk.FLAT,
                    padx=10,
                    pady=5,
                ).pack(side=tk.LEFT, padx=(0, 8))

            def apply_selection():
                selected_indices = {
                    site_index
                    for site_index, variable in variables.items()
                    if variable.get()
                }
                set_site_selection(selected_indices)
                on_selection = chart_bundle.get("on_site_selection")
                if on_selection is not None:
                    on_selection(selected_indices)
                dialog.destroy()

            footer = tk.Frame(dialog, background="#101722")
            footer.pack(fill=tk.X, padx=12, pady=(0, 12))
            tk.Button(
                footer,
                text="Appliquer",
                command=apply_selection,
                background="#2C765F",
                foreground="#FFFFFF",
                relief=tk.FLAT,
                padx=14,
                pady=7,
            ).pack(side=tk.RIGHT)
            tk.Button(
                footer,
                text="Annuler",
                command=dialog.destroy,
                background="#253142",
                foreground="#E5EDF8",
                relief=tk.FLAT,
                padx=14,
                pady=7,
            ).pack(side=tk.RIGHT, padx=(8, 0))
            dialog.grab_set()
            root.wait_window(dialog)

        def set_chart_mode(mode):
            chart_bundle = current_chart["bundle"]
            if chart_bundle is None:
                return
            figure = chart_bundle["figure"]
            axes = chart_bundle["axes"]
            if mode == "all":
                for name, axis in axes.items():
                    axis.set_visible(True)
                    axis.set_position(chart_bundle["original_positions"][name])
                    axis.set_xlabel("" if name != "illumination" else "Date UTC")
                    axis.tick_params(axis="x", labelbottom=name == "illumination")
            else:
                for name, axis in axes.items():
                    is_selected = name == mode
                    axis.set_visible(is_selected)
                    if is_selected:
                        axis.set_position((0.09, 0.13, 0.69, 0.76))
                        axis.set_xlabel("Date UTC")
                        axis.tick_params(axis="x", labelbottom=True)
            if canvas is not None:
                canvas.draw_idle()

        for label, mode in (
            ("Tout", "all"),
            ("Visibilité maximale", "visibility"),
            ("Meilleure base", "stations"),
            ("Éclairement", "illumination"),
        ):
            tk.Button(
                chart_controls,
                text=label,
                command=lambda mode=mode: set_chart_mode(mode),
                background="#18263A",
                foreground="#E5EDF8",
                activebackground="#264158",
                activeforeground="#FFFFFF",
                relief=tk.FLAT,
                padx=10,
                pady=5,
            ).pack(side=tk.LEFT, padx=(8, 0), pady=6)

        tk.Button(
            chart_controls,
            text="Choisir les sites",
            command=open_site_selector,
            background="#284337",
            foreground="#E5EDF8",
            activebackground="#35624F",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=10,
            pady=5,
        ).pack(side=tk.RIGHT, padx=8, pady=6)

        def install_chart(chart_bundle):
            nonlocal canvas, toolbar
            current_chart["bundle"] = chart_bundle
            chart_bundle["original_positions"] = {
                name: axis.get_position().frozen()
                for name, axis in chart_bundle["axes"].items()
            }
            if toolbar is not None:
                toolbar.destroy()
            if canvas is not None:
                canvas.get_tk_widget().destroy()
            canvas = FigureCanvasTkAgg(chart_bundle["figure"], master=container)
            toolbar = NavigationToolbar2Tk(
                canvas,
                container,
                pack_toolbar=False,
            )
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(
                side=tk.TOP,
                fill=tk.BOTH,
                expand=True,
            )
            set_site_selection(
                chart_bundle["selected_site_indices"],
                redraw=False,
            )
            canvas.draw()
            set_chart_mode("all")

        def process_figure_updates():
            newest_chart = None
            while True:
                try:
                    newest_chart = chart_updates.get_nowait()
                except Empty:
                    break
            if newest_chart is not None:
                install_chart(newest_chart)
            root.after(100, process_figure_updates)

        install_chart(initial_chart)
        root.after(100, process_figure_updates)
        root.mainloop()
    except Exception as error:
        with lock:
            window_state["error"] = str(error)
        print(f"Fenêtre graphique impossible à ouvrir : {error}")
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        with lock:
            window_state["running"] = False


def _open_or_update_interactive_chart(chart_bundle, window_state):
    """Ouvre le graphique interactif ou transmet une nouvelle figure à la fenêtre."""
    lock = window_state["lock"]
    with lock:
        window_state["error"] = None
        if window_state["running"]:
            window_state["updates"].put(chart_bundle)
            return
        while True:
            try:
                window_state["updates"].get_nowait()
            except Empty:
                break
        window_state["running"] = True
        Thread(
            target=_interactive_chart_window,
            args=(chart_bundle, window_state["updates"], window_state, lock),
            daemon=True,
        ).start()


def _shift_years(epoch, years):
    """Décale une date de plusieurs années, y compris autour du 29 février."""
    try:
        return epoch.replace(year=epoch.year + years)
    except ValueError:
        return epoch.replace(year=epoch.year + years, month=2, day=28)


def _moon_position_j2000(epoch):
    """Retourne la position géométrique de la Lune par rapport à la Terre [km]."""
    return _spk_position_j2000("MOON", "EARTH", _spice_et(epoch))


def _sun_position_j2000(epoch):
    """Retourne la position géométrique du Soleil par rapport à la Terre [km]."""
    return _spk_position_j2000("SUN", "EARTH", _spice_et(epoch))


def _format_km(value, decimals=0):
    """Formate une distance avec un séparateur de milliers français."""
    return f"{value:,.{decimals}f}".replace(",", "\u202f")


def _calculate_track(start, end, reference_date):
    """Calcule la trajectoire quotidienne en J2000 avec DE440s."""
    day_count = (end - start).days
    sample_dates = [start + timedelta(days=day) for day in range(day_count + 1)]
    et_values = np.asarray([_spice_et(date) for date in sample_dates], dtype=float)

    print(
        f"Calcul de la trajectoire lunaire SPICE/DE440s : {len(sample_dates)} dates "
        f"du {start:%Y-%m-%d} au {end:%Y-%m-%d}..."
    )
    positions = _spk_position_series_j2000("MOON", "EARTH", et_values)

    reference_index = (reference_date - start).days
    return positions, et_values, reference_index


def create_scene():
    """Construit la fenêtre PyVista et son curseur temporel de ±20 ans."""
    earth_texture_path = IMAGE_DIR / "World with Ice (4000x2000).jpg"
    # Dérivée 4K 16 bits de la carte LROC, plus adaptée à une texture GPU.
    moon_texture_path = IMAGE_DIR / "lroc_color_16bit_srgb_4k.png"
    starfield_path = IMAGE_DIR / "starmap_16k.jpg"
    for texture_path in (earth_texture_path, moon_texture_path, starfield_path):
        if not texture_path.is_file():
            raise FileNotFoundError(f"Image introuvable : {texture_path}")

    reference_date = datetime.now(timezone.utc).replace(microsecond=0)
    start_date = _shift_years(reference_date, -20)
    end_date = _shift_years(reference_date, 20)
    track_positions, track_ets, reference_index = _calculate_track(
        start_date,
        end_date,
        reference_date,
    )
    track_dates = [
        start_date + timedelta(days=day)
        for day in range(len(track_positions))
    ]

    history = pv.lines_from_points(
        track_positions[: reference_index + 1],
        close=False,
    )
    future = pv.lines_from_points(
        track_positions[reference_index:],
        close=False,
    )

    initial_sun_position = _sun_position_j2000(reference_date)
    plotter = pv.Plotter(window_size=WINDOW_SIZE, lighting="none")
    plotter.set_background("#050A16")
    sky_mesh = pv.Sphere(
        radius=SKY_SPHERE_RADIUS_KM,
        theta_resolution=256,
        phi_resolution=128,
        texture_coordinates=True,
    )
    starfield_actor = plotter.add_mesh(
        sky_mesh,
        texture=pv.read_texture(str(starfield_path)),
        name="Sphère céleste",
        lighting=False,
        backface_culling=False,
        pickable=False,
    )
    starfield_actor.orientation = (0.0, 0.0, ComputeGMST(reference_date))
    plotter.renderer.SetErase(True)
    plotter.render_window.SetErase(True)
    view_direction = np.array([1.0, -1.7, 1.1])
    view_direction /= np.linalg.norm(view_direction)

    history_actor = plotter.add_mesh(
        history,
        name="Passé (DE440s)",
        color="#58C4DD",
        line_width=1.5,
        opacity=0.85,
        lighting=False,
    )
    future_actor = plotter.add_mesh(
        future,
        name="Futur (DE440s)",
        color="#F3B45B",
        line_width=1.5,
        opacity=0.85,
        lighting=False,
    )

    # La texture terrestre suit GMST pour l'affichage; les calculs utilisent J2000/SPICE.
    earth_mesh = geodetic_ellipsoid_mesh(Earth.r1_km, Earth.r2_km)
    earth_mesh.points *= DISPLAY_RADIUS_SCALE
    earth_actor = plotter.add_mesh(
        earth_mesh,
        texture=pv.read_texture(str(earth_texture_path)),
        name="Terre",
        smooth_shading=True,
        ambient=0.25,
        diffuse=0.85,
        specular=0.05,
    )
    earth_station_actors = {}
    earth_station_positions_fixed = {}
    initial_sidereal_angle = ComputeGMST(reference_date)
    initial_earth_rotation = _earth_rotation_matrix_series_j2000(
        [track_ets[reference_index]]
    )
    for station in DSN_STATIONS:
        station_marker_mesh = pv.Sphere(
            radius=Earth.r1_km * DISPLAY_RADIUS_SCALE * 0.035,
            theta_resolution=24,
            phi_resolution=16,
        )
        station_actor = plotter.add_mesh(
            station_marker_mesh,
            name=f"Station DSN {station['name']}",
            color=station["color"],
            lighting=False,
            pickable=False,
            render=False,
        )
        station_position = _earth_surface_position_km(
            station["latitude_deg"],
            station["longitude_deg"],
            Earth.r1_km,
            Earth.r2_km,
        )
        station_actor.position = tuple(
            _earth_fixed_vector_series_j2000(
                station_position,
                initial_earth_rotation,
            )[0]
            * DISPLAY_RADIUS_SCALE
        )
        earth_station_actors[station["name"]] = station_actor
        earth_station_positions_fixed[station["name"]] = station_position

    moon_radius = Moon.r_mean_km * DISPLAY_RADIUS_SCALE
    moon_mesh = pv.Sphere(
        radius=moon_radius,
        center=(0.0, 0.0, 0.0),
        theta_resolution=120,
        phi_resolution=60,
    )
    local_points = moon_mesh.points.copy()
    local_radius = np.linalg.norm(local_points, axis=1)
    local_normals = local_points / local_radius[:, np.newaxis]
    u_coord = 0.5 + (
        np.arctan2(-local_points[:, 0], local_points[:, 1]) + np.pi / 2
    ) / (2 * np.pi)
    v_coord = 0.5 + np.arcsin(local_points[:, 2] / local_radius) / np.pi
    moon_mesh.active_texture_coordinates = np.column_stack((u_coord, v_coord))
    # La carte LROC centre le côté moyen tourné vers la Terre en (u, v) = (0.5, 0.5).
    # En MOON_ME, ce côté est l'axe +X; +Y reste l'est et +Z le nord sélénographiques.
    initial_position = track_positions[reference_index]
    initial_rotation = _moon_orientation_matrix(reference_date)
    moon_mesh.points = local_points @ initial_rotation.T + initial_position
    moon_mesh.point_data["Normals"] = local_normals @ initial_rotation.T
    moon_mesh.point_data.active_normals_name = "Normals"
    moon_actor = plotter.add_mesh(
        moon_mesh,
        texture=pv.read_texture(str(moon_texture_path)),
        name="Lune",
        ambient=0.04,
        diffuse=1.0,
        specular=0.0,
    )
    moon_site_halo_actors = []
    moon_site_halo_sources = []
    moon_site_actors = []
    moon_site_sources = []
    moon_site_number_actors = []
    moon_link_lines = np.column_stack(
        (
            np.full(MOON_LINK_DASH_COUNT, 2, dtype=np.int64),
            np.arange(2 * MOON_LINK_DASH_COUNT, dtype=np.int64).reshape(-1, 2),
        )
    ).ravel()
    moon_link_mesh = pv.PolyData(
        _moon_link_dash_points(initial_position),
        lines=moon_link_lines,
    )
    plotter.add_mesh(
        moon_link_mesh,
        name="Ligne pointillée Terre–Lune",
        color="#FFFFFF",
        line_width=3.0,
        render_lines_as_tubes=True,
        lighting=False,
        opacity=1.0,
        pickable=False,
    )

    initial_sun_direction = initial_sun_position / np.linalg.norm(initial_sun_position)
    initial_sun_display_position = initial_sun_direction * SUN_VISUAL_DISTANCE_KM
    sun_mesh = pv.Sphere(
        radius=Earth.r1_km * DISPLAY_RADIUS_SCALE * 0.65,
        center=(0.0, 0.0, 0.0),
        theta_resolution=48,
        phi_resolution=32,
    )
    sun_actor = plotter.add_mesh(
        sun_mesh,
        name="Soleil (direction réelle)",
        color="#FFC857",
        lighting=False,
    )
    sun_actor.position = tuple(initial_sun_display_position)
    sun_halo = pv.Sphere(
        radius=Earth.r1_km * DISPLAY_RADIUS_SCALE * 0.95,
        center=(0.0, 0.0, 0.0),
        theta_resolution=32,
        phi_resolution=24,
    )
    sun_halo_actor = plotter.add_mesh(
        sun_halo,
        name="Halo du Soleil",
        color="#F6B93B",
        opacity=0.16,
        lighting=False,
    )
    sun_halo_actor.position = tuple(initial_sun_display_position)

    sun_ray_mesh = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], initial_sun_display_position]),
        lines=np.array([2, 0, 1]),
    )
    plotter.add_mesh(
        sun_ray_mesh,
        name="Direction du Soleil",
        color="#F4C95D",
        line_width=1.7,
        opacity=0.55,
        lighting=False,
    )

    sun_light = pv.Light(
        light_type="scene light",
        position=tuple(initial_sun_position),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=0.9,
        cone_angle=180.0,
    )
    # Source ponctuelle placée au Soleil réel : Terre et Lune reçoivent chacune
    # la lumière selon leur propre vecteur surface→Soleil.
    sun_light.positional = True
    sun_light.attenuation_values = (1.0, 0.0, 0.0)
    plotter.add_light(sun_light)

    plotter.add_text(
        "TERRE · LUNE  |  POSITION GÉOCENTRIQUE",
        position="upper_left",
        font_size=15,
        color="white",
        font="arial",
    )
    plotter.add_text(
        f"Passé — DE440s  {start_date:%Y-%m-%d} → {reference_date:%Y-%m-%d}",
        position=(0.72, 0.95),
        font_size=9,
        color="#58C4DD",
        font="arial",
        viewport=True,
    )
    plotter.add_text(
        f"Futur — DE440s  {reference_date:%Y-%m-%d} → {end_date:%Y-%m-%d}",
        position=(0.72, 0.92),
        font_size=9,
        color="#F3B45B",
        font="arial",
        viewport=True,
    )
    status_actor = plotter.add_text(
        "",
        position=(0.02, 0.88),
        font_size=9,
        color="#E5EDF8",
        font="courier",
        viewport=True,
    )
    date_entry_actor = plotter.add_text(
        "",
        position=(0.02, 0.28),
        font_size=9,
        color="#58C4DD",
        font="courier",
        viewport=True,
    )
    date_entry_actor.SetVisibility(False)
    sun_status_actor = plotter.add_text(
        "",
        position=(0.72, 0.72),
        font_size=8,
        color="#FFE3A3",
        font="courier",
        viewport=True,
    )
    plotter.add_text(
        "Bases DSN",
        position=(0.72, 0.43),
        font_size=8,
        color="#B9C8D8",
        font="arial",
        viewport=True,
    )
    for station_index, station in enumerate(DSN_STATIONS):
        plotter.add_text(
            f"{station['name']}  ·  {station['antenna']}",
            position=(0.72, 0.405 - station_index * 0.027),
            font_size=8,
            color=station["color"],
            font="arial",
            viewport=True,
        )
    orbit_toggle_position = (22, 475)
    orbit_toggle_size = 32
    orbit_label_font_size = 8
    orbit_label_position = (
        orbit_toggle_position[0] + orbit_toggle_size + 12,
        orbit_toggle_position[1]
        + (orbit_toggle_size - orbit_label_font_size) // 2
        - 1,
    )
    orbit_status_actor = plotter.add_text(
        "  Orbite lunaire · affichée  ",
        position=orbit_label_position,
        font_size=orbit_label_font_size,
        color="#B9C8DF",
        font="arial",
    )
    plotter.add_text(
        "SAISIR DATE : cliquer le bouton ou son libellé, puis entrer la date UTC\n"
        "Pointillés : Terre–Lune · Curseur : date UTC · Bouton vert : lecture/pause\n"
        "Clic gauche sur la Lune : ajouter un site (requis pour le graphique externe)\n"
        f"Rayons ×{DISPLAY_RADIUS_SCALE:g} · Soleil : direction réelle, distance comprimée",
        position=(0.02, 0.18),
        font_size=8,
        color="#B9C8DF",
        font="arial",
        viewport=True,
    )

    camera_distance = Moon.a * 7.0
    plotter.camera_position = [
        camera_distance * view_direction,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.show_axes()

    simulation = {
        "date": reference_date,
        "playing": False,
        "timer_started": False,
        "days_per_second": DEFAULT_SIMULATION_DAYS_PER_SECOND,
        "moon_position": np.array(initial_position, dtype=float),
        "sun_position": np.array(initial_sun_position, dtype=float),
        "camera_focus": "earth",
        "moon_sites": [],
        "chart_site_indices": None,
    }
    moon_sites_status_actor = plotter.add_text(
        "",
        position=(0.72, 0.56),
        font_size=8,
        color="#FFE600",
        font="courier",
        viewport=True,
    )
    satellite_states = {}
    for satellite_key, satellite in CISLUNAR_SATELLITES.items():
        marker_mesh = pv.Sphere(
            radius=2_100.0,
            theta_resolution=24,
            phi_resolution=16,
        )
        marker_actor = plotter.add_mesh(
            marker_mesh,
            name=f"Position {satellite['name']}",
            color=satellite["color"],
            lighting=False,
            pickable=False,
            render=False,
        )
        marker_actor.visibility = False
        satellite_states[satellite_key] = {
            "enabled": False,
            "loading": False,
            "generation": 0,
            "coverage": None,
            "desired_date": None,
            "retry_after": 0.0,
            "track_start": None,
            "track_stop": None,
            "times": None,
            "positions": None,
            "track_actor": None,
            "track_mesh": None,
            "marker_actor": marker_actor,
            "error": None,
        }
    satellite_panel_heading = plotter.add_text(
        "Satellites cis-lunaires",
        position=(22, 441),
        font_size=10,
        color="#F4F8FC",
        font="arial",
    )
    satellite_panel_heading.GetTextProperty().SetBold(True)
    satellite_panel_heading.GetTextProperty().SetBackgroundColor(
        pv.Color("#132238").float_rgb
    )
    satellite_panel_heading.GetTextProperty().SetFrameColor(
        pv.Color("#52677D").float_rgb
    )
    satellite_panel_heading.GetTextProperty().SetBackgroundOpacity(0.96)
    satellite_panel_heading.GetTextProperty().SetFrame(True)
    satellite_label_actors = {
        "lro": plotter.add_text(
            f"{CISLUNAR_SATELLITES['lro']['name']}  ·  "
            f"{CISLUNAR_SATELLITES['lro']['description']}",
            position=(60, 399),
            font_size=9,
            color=CISLUNAR_SATELLITES["lro"]["color"],
            font="arial",
        ),
        "capstone": plotter.add_text(
            f"{CISLUNAR_SATELLITES['capstone']['name']}  ·  "
            f"{CISLUNAR_SATELLITES['capstone']['description']}",
            position=(60, 351),
            font_size=9,
            color=CISLUNAR_SATELLITES["capstone"]["color"],
            font="arial",
        ),
    }
    for satellite_key, actor in satellite_label_actors.items():
        text_property = actor.GetTextProperty()
        text_property.SetBold(True)
        text_property.SetBackgroundColor(pv.Color("#111B2C").float_rgb)
        text_property.SetFrameColor(
            pv.Color(CISLUNAR_SATELLITES[satellite_key]["color"]).float_rgb
        )
        text_property.SetBackgroundOpacity(0.94)
        text_property.SetFrame(True)
    satellite_status_actor = plotter.add_text(
        "Sélectionnez un orbiteur pour charger sa trajectoire.",
        position=(60, 320),
        font_size=8,
        color="#B9C8D8",
        font="arial",
    )
    satellite_status_actor.GetTextProperty().SetBackgroundColor(
        pv.Color("#111B2C").float_rgb
    )
    satellite_status_actor.GetTextProperty().SetFrameColor(
        pv.Color("#52677D").float_rgb
    )
    satellite_status_actor.GetTextProperty().SetBackgroundOpacity(0.94)
    satellite_status_actor.GetTextProperty().SetFrame(True)
    satellite_panel_heading.SetVisibility(False)
    satellite_status_actor.SetVisibility(False)
    for actor in satellite_label_actors.values():
        actor.SetVisibility(False)
    focus_title_actors = {}
    satellite_checkbox_widgets = {}
    satellite_tab_title_actor = None
    satellite_panel_is_open = False
    horizons_poll_timer_started = False
    horizons_results = Queue()
    time_slider = None

    def create_site_marker_overlay(radius, color):
        """Crée une pastille circulaire en coordonnées écran."""
        source = vtkRegularPolygonSource()
        source.SetNumberOfSides(32)
        source.SetRadius(radius)
        source.SetCenter(0.0, 0.0, 0.0)
        source.SetGeneratePolygon(True)
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(pv.Color(color).float_rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.PickableOff()
        actor.VisibilityOff()
        plotter.renderer.add_actor(
            actor,
            reset_camera=False,
            pickable=False,
            render=False,
        )
        return actor, source

    def update_moon_site_elevations(moon_position):
        """Actualise la meilleure liaison DSN de chaque site à cette date."""
        moon_positions = np.asarray([moon_position], dtype=float)
        et = _spice_et(simulation["date"])
        station_geometry = _dsn_station_geometry([et])
        moon_rotations = _moon_orientation_matrix_series([et])
        for site in simulation["moon_sites"]:
            elevations, station_indices, _ = _dsn_visibility_for_site_series(
                moon_positions,
                moon_rotations,
                site["latitude_deg"],
                site["longitude_deg"],
                station_geometry,
                moon_radius_km=Moon.r_mean_km,
            )
            station_index = int(station_indices[0])
            site["best_station"] = (
                DSN_STATIONS[station_index]["name"]
                if station_index >= 0
                else None
            )
            site["max_visibility_deg"] = (
                float(elevations[0]) if station_index >= 0 else None
            )

    def update_moon_site_illumination(moon_position, sun_position, epoch):
        """Actualise l'élévation solaire et l'éclairement direct des sites."""
        for site in simulation["moon_sites"]:
            sun_elevation, illumination_percent = _solar_illumination_moon_j2000(
                moon_position,
                sun_position,
                epoch,
                site["latitude_deg"],
                site["longitude_deg"],
                moon_radius_km=Moon.r_mean_km,
            )
            site["sun_elevation_deg"] = sun_elevation
            site["solar_illumination_percent"] = illumination_percent

    def refresh_moon_sites():
        """Redessine les sites lunaires à partir de leurs coordonnées fixes."""
        sites = simulation["moon_sites"]
        if not sites:
            for actor in (
                *moon_site_halo_actors,
                *moon_site_actors,
                *moon_site_number_actors,
            ):
                actor.SetVisibility(False)
            moon_sites_status_actor.SetInput("")
            return

        latitudes = np.deg2rad([site["latitude_deg"] for site in sites])
        longitudes = np.deg2rad([site["longitude_deg"] for site in sites])
        local_directions = np.column_stack(
            (
                np.cos(latitudes) * np.cos(longitudes),
                np.cos(latitudes) * np.sin(longitudes),
                np.sin(latitudes),
            )
        )
        for index in range(len(sites) - len(moon_site_actors)):
            halo_actor, halo_source = create_site_marker_overlay(15.0, "#07101A")
            marker_actor, marker_source = create_site_marker_overlay(12.0, "#FFE600")
            number_actor = plotter.add_text(
                str(len(moon_site_actors) + 1),
                position=(0.0, 0.0),
                font_size=12,
                color="#17202B",
                font="arial",
                viewport=True,
                render=False,
            )
            number_text = number_actor.GetTextProperty()
            number_text.SetBold(True)
            number_text.SetJustificationToCentered()
            number_text.SetVerticalJustificationToCentered()
            moon_site_halo_actors.append(halo_actor)
            moon_site_halo_sources.append(halo_source)
            moon_site_actors.append(marker_actor)
            moon_site_sources.append(marker_source)
            moon_site_number_actors.append(number_actor)

        site_lines = ["SITES LUNAIRES · MEILLEURE BASE DSN"]
        for index, site in enumerate(sites, start=1):
            latitude = site["latitude_deg"]
            longitude = site["longitude_deg"]
            sun_elevation = site["sun_elevation_deg"]
            solar_illumination = site["solar_illumination_percent"]
            latitude_label = "N" if latitude >= 0 else "S"
            longitude_label = "E" if longitude >= 0 else "O"
            if site["best_station"] is None:
                visibility_label = "aucune base visible"
            else:
                visibility_label = (
                    f"{site['best_station']} "
                    f"{site['max_visibility_deg']:+.1f}°"
                )
            site_lines.append(
                f"{index:02d}  {abs(latitude):.2f}° {latitude_label}  "
                f"{abs(longitude):.2f}° {longitude_label}  "
                f"{visibility_label}  "
                f"Soleil {sun_elevation:+.1f}°/{solar_illumination:.0f}%"
            )
        moon_sites_status_actor.SetInput("\n".join(site_lines))
        update_moon_site_marker_overlays()

    def update_moon_site_marker_overlays():
        """Projette les pastilles sur la surface visible, sans dépendre du zoom."""
        sites = simulation["moon_sites"]
        if not sites:
            return

        latitudes = np.deg2rad([site["latitude_deg"] for site in sites])
        longitudes = np.deg2rad([site["longitude_deg"] for site in sites])
        local_directions = np.column_stack(
            (
                np.cos(latitudes) * np.cos(longitudes),
                np.cos(latitudes) * np.sin(longitudes),
                np.sin(latitudes),
            )
        )
        moon_rotation = _moon_orientation_matrix(simulation["date"])
        site_directions = local_directions @ moon_rotation.T
        site_positions = (
            simulation["moon_position"]
            + site_directions * (moon_radius * 1.002)
        )
        camera_to_moon = (
            np.asarray(plotter.camera.position, dtype=float)
            - simulation["moon_position"]
        )
        width, height = plotter.render_window.GetSize()

        for index, (direction, position) in enumerate(
            zip(site_directions, site_positions)
        ):
            is_visible = float(np.dot(direction, camera_to_moon)) > 0.0
            if is_visible:
                plotter.renderer.SetWorldPoint(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    1.0,
                )
                plotter.renderer.WorldToDisplay()
                display_x, display_y, display_z = plotter.renderer.GetDisplayPoint()
                is_visible = (
                    0.0 <= display_x <= width
                    and 0.0 <= display_y <= height
                    and 0.0 <= display_z <= 1.0
                )
            halo_actor = moon_site_halo_actors[index]
            marker_actor = moon_site_actors[index]
            number_actor = moon_site_number_actors[index]
            if is_visible:
                for source in (
                    moon_site_halo_sources[index],
                    moon_site_sources[index],
                ):
                    source.SetCenter(display_x, display_y, 0.0)
                    source.Modified()
                number_actor.GetPositionCoordinate().SetValue(
                    display_x / width,
                    display_y / height,
                    0.0,
                )
                number_actor.SetInput(str(index + 1))
            halo_actor.SetVisibility(is_visible)
            marker_actor.SetVisibility(is_visible)
            number_actor.SetVisibility(is_visible)

    def on_moon_surface_picked(world_point, picker):
        """Enregistre le point sélectionné en latitude et longitude lunaires."""
        picked_actor = picker.GetActor()
        if (
            picked_actor is None
            or picked_actor.GetAddressAsString("")
            != moon_actor.GetAddressAsString("")
        ):
            return

        moon_rotation = _moon_orientation_matrix(simulation["date"])
        local_point = moon_rotation.T @ (
            np.asarray(world_point, dtype=float) - simulation["moon_position"]
        )
        local_direction = local_point / np.linalg.norm(local_point)
        latitude_deg = np.rad2deg(
            np.arcsin(np.clip(local_direction[2], -1.0, 1.0))
        )
        longitude_deg = np.rad2deg(
            np.arctan2(local_direction[1], local_direction[0])
        )
        new_site_index = len(simulation["moon_sites"])
        simulation["moon_sites"].append(
            {
                "latitude_deg": float(latitude_deg),
                "longitude_deg": float(longitude_deg),
            }
        )
        if simulation["chart_site_indices"] is not None:
            simulation["chart_site_indices"].add(new_site_index)
        update_moon_site_elevations(simulation["moon_position"])
        update_moon_site_illumination(
            simulation["moon_position"],
            simulation["sun_position"],
            simulation["date"],
        )
        refresh_moon_sites()
        refresh_screen()

    clear_sites_button = None
    elevation_chart_button = None
    elevation_chart_state = {
        "figure": None,
        "station_geometry": None,
        "moon_rotations": None,
        "sun_positions": None,
        "window": {
            "running": False,
            "updates": Queue(),
            "lock": Lock(),
            "error": None,
            "error_poll_timer_started": False,
        },
    }

    def clear_moon_sites(is_pressed=True):
        """Supprime tous les sites enregistrés sur la Lune."""
        if not is_pressed:
            return
        simulation["moon_sites"].clear()
        simulation["chart_site_indices"] = None
        refresh_moon_sites()
        if clear_sites_button is not None:
            clear_sites_button.GetRepresentation().SetState(0)
        refresh_screen()

    def poll_chart_window_error(_step):
        """Affiche dans la scène toute erreur d'ouverture de la fenêtre Tk."""
        window_state = elevation_chart_state["window"]
        with window_state["lock"]:
            error = window_state["error"]
            window_state["error"] = None
        if error:
            moon_sites_status_actor.SetInput(
                f"Graphique indisponible : {error}"
            )
            refresh_screen()

    def show_elevation_chart(is_pressed=True):
        """Trace l'accès DSN et l'éclairement solaire de chaque site lunaire."""
        if not is_pressed:
            return
        try:
            import matplotlib.dates as mdates
            from matplotlib.figure import Figure
            from matplotlib.colors import ListedColormap
            from matplotlib.patches import Patch

            all_sites = simulation["moon_sites"]
            selected_indices = simulation["chart_site_indices"]
            sites = list(enumerate(all_sites))
            if not all_sites:
                moon_sites_status_actor.SetInput(
                    "Pour tracer le graphique, cliquez d'abord sur un point de la Lune."
                )
                refresh_screen()
                return

            moon_sites_status_actor.SetInput(
                "Calcul du graphique sur la période de simulation…"
            )
            refresh_screen()
            figure = Figure(figsize=(14, 9))
            elevation_chart_state["figure"] = figure
            axes_elevation, axes_availability, axes_illumination = figure.subplots(
                3,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": (2.3, 1.0, 1.25)},
            )

            track_date_numbers = np.asarray(
                mdates.date2num(track_dates),
                dtype=float,
            )
            date_edges = np.empty(len(track_date_numbers) + 1, dtype=float)
            date_edges[1:-1] = (
                track_date_numbers[:-1] + track_date_numbers[1:]
            ) / 2.0
            date_edges[0] = track_date_numbers[0] - (
                track_date_numbers[1] - track_date_numbers[0]
            ) / 2.0
            date_edges[-1] = track_date_numbers[-1] + (
                track_date_numbers[-1] - track_date_numbers[-2]
            ) / 2.0

            if elevation_chart_state["station_geometry"] is None:
                elevation_chart_state["station_geometry"] = _dsn_station_geometry(
                    track_ets
                )
            if elevation_chart_state["moon_rotations"] is None:
                elevation_chart_state["moon_rotations"] = (
                    _moon_orientation_matrix_series(track_ets)
                )
            if elevation_chart_state["sun_positions"] is None:
                elevation_chart_state["sun_positions"] = (
                    _spk_position_series_j2000("SUN", "EARTH", track_ets)
                )
            station_geometry = elevation_chart_state["station_geometry"]
            moon_rotations = elevation_chart_state["moon_rotations"]
            sun_positions = elevation_chart_state["sun_positions"]

            best_station_rows = []
            site_labels = []
            site_artists = {}
            site_options = []

            for site_index, site in sites:
                index = site_index + 1
                lat = site["latitude_deg"]
                lon = site["longitude_deg"]
                latitude_label = "N" if lat >= 0 else "S"
                longitude_label = "E" if lon >= 0 else "O"
                dsn_elevations, best_station_indices, _ = (
                    _dsn_visibility_for_site_series(
                        track_positions,
                        moon_rotations,
                        lat,
                        lon,
                        station_geometry,
                        moon_radius_km=Moon.r_mean_km,
                    )
                )
                has_visibility = np.isfinite(dsn_elevations)
                is_available = has_visibility & (
                    dsn_elevations >= MIN_EARTH_ELEVATION_DEG
                )
                best_station_rows.append(best_station_indices)

                earth_elevations = _earth_elevation_moon_series_j2000(
                    track_positions,
                    moon_rotations,
                    lat,
                    lon,
                    moon_radius_km=Moon.r_mean_km,
                )

                longest_outage = 0
                current_outage = 0
                for available in is_available:
                    if available:
                        current_outage = 0
                    else:
                        current_outage += 1
                        longest_outage = max(longest_outage, current_outage)
                available_percent = 100.0 * float(np.mean(is_available))
                site_label = (
                    f"Point {index} ({abs(lat):.1f}° {latitude_label}, "
                    f"{abs(lon):.1f}° {longitude_label}) — "
                    f"DSN {available_percent:.1f}% disponible, "
                    f"coupure max. {longest_outage} j"
                )
                site_labels.append(f"Point {index}")
                site_options.append(
                    (
                        site_index,
                        f"Point {index} · {abs(lat):.1f}° {latitude_label}, "
                        f"{abs(lon):.1f}° {longitude_label}",
                    )
                )
                elevation_line, = axes_elevation.plot(
                    track_dates,
                    earth_elevations,
                    linewidth=1.2,
                    label=site_label,
                )
                _, solar_illumination = _solar_illumination_moon_series_j2000(
                    track_positions,
                    sun_positions,
                    moon_rotations,
                    lat,
                    lon,
                    moon_radius_km=Moon.r_mean_km,
                )
                illumination_line, = axes_illumination.plot(
                    track_dates,
                    solar_illumination,
                    color=elevation_line.get_color(),
                    linewidth=1.2,
                    label=site_label,
                )
                site_artists[site_index] = (
                    elevation_line,
                    illumination_line,
                )

            axes_elevation.axhline(
                0.0,
                color="#555555",
                linestyle="--",
                linewidth=0.9,
                label="Horizon géométrique (0°)",
            )
            axes_elevation.set_xlim(track_dates[0], track_dates[-1])
            axes_elevation.set_ylim(-90.0, 90.0)
            axes_elevation.set_yticks(np.arange(-90.0, 91.0, 30.0))
            axes_elevation.set_ylabel("Élévation du centre de la Terre (°)")
            axes_elevation.set_title(
                "Élévation géométrique de la Terre depuis chaque site lunaire"
            )
            axes_elevation.grid(True, alpha=0.25)
            axes_elevation.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=7,
                title="Sites · accès DSN",
            )

            best_station_matrix = np.vstack(best_station_rows)
            best_station_mesh = axes_availability.pcolormesh(
                date_edges,
                np.arange(len(sites) + 1, dtype=float),
                best_station_matrix,
                cmap=ListedColormap(
                    ("#454F5C", *(station["color"] for station in DSN_STATIONS))
                ),
                vmin=-1.0,
                vmax=float(len(DSN_STATIONS) - 1),
                shading="flat",
            )
            axes_availability.set_yticks(
                np.arange(len(sites), dtype=float) + 0.5
            )
            axes_availability.set_yticklabels(site_labels)
            axes_availability.set_ylim(len(sites), 0)
            axes_availability.set_ylabel("Point lunaire")
            axes_availability.xaxis.set_major_locator(mdates.YearLocator(5))
            axes_availability.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y")
            )
            axes_availability.legend(
                handles=(
                    Patch(facecolor="#454F5C", label="Aucune visibilité"),
                    *(
                        Patch(
                            facecolor=station["color"],
                            label=station["name"],
                        )
                        for station in DSN_STATIONS
                    ),
                ),
                loc="upper right",
                ncol=4,
                fontsize=7,
            )
            axes_illumination.axhline(
                0.0,
                color="#555555",
                linestyle="--",
                linewidth=0.9,
            )
            axes_illumination.set_ylim(0.0, 110.0)
            axes_illumination.set_yticks((0.0, 25.0, 50.0, 75.0, 100.0))
            axes_illumination.set_ylabel("Éclairement (%)")
            axes_illumination.set_xlabel("Date UTC")
            axes_illumination.set_title(
                "Éclairement direct d'un panneau horizontal"
            )
            axes_illumination.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=7,
                title="Points lunaires",
            )
            axes_illumination.xaxis.set_major_locator(mdates.YearLocator(5))
            axes_illumination.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y")
            )
            axes_illumination.grid(True, alpha=0.25)
            figure.suptitle(
                "Terre, accès DSN et éclairement des sites lunaires",
                fontsize=13,
            )
            figure.text(
                0.09,
                0.025,
                "Courbe : centre de la Terre par rapport à l'horizon géométrique du site. "
                "Carte : base DSN seulement si les deux extrémités voient la liaison; "
                "seuil d'accès 10°. Soleil : irradiance relative à 1 UA sur une surface "
                "horizontale. Le relief lunaire n'est pas inclus.",
                fontsize=7.5,
            )
            figure.subplots_adjust(
                left=0.09,
                right=0.69,
                top=0.91,
                bottom=0.12,
                hspace=0.27,
            )
            _open_or_update_interactive_chart(
                {
                    "figure": figure,
                    "axes": {
                        "visibility": axes_elevation,
                        "stations": axes_availability,
                        "illumination": axes_illumination,
                    },
                    "site_indices": [site_index for site_index, _site in sites],
                    "site_options": site_options,
                    "site_labels": site_labels,
                    "site_artists": site_artists,
                    "station_matrix": best_station_matrix,
                    "station_mesh": best_station_mesh,
                    "selected_site_indices": (
                        set(range(len(all_sites)))
                        if selected_indices is None
                        else set(selected_indices)
                    ),
                    "on_site_selection": lambda indices: simulation.__setitem__(
                        "chart_site_indices",
                        set(indices),
                    ),
                },
                elevation_chart_state["window"],
            )
            if not elevation_chart_state["window"]["error_poll_timer_started"]:
                plotter.add_timer_event(
                    max_steps=2_000_000_000,
                    duration=200,
                    callback=poll_chart_window_error,
                )
                elevation_chart_state["window"]["error_poll_timer_started"] = True
            refresh_moon_sites()
        except Exception as error:
            message = f"Graphique indisponible : {error}"
            print(message)
            moon_sites_status_actor.SetInput(message)
            refresh_screen()
        finally:
            if elevation_chart_button is not None:
                elevation_chart_button.GetRepresentation().SetState(0)

    surface_picker = vtkCellPicker()
    surface_picker.SetTolerance(0.0005)
    click_state = {"origin": None}

    def click_is_over_interface(x, y):
        """Évite de créer un site en cliquant sur les contrôles ou leurs textes."""
        width, height = plotter.window_size
        if 15 < x < 165 and 15 < y < 72:  # bouton et étiquette Lecture
            return True
        if 180 < x < 315 and 15 < y < 72:  # bouton et étiquette Date UTC
            return True
        if 340 < x < 500 and 15 < y < 65:  # bouton et étiquette Effacer
            return True
        if 510 < x < 660 and 15 < y < 65:  # bouton et étiquette Graphique
            return True
        if x < 160 and 560 < y < 670:  # boutons de cadrage Terre/Lune
            return True
        if x < 260 and 450 < y < 520:  # bouton de visibilité de l'orbite
            return True
        if width * 0.15 < x < width * 0.96 and height * 0.035 < y < height * 0.18:
            return True  # curseur de date
        if width * 0.68 < x < width * 0.98 and height * 0.83 < y < height * 0.93:
            return True  # curseur de vitesse
        return False

    def interactor_from_event(event_source):
        """Récupère le render-window interactor pour les événements VTK."""
        if hasattr(event_source, "GetEventPosition"):
            return event_source
        get_interactor = getattr(event_source, "GetInteractor", None)
        if get_interactor is not None:
            interactor = get_interactor()
            if interactor is not None:
                return interactor
        return plotter.iren.interactor

    def remember_left_click(event_source, _event):
        interactor = interactor_from_event(event_source)
        click_state["origin"] = tuple(interactor.GetEventPosition())

    def pick_moon_on_click_release(event_source, _event):
        """N'ajoute un site qu'après un clic bref intersectant la Lune."""
        interactor = interactor_from_event(event_source)
        origin = click_state["origin"]
        click_state["origin"] = None
        if origin is None:
            return

        x, y = interactor.GetEventPosition()
        if np.hypot(x - origin[0], y - origin[1]) > 5:
            return  # un glissement sert à déplacer la caméra
        if 238 < x < 320 and 26 < y < 55:
            enter_date_utc(True)
            return
        if click_is_over_interface(x, y):
            return
        if not surface_picker.Pick(x, y, 0.0, plotter.renderer):
            return

        picked_actor = surface_picker.GetActor()
        if (
            picked_actor is None
            or picked_actor.GetAddressAsString("")
            != moon_actor.GetAddressAsString("")
        ):
            return

        world_point = np.asarray(surface_picker.GetPickPosition(), dtype=float)
        world_point /= np.asarray(plotter.scale, dtype=float)
        distance_to_moon_center = np.linalg.norm(
            world_point - simulation["moon_position"]
        )
        if not np.isclose(
            distance_to_moon_center,
            moon_radius,
            rtol=0.0,
            atol=moon_radius * 0.01,
        ):
            return
        on_moon_surface_picked(world_point, surface_picker)

    def date_from_slider(year_offset):
        """Convertit la valeur du curseur en date UTC dans la fenêtre de 40 ans."""
        fraction = min(abs(float(year_offset)) / 20.0, 1.0)
        if year_offset < 0:
            return reference_date - (reference_date - start_date) * fraction
        return reference_date + (end_date - reference_date) * fraction

    def slider_from_date(selected_date):
        """Convertit une date UTC en valeur pour le curseur temporel."""
        if selected_date < reference_date:
            fraction = (reference_date - selected_date).total_seconds() / (
                reference_date - start_date
            ).total_seconds()
            return -20.0 * fraction
        fraction = (selected_date - reference_date).total_seconds() / (
            end_date - reference_date
        ).total_seconds()
        return 20.0 * fraction

    def update_scene(selected_date):
        """Place la Lune à la date choisie et actualise les coordonnées."""
        selected_date = selected_date.replace(microsecond=0)
        if selected_date == reference_date:
            position = initial_position
            sun_position = initial_sun_position
        else:
            position = _moon_position_j2000(selected_date)
            sun_position = _sun_position_j2000(selected_date)

        previous_moon_position = simulation["moon_position"]
        simulation["date"] = selected_date
        simulation["moon_position"] = np.array(position, dtype=float)
        simulation["sun_position"] = np.array(sun_position, dtype=float)
        update_satellite_scene(selected_date)
        update_moon_site_elevations(position)
        update_moon_site_illumination(position, sun_position, selected_date)
        distance = np.linalg.norm(position)
        sun_distance = np.linalg.norm(sun_position)
        sun_direction = sun_position / sun_distance
        sun_display_position = sun_direction * SUN_VISUAL_DISTANCE_KM
        moon_rotation = _moon_orientation_matrix(selected_date)
        moon_mesh.points[...] = local_points @ moon_rotation.T + position
        moon_mesh.point_data["Normals"] = local_normals @ moon_rotation.T
        moon_mesh.point_data.active_normals_name = "Normals"
        moon_mesh.Modified()
        refresh_moon_sites()
        moon_link_mesh.points[...] = _moon_link_dash_points(position)
        moon_link_mesh.Modified()
        sun_actor.position = tuple(sun_display_position)
        sun_halo_actor.position = tuple(sun_display_position)
        sun_ray_mesh.points = np.array([[0.0, 0.0, 0.0], sun_display_position])
        sun_ray_mesh.Modified()
        # La source ponctuelle suit la position DE440s réelle du Soleil.
        # VTK calcule ainsi un vecteur d'éclairage propre à chaque surface.
        sun_light.position = tuple(sun_position)
        sun_light.focal_point = (0.0, 0.0, 0.0)
        # La rotation terrestre sert à orienter la texture pour le rendu seulement.
        sidereal_angle = ComputeGMST(selected_date)
        earth_actor.orientation = (0.0, 0.0, sidereal_angle)
        earth_rotation = _earth_rotation_matrix_series_j2000(
            [_spice_et(selected_date)]
        )
        for station_name, station_actor in earth_station_actors.items():
            station_actor.position = tuple(
                _earth_fixed_vector_series_j2000(
                    earth_station_positions_fixed[station_name],
                    earth_rotation,
                )[0]
                * DISPLAY_RADIUS_SCALE
            )
        starfield_actor.orientation = (0.0, 0.0, sidereal_angle)
        if simulation["camera_focus"] == "moon":
            camera_shift = simulation["moon_position"] - previous_moon_position
            plotter.camera.focal_point = tuple(
                np.asarray(plotter.camera.focal_point) + camera_shift
            )
            plotter.camera.position = tuple(
                np.asarray(plotter.camera.position) + camera_shift
            )
        clamp_camera_to_sky_sphere()
        status_actor.SetInput(
            f"Date UTC : {selected_date:%Y-%m-%d %H:%M:%S}\n"
            f"Distance Terre–Lune : {_format_km(distance)} km\n"
            f"J2000 X : {_format_km(position[0], 1)} km\n"
            f"J2000 Y : {_format_km(position[1], 1)} km\n"
            f"J2000 Z : {_format_km(position[2], 1)} km"
        )
        sun_status_actor.SetInput(
            f"SOLEIL — position DE440s\n"
            f"Distance réelle : {_format_km(sun_distance)} km "
            f"({sun_distance / Earth.AU:.3f} AU)\n"
            f"J2000 X : {_format_km(sun_position[0])} km\n"
            f"J2000 Y : {_format_km(sun_position[1])} km\n"
            f"J2000 Z : {_format_km(sun_position[2])} km"
        )
        refresh_screen()

    def on_time_slider(year_offset):
        update_scene(date_from_slider(year_offset))

    date_entry_state = {"active": False, "text": ""}
    date_entry_button = None

    def enter_date_utc(is_active):
        """Ouvre un champ de saisie direct pour choisir la date UTC."""
        if not is_active:
            return

        if date_entry_button is not None:
            date_entry_button.GetRepresentation().SetState(0)

        dialog_root = None
        selected_date = None
        try:
            import tkinter as tk
            from tkinter import messagebox, simpledialog

            dialog_root = tk.Tk()
            dialog_root.withdraw()
            date_text = simpledialog.askstring(
                "Date UTC",
                "Saisis la date au format AAAA/MM/JJ HH:MM:SS :",
                initialvalue=simulation["date"].strftime(
                    "%Y/%m/%d %H:%M:%S"
                ),
                parent=dialog_root,
            )
            if date_text is None:
                return
            try:
                selected_date = datetime.strptime(
                    date_text.strip(),
                    "%Y/%m/%d %H:%M:%S",
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                messagebox.showerror(
                    "Date invalide",
                    "Utilise le format AAAA/MM/JJ HH:MM:SS.",
                    parent=dialog_root,
                )
                return
            if not start_date <= selected_date <= end_date:
                messagebox.showerror(
                    "Date hors plage",
                    f"Choisis une date entre {start_date:%Y/%m/%d} et "
                    f"{end_date:%Y/%m/%d} UTC.",
                    parent=dialog_root,
                )
                return
        except Exception as error:
            date_entry_state["active"] = True
            date_entry_state["text"] = simulation["date"].strftime(
                "%Y/%m/%d %H:%M:%S"
            )
            show_date_entry(
                f"Fenêtre de saisie indisponible ({error}); "
                "saisis la date au clavier puis Entrée."
            )
            return
        finally:
            if dialog_root is not None:
                try:
                    dialog_root.destroy()
                except Exception:
                    pass

        if selected_date is not None:
            time_slider.GetRepresentation().SetValue(
                slider_from_date(selected_date)
            )
            update_scene(selected_date)

    def show_date_entry(message=None):
        """Affiche le repli de saisie clavier si la boîte de dialogue échoue."""
        if message is None:
            message = (
                "Entrée : appliquer · Suppr : tout effacer · "
                "Retour arrière : retirer un caractère · Échap : annuler"
            )
        date_entry_actor.SetInput(
            f"DATE UTC (AAAA/MM/JJ HH:MM:SS) : {date_entry_state['text']}\n"
            f"{message}"
        )
        date_entry_actor.SetVisibility(True)
        refresh_screen()

    def close_date_entry():
        """Ferme la saisie sans modifier la date simulée."""
        date_entry_state["active"] = False
        date_entry_actor.SetVisibility(False)
        if date_entry_button is not None:
            date_entry_button.GetRepresentation().SetState(0)
        refresh_screen()

    def append_date_character(character):
        if date_entry_state["active"] and len(date_entry_state["text"]) < 19:
            date_entry_state["text"] += character
            show_date_entry()

    def remove_last_date_character():
        if date_entry_state["active"]:
            date_entry_state["text"] = date_entry_state["text"][:-1]
            show_date_entry()

    def clear_date_entry():
        if date_entry_state["active"]:
            date_entry_state["text"] = ""
            show_date_entry()

    def commit_date_entry():
        if not date_entry_state["active"]:
            return
        try:
            selected_date = datetime.strptime(
                date_entry_state["text"], "%Y/%m/%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            show_date_entry("Format requis : AAAA/MM/JJ HH:MM:SS")
            return

        if not start_date <= selected_date <= end_date:
            show_date_entry(
                f"Date hors plage : {start_date:%Y/%m/%d} à "
                f"{end_date:%Y/%m/%d} UTC"
            )
            return

        date_entry_state["active"] = False
        date_entry_actor.SetVisibility(False)
        if date_entry_button is not None:
            date_entry_button.GetRepresentation().SetState(0)
        time_slider.GetRepresentation().SetValue(slider_from_date(selected_date))
        update_scene(selected_date)

    def on_speed_slider(days_per_second):
        simulation["days_per_second"] = float(days_per_second)

    def refresh_screen():
        """Efface l'image précédente puis redessine la scène complète."""
        plotter.renderer.SetErase(True)
        render_window = plotter.render_window
        if render_window is not None:
            render_window.SetErase(True)
        update_moon_site_marker_overlays()
        plotter.render()

    def update_satellite_status():
        """Résume l'état des trajectoires sélectionnées dans l'onglet."""
        lines = []
        for satellite_key, satellite in CISLUNAR_SATELLITES.items():
            state = satellite_states[satellite_key]
            if not state["enabled"]:
                continue
            if state["loading"]:
                lines.append(f"{satellite['name']} : chargement depuis Horizons…")
            elif state["error"]:
                lines.append(f"{satellite['name']} : {state['error']}")
            elif state["track_start"] is not None:
                lines.append(f"{satellite['name']} : trajectoire active")

        if not lines:
            lines.append("Choisissez une trajectoire pour la date simulée.")
        satellite_status_actor.SetInput("\n".join(lines))
        satellite_status_actor.SetVisibility(satellite_panel_is_open)

    def set_satellite_panel_open(is_open):
        """Ouvre ou ferme le panneau de sélection des satellites."""
        nonlocal satellite_panel_is_open
        satellite_panel_is_open = bool(is_open)
        satellite_panel_heading.SetVisibility(satellite_panel_is_open)
        for actor in satellite_label_actors.values():
            actor.SetVisibility(satellite_panel_is_open)
        satellite_status_actor.SetVisibility(satellite_panel_is_open)
        for widget in satellite_checkbox_widgets.values():
            widget.SetEnabled(satellite_panel_is_open)

        text_property = satellite_tab_title_actor.GetTextProperty()
        if satellite_panel_is_open:
            text_property.SetColor(pv.Color("#F4F8FC").float_rgb)
            text_property.SetBackgroundColor(pv.Color("#17313C").float_rgb)
            text_property.SetFrameColor(pv.Color("#58C4DD").float_rgb)
            text_property.SetBold(True)
        else:
            text_property.SetColor(pv.Color("#AAB7C8").float_rgb)
            text_property.SetBackgroundColor(pv.Color("#17202B").float_rgb)
            text_property.SetFrameColor(pv.Color("#354253").float_rgb)
            text_property.SetBold(False)
        text_property.SetBackgroundOpacity(0.96)
        text_property.SetFrame(True)
        refresh_screen()

    def start_horizons_poll_timer():
        """Dépile les réponses réseau sur le fil de l'interface VTK."""
        nonlocal horizons_poll_timer_started
        if horizons_poll_timer_started:
            return
        plotter.add_timer_event(
            max_steps=2_000_000_000,
            duration=200,
            callback=poll_horizons_results,
        )
        horizons_poll_timer_started = True

    def request_satellite_track(satellite_key, selected_date):
        """Charge une fenêtre locale sans bloquer l'interface graphique."""
        state = satellite_states[satellite_key]
        if state["loading"] or monotonic() < state["retry_after"]:
            state["desired_date"] = selected_date
            return

        coverage = state["coverage"]
        if coverage is not None and not (
            coverage[0] - HORIZONS_EDGE_TOLERANCE
            <= selected_date
            <= coverage[1] + HORIZONS_EDGE_TOLERANCE
        ):
            state["error"] = (
                f"pas de données le {selected_date:%Y-%m-%d} "
                f"({coverage[0]:%Y-%m-%d} → {coverage[1]:%Y-%m-%d})"
            )
            state["marker_actor"].visibility = False
            if state["track_actor"] is not None:
                state["track_actor"].visibility = False
            update_satellite_status()
            return

        state["generation"] += 1
        generation = state["generation"]
        state["loading"] = True
        state["desired_date"] = selected_date
        state["error"] = None
        state["marker_actor"].visibility = False
        if state["track_actor"] is not None:
            state["track_actor"].visibility = False
        update_satellite_status()
        start_horizons_poll_timer()

        satellite = CISLUNAR_SATELLITES[satellite_key]
        known_coverage = coverage

        def load_in_background():
            try:
                result = _load_horizons_satellite_track(
                    satellite,
                    selected_date,
                    coverage=known_coverage,
                )
            except Exception as error:
                result = {"error": str(error)}
            horizons_results.put((satellite_key, generation, result))

        Thread(target=load_in_background, daemon=True).start()

    def update_satellite_scene(selected_date):
        """Suit les marqueurs et la portion d'orbite de la date simulée."""
        selected_seconds = selected_date.timestamp()
        for satellite_key, satellite in CISLUNAR_SATELLITES.items():
            state = satellite_states[satellite_key]
            if not state["enabled"]:
                continue
            state["desired_date"] = selected_date

            coverage = state["coverage"]
            if coverage is not None and not (
                coverage[0] - HORIZONS_EDGE_TOLERANCE
                <= selected_date
                <= coverage[1] + HORIZONS_EDGE_TOLERANCE
            ):
                state["marker_actor"].visibility = False
                if state["track_actor"] is not None:
                    state["track_actor"].visibility = False
                if not state["loading"]:
                    state["error"] = (
                        f"pas de données le {selected_date:%Y-%m-%d} "
                        f"({coverage[0]:%Y-%m-%d} → {coverage[1]:%Y-%m-%d})"
                    )
                    update_satellite_status()
                continue

            times = state["times"]
            if (
                times is not None
                and state["track_start"] - HORIZONS_EDGE_TOLERANCE
                <= selected_date
                <= state["track_stop"] + HORIZONS_EDGE_TOLERANCE
            ):
                index = int(np.searchsorted(times, selected_seconds))
                if index <= 0:
                    position = state["positions"][0]
                elif index >= len(times):
                    position = state["positions"][-1]
                else:
                    before = index - 1
                    fraction = (selected_seconds - times[before]) / (
                        times[index] - times[before]
                    )
                    position = (
                        state["positions"][before] * (1.0 - fraction)
                        + state["positions"][index] * fraction
                    )
                state["marker_actor"].position = tuple(position)
                state["marker_actor"].visibility = True
                state["track_actor"].visibility = True
                if state["error"] is not None:
                    state["error"] = None
                    update_satellite_status()
                continue

            state["marker_actor"].visibility = False
            if state["track_actor"] is not None:
                state["track_actor"].visibility = False
            if not state["loading"]:
                request_satellite_track(satellite_key, selected_date)

    def poll_horizons_results(step):
        """Applique les trajectoires téléchargées depuis le fil principal."""
        changed = False
        while True:
            try:
                satellite_key, generation, result = horizons_results.get_nowait()
            except Empty:
                break

            state = satellite_states[satellite_key]
            if generation != state["generation"]:
                continue
            state["loading"] = False
            if not state["enabled"]:
                continue
            if result.get("error"):
                state["error"] = result["error"]
                state["retry_after"] = monotonic() + 30.0
                changed = True
                continue

            state["coverage"] = result["coverage"]
            if result.get("unavailable"):
                selected_date = state["desired_date"] or simulation["date"]
                start, stop = state["coverage"]
                state["error"] = (
                    f"pas de données le {selected_date:%Y-%m-%d} "
                    f"({start:%Y-%m-%d} → {stop:%Y-%m-%d})"
                )
                state["retry_after"] = 0.0
                changed = True
                continue

            track_mesh = pv.lines_from_points(result["positions"], close=False)
            if state["track_actor"] is None:
                state["track_actor"] = plotter.add_mesh(
                    track_mesh,
                    name=f"Trajectoire {CISLUNAR_SATELLITES[satellite_key]['name']}",
                    color=CISLUNAR_SATELLITES[satellite_key]["color"],
                    line_width=2.4,
                    opacity=0.96,
                    lighting=False,
                    pickable=False,
                    render=False,
                )
            else:
                state["track_actor"].mapper.SetInputData(track_mesh)
                state["track_actor"].mapper.Update()
            state["track_mesh"] = track_mesh
            state["track_start"] = result["start"]
            state["track_stop"] = result["stop"]
            state["times"] = result["times"]
            state["positions"] = result["positions"]
            state["error"] = None
            state["retry_after"] = 0.0
            changed = True

        for satellite_key, state in satellite_states.items():
            if state["enabled"] and not state["loading"]:
                update_satellite_scene(simulation["date"])
        if changed:
            update_satellite_status()
            refresh_screen()

    def toggle_satellite(satellite_key, is_visible):
        """Active ou masque un satellite cis-lunaire."""
        state = satellite_states[satellite_key]
        state["enabled"] = bool(is_visible)
        if not is_visible:
            state["generation"] += 1
            state["loading"] = False
            state["error"] = None
            state["marker_actor"].visibility = False
            if state["track_actor"] is not None:
                state["track_actor"].visibility = False
        else:
            state["error"] = None
            update_satellite_scene(simulation["date"])
        update_satellite_status()
        refresh_screen()

    def clamp_camera_to_sky_sphere(*_):
        """Empêche la caméra de sortir de la sphère céleste."""
        camera = plotter.camera
        focal = np.asarray(camera.focal_point, dtype=float)
        position = np.asarray(camera.position, dtype=float)
        camera_radius = float(np.linalg.norm(position))
        camera_limit = SKY_SPHERE_RADIUS_KM * MAX_CAMERA_RADIUS_FRACTION
        camera_was_clamped = False

        # Si un déplacement place aussi le point focal hors de la sphère,
        # translate toute la caméra vers l'intérieur en conservant sa direction.
        if np.linalg.norm(focal) >= camera_limit:
            if camera_radius > camera_limit:
                corrected_position = position * (camera_limit / camera_radius)
                shift = corrected_position - position
                camera.position = tuple(corrected_position)
                camera.focal_point = tuple(focal + shift)
                camera_was_clamped = True
            return camera_was_clamped

        offset = position - focal
        distance = float(np.linalg.norm(offset))
        if distance == 0.0:
            return camera_was_clamped

        direction = offset / distance
        projection = float(np.dot(focal, direction))
        focal_radius_squared = float(np.dot(focal, focal))
        discriminant = (
            projection**2 + camera_limit**2 - focal_radius_squared
        )
        max_distance = max(0.0, -projection + np.sqrt(max(discriminant, 0.0)))
        if distance > max_distance:
            camera.position = tuple(focal + direction * max_distance)
            camera_was_clamped = True
        return camera_was_clamped

    def clamp_camera_during_interaction(*_):
        """Réapplique la limite après chaque zoom, y compris à la molette."""
        if clamp_camera_to_sky_sphere():
            refresh_screen()
        else:
            update_moon_site_marker_overlays()

    def toggle_play(is_playing):
        simulation["playing"] = bool(is_playing)
        play_label_actor.SetInput("⏸  PAUSE" if is_playing else "▶  LECTURE")
        if is_playing and not simulation["timer_started"]:
            plotter.add_timer_event(
                max_steps=2_000_000_000,
                duration=TIMER_INTERVAL_MS,
                callback=advance_simulation,
            )
            simulation["timer_started"] = True
            advance_simulation(0)
        refresh_screen()

    def toggle_orbits(is_visible):
        history_actor.visibility = bool(is_visible)
        future_actor.visibility = bool(is_visible)
        update_orbit_control_style(is_visible)
        refresh_screen()

    def update_orbit_control_style(is_visible):
        """Adapte l'étiquette à l'état d'affichage de l'orbite."""
        text_property = orbit_status_actor.GetTextProperty()
        if is_visible:
            orbit_status_actor.SetInput("  Orbite lunaire · affichée  ")
            text_property.SetColor(pv.Color("#F4F8FC").float_rgb)
            text_property.SetBackgroundColor(pv.Color("#17313C").float_rgb)
            text_property.SetFrameColor(pv.Color("#58C4DD").float_rgb)
            text_property.SetBold(True)
        else:
            orbit_status_actor.SetInput("  Orbite lunaire · masquée  ")
            text_property.SetColor(pv.Color("#AAB7C8").float_rgb)
            text_property.SetBackgroundColor(pv.Color("#17202B").float_rgb)
            text_property.SetFrameColor(pv.Color("#354253").float_rgb)
            text_property.SetBold(False)
        text_property.SetBackgroundOpacity(0.96)
        text_property.SetFrame(True)

    def center_camera(target, focus_name):
        """Recentre la caméra en conservant sa direction et son zoom."""
        camera = plotter.camera
        target = np.asarray(target, dtype=float)
        camera_offset = np.asarray(camera.position) - np.asarray(camera.focal_point)
        camera.focal_point = tuple(target)
        camera.position = tuple(target + camera_offset)
        simulation["camera_focus"] = focus_name
        update_focus_control_style(focus_name)
        clamp_camera_to_sky_sphere()
        refresh_screen()

    def update_focus_control_style(focus_name):
        """Met en évidence l'objet choisi comme centre de la vue."""
        focus_colors = {
            "earth": ("#58C4DD", "#17313C"),
            "moon": ("#F3B45B", "#382B1C"),
        }
        for name, actor in focus_title_actors.items():
            text_property = actor.GetTextProperty()
            if name == focus_name:
                accent, background = focus_colors[name]
                text_property.SetColor(pv.Color("#F4F8FC").float_rgb)
                text_property.SetBackgroundColor(pv.Color(background).float_rgb)
                text_property.SetFrameColor(pv.Color(accent).float_rgb)
                text_property.SetBold(True)
            else:
                text_property.SetColor(pv.Color("#AAB7C8").float_rgb)
                text_property.SetBackgroundColor(pv.Color("#17202B").float_rgb)
                text_property.SetFrameColor(pv.Color("#354253").float_rgb)
                text_property.SetBold(False)
            text_property.SetBackgroundOpacity(0.96)
            text_property.SetFrame(True)

    def center_on_earth():
        center_camera(np.zeros(3), "earth")

    def center_on_moon():
        center_camera(simulation["moon_position"], "moon")

    def advance_simulation(step):
        if not simulation["playing"]:
            return

        elapsed_days = (
            simulation["days_per_second"] * TIMER_INTERVAL_MS / 1000.0
        )
        next_date = simulation["date"] + timedelta(days=elapsed_days)
        if next_date > end_date:
            next_date = start_date

        time_slider.GetRepresentation().SetValue(slider_from_date(next_date))
        update_scene(next_date)

    plotter.add_checkbox_button_widget(
        toggle_play,
        value=False,
        position=(22, 24),
        size=42,
        border_size=3,
        color_on="#45C98B",
        color_off="#64748B",
        background_color="#111B2C",
    )
    date_entry_button = plotter.add_checkbox_button_widget(
        enter_date_utc,
        value=False,
        position=(190, 24),
        size=42,
        border_size=3,
        color_on="#58C4DD",
        color_off="#64748B",
        background_color="#111B2C",
    )
    plotter.add_text(
        "SAISIR DATE",
        position=(245, 37),
        font_size=9,
        color="#E5EDF8",
        font="arial",
    )
    clear_sites_button = plotter.add_checkbox_button_widget(
        clear_moon_sites,
        value=False,
        position=(350, 24),
        size=30,
        border_size=3,
        color_on="#D96C6C",
        color_off="#64748B",
        background_color="#111B2C",
    )
    plotter.add_text(
        "EFFACER LES SITES",
        position=(389, 34),
        font_size=9,
        color="#E5EDF8",
        font="arial",
    )
    elevation_chart_button = plotter.add_checkbox_button_widget(
        show_elevation_chart,
        value=False,
        position=(520, 24),
        size=30,
        border_size=3,
        color_on="#8CB9E8",
        color_off="#64748B",
        background_color="#111B2C",
    )
    plotter.add_text(
        "GRAPHIQUE EXTERNE",
        position=(559, 34),
        font_size=9,
        color="#E5EDF8",
        font="arial",
    )
    plotter.add_checkbox_button_widget(
        set_satellite_panel_open,
        value=False,
        position=(685, 24),
        size=30,
        border_size=3,
        color_on="#58C4DD",
        color_off="#64748B",
        background_color="#111B2C",
    )
    satellite_tab_title_actor = plotter.add_text(
        "  Satellites  ",
        position=(724, 34),
        font_size=9,
        color="#AAB7C8",
        font="arial",
    )
    satellite_tab_property = satellite_tab_title_actor.GetTextProperty()
    satellite_tab_property.SetBackgroundColor(pv.Color("#17202B").float_rgb)
    satellite_tab_property.SetFrameColor(pv.Color("#354253").float_rgb)
    satellite_tab_property.SetBackgroundOpacity(0.96)
    satellite_tab_property.SetFrame(True)
    for satellite_key, position in (("lro", (22, 392)), ("capstone", (22, 344))):
        satellite = CISLUNAR_SATELLITES[satellite_key]
        widget = plotter.add_checkbox_button_widget(
            lambda is_visible, key=satellite_key: toggle_satellite(
                key,
                is_visible,
            ),
            value=False,
            position=position,
            size=26,
            border_size=3,
            color_on=satellite["color"],
            color_off="#64748B",
            background_color="#111B2C",
        )
        widget.SetEnabled(False)
        satellite_checkbox_widgets[satellite_key] = widget
    plotter.add_checkbox_button_widget(
        toggle_orbits,
        value=True,
        position=orbit_toggle_position,
        size=orbit_toggle_size,
        border_size=4,
        color_on="#58C4DD",
        color_off="#64748B",
        background_color="#17202B",
    )
    update_orbit_control_style(True)
    plotter.add_radio_button_widget(
        center_on_earth,
        radio_button_group="center_target",
        value=True,
        title="  Terre  ",
        position=(22, 626),
        size=28,
        border_size=4,
        color_on="#58C4DD",
        color_off="#64748B",
        background_color="#17202B",
    )
    plotter.add_radio_button_widget(
        center_on_moon,
        radio_button_group="center_target",
        value=False,
        title="  Lune  ",
        position=(22, 578),
        size=28,
        border_size=4,
        color_on="#F3B45B",
        color_off="#64748B",
        background_color="#17202B",
    )
    focus_title_actors.update(
        zip(("earth", "moon"), plotter.radio_button_title_dict["center_target"])
    )
    # PyVista fixe par défaut les titres des boutons radio à 15 px.
    for actor in focus_title_actors.values():
        actor.GetTextProperty().SetFontSize(11)
    update_focus_control_style("earth")
    camera_focus_heading = plotter.add_text(
        "Centre de la vue",
        position=(0.02, 0.75),
        font_size=9,
        color="#B9C8D8",
        font="arial",
        viewport=True,
    )
    camera_focus_heading.GetTextProperty().SetBold(True)
    play_label_actor = plotter.add_text(
        "▶  LECTURE",
        position=(76, 37),
        font_size=9,
        color="#E5EDF8",
        font="arial",
    )
    plotter.add_slider_widget(
        on_speed_slider,
        rng=(0.001, 30.0),
        value=DEFAULT_SIMULATION_DAYS_PER_SECOND,
        title="Jours simulés / seconde",
        pointa=(0.72, 0.87),
        pointb=(0.95, 0.87),
        title_height=0.022,
        color="#D7E3F4",
        title_color="white",
        fmt="%.3g",
        interaction_event="end",
        style="modern",
    )
    time_slider = plotter.add_slider_widget(
        on_time_slider,
        rng=(-20.0, 20.0),
        value=0.0,
        title="Date UTC — années par rapport à aujourd’hui",
        pointa=(0.24, 0.12),
        pointb=(0.88, 0.12),
        title_height=0.022,
        color="#D7E3F4",
        title_color="white",
        fmt="%+.1f ans",
        interaction_event="end",
        style="modern",
    )
    time_slider_representation = time_slider.GetRepresentation()
    time_slider_representation.SetTubeWidth(0.0045)
    time_slider_representation.SetSliderWidth(0.014)
    for digit in "0123456789":
        plotter.add_key_event(
            digit,
            lambda digit=digit: append_date_character(digit),
        )
    for key, character in (
        ("slash", "/"),
        ("space", " "),
        ("colon", ":"),
    ):
        plotter.add_key_event(
            key,
            lambda character=character: append_date_character(character),
        )
    plotter.add_key_event("BackSpace", remove_last_date_character)
    plotter.add_key_event("Delete", clear_date_entry)
    plotter.add_key_event("Return", commit_date_entry)
    plotter.add_key_event("KP_Enter", commit_date_entry)
    plotter.add_key_event("Escape", close_date_entry)
    plotter.iren.add_observer("InteractionEvent", clamp_camera_during_interaction)
    plotter.iren.add_observer("EndInteractionEvent", clamp_camera_during_interaction)
    plotter.iren.add_observer("LeftButtonPressEvent", remember_left_click)
    plotter.iren.add_observer("LeftButtonReleaseEvent", pick_moon_on_click_release)
    update_scene(reference_date)
    return plotter


def main():
    """Ouvre le visualiseur interactif."""
    plotter = create_scene()
    plotter.show(title="NASA — trajectoire de la Lune sur 40 ans")


if __name__ == "__main__":
    main()
