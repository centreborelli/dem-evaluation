import numpy as np
import subprocess
import os
import json
import zipfile
import requests
import rpcm
import bs4
import utm


def bounding_box2D(pts):
    """
    bounding box for the points pts
    """
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[0]


def get_srtm_tif_name(lat, lon):
    """Download srtm tiles"""
    # longitude: [1, 72] == [-180, +180]
    tlon = (1+np.floor((lon+180)/5)) % 72
    tlon = 72 if tlon == 0 else tlon

    # latitude: [1, 24] == [60, -60]
    tlat = 1+np.floor((60-lat)/5)
    tlat = 24 if tlat == 25 else tlat

    srtm = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_%02d_%02d.zip" % (tlon, tlat)
    return srtm


def download_and_extract_dem(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, os.path.basename(url))
    
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Downloaded: {zip_path}")
    else:
        raise Exception(f"Failed to download {url}")
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        extracted_files = zip_ref.namelist()
    
    # Find the DEM file (assuming it's a .tif file)
    dem_files = [os.path.join(output_dir, f) for f in extracted_files if f.lower().endswith('.tif')]
    
    if not dem_files:
        raise Exception("No DEM file found in the extracted contents")
    
    print(f"Extracted DEM: {dem_files[0]}")
    return dem_files[0]


def kml_roi_process(file, kml):
    """
    """
    rpc = rpcm.rpc_from_geotiff(file)
    # extract lon lat from kml
    f = open(kml, 'r')
    a = bs4.BeautifulSoup(f, "lxml").find_all('coordinates')[0].text.split()
    f.close()
    #ll_bbx = np.array([list(map(float, x.split(','))) for x in a])
    #print(ll_bbx)
    ll_bbx = np.array([list(map(float, x.split(','))) for x in a])[:4, :2]

    # save lon lat bounding box to cfg dictionary
    lon_min = min(ll_bbx[:, 0])
    lon_max = max(ll_bbx[:, 0])
    lat_min = min(ll_bbx[:, 1])
    lat_max = max(ll_bbx[:, 1])
    #cfg['ll_bbx'] = (lon_min, lon_max, lat_min, lat_max)

    # convert lon lat bbox to utm
    z = utm.conversion.latlon_to_zone_number((lat_min + lat_max) * .5,
                                             (lon_min + lon_max) * .5)
    utm_bbx = np.array([utm.from_latlon(p[1], p[0], force_zone_number=z)[:2] for
                        p in ll_bbx])
    east_min = min(utm_bbx[:, 0])
    east_max = max(utm_bbx[:, 0])
    nort_min = min(utm_bbx[:, 1])
    nort_max = max(utm_bbx[:, 1])
    #cfg['utm_bbx'] = (east_min, east_max, nort_min, nort_max)

    # project lon lat vertices into the image
    if not isinstance(rpc, rpcm.rpc_model.RPCModel):
        rpc = rpcm.rpc_model.RPCModel(rpc)
    img_pts = [rpc.projection(p[0], p[1], rpc.alt_offset)[:2] for p in ll_bbx]

    # return image roi
    x, y, w, h = bounding_box2D(img_pts)
    return {'x': x, 'y': y, 'w': w, 'h': h}

def get_rmse(output_dir):
    stats_json = os.path.join(output_dir, 'comparison', 'stats', 'alti-diff', 'global', 'stats_results.json')
    with open(stats_json) as f:
        stats = json.load(f)
    rmse = stats['0']['rmse']
    return rmse