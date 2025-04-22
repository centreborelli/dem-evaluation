import numpy as np
import subprocess
import os
import json
import time
from glob import glob
import sys
import multiprocessing
import rasterio
from rasterio import Affine
from typing import List, Tuple, Union
import datetime
# from osgeo import gdal
import zipfile
import requests
import rpcm
import bs4
import utm
from s2p import common

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def convert_pix_to_coord(
    transform_array: Union[List, np.ndarray],
    row: Union[float, int, np.ndarray],
    col: Union[float, int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert input (row, col) pixels to dataset geographic coordinates
    from affine rasterio transform in upper left convention.
    See: https://gdal.org/tutorials/geotransforms_tut.html

    :param transform_array: Array containing 6 Affine Geo Transform coefficients
    :type transform_array: List or np.ndarray
    :param row: row to convert
    :type row: float, int or np.ndarray
    :param col: column to convert
    :type col: float, int or np.ndarray
    :return: converted x,y in geographic coordinates from affine transform
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Obtain the dataset transform in affine format from coefficients
    transform = Affine.from_gdal(
        transform_array[0],
        transform_array[1],
        transform_array[2],
        transform_array[3],
        transform_array[4],
        transform_array[5],
    )
    # Set the offset to ul (upper left)
    # Transform the input pixels to dataset geographic coordinates
    x, y = rasterio.transform.xy(transform, row, col, offset="ul")

    if not isinstance(x, int):
        x = np.array(x)
        y = np.array(y)

    return x, y

def compute_gdal_translate_bounds(
    y_offset: Union[float, int, np.ndarray],
    x_offset: Union[float, int, np.ndarray],
    shape: Tuple[int, int],
    georef_transform: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Obtain the gdal coordinate bounds to apply the translation offsets to
    the DEM to coregister/translate with gdal.

    The offsets can be applied with the command line:
    gdal_translate -a_ullr <ulx> <uly> <lrx> <lry>
    /path_to_original_dem.tif /path_to_coregistered_dem.tif

    :param y_offset: y pixel offset
    :type y_offset: Union[float, int, ndarray]
    :param x_offset: x pixel offset
    :type x_offset: Union[float, int, ndarray]
    :param shape: rasterio tuple containing x size and y size
    :type shape: Tuple[int, int]
    :param georef_transform: Array with 6 Affine Geo Transform coefficients
    :type georef_transform: np.ndarray
    :return: coordinate bounds to apply the offsets
    :rtype: Tuple[float,float,float,float]
    """
    # Read original secondary dem
    ysize, xsize = shape
    # Compute the coordinates of the new bounds
    x_0, y_0 = convert_pix_to_coord(georef_transform, y_offset, x_offset)
    x_1, y_1 = convert_pix_to_coord(
        georef_transform, y_offset + ysize, x_offset + xsize
    )

    return float(x_0), float(y_0), float(x_1), float(y_1)


def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata,
                       gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    pred_rdsm_path = "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

    # read dsm metadata
    xoff, yoff = dsm_metadata['ModelTiepointTag'][3], dsm_metadata['ModelTiepointTag'][4]
    xsize = dsm_metadata['ImageWidth']
    ysize = dsm_metadata['ImageLength']
    resolution = dsm_metadata['ModelPixelScaleTag'][0]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    ds = gdal.Open(in_dsm_path)
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    ds = None

    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            pred_dsm[water_mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    def count_nan(dsm):
        c = 0
        import numpy as np
        for i in range(dsm.shape[0]):
            for j in range(dsm.shape[1]):
                if np.isnan(dsm[i, j]):
                    c += 1
        return c, c / dsm.size

    nan_gt, pct_gt = count_nan(gt_dsm)
    nan_pred, pct_pred = count_nan(pred_dsm)
    # print('gt_dsm', nan_gt, pct_gt)
    # print('pred', nan_pred, pct_pred)

    # register and compute mae
    fix_xy = False
    try:
        import dsmr
    except:
        print("Warning: dsmr not found ! DSM registration will only use the Z dimension")
        fix_xy = True
    if fix_xy:
        pred_rdsm = pred_dsm + np.nanmean((gt_dsm - pred_dsm).ravel())
        with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
            dst.write(pred_rdsm, 1)
    else:
        transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
        dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
        with rasterio.open(pred_rdsm_path, "r") as f:
            pred_rdsm = f.read()[0, :, :]
    err = pred_rdsm - gt_dsm

    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(err, 1)

    return err


def compute_mae(pred_dsm_path, gt_dsm_path):

    with tifffile.TiffFile(gt_dsm_path) as t:
        metadata = {}
        for tag in t.pages[0].tags.values():
            name, value = tag.name, tag.value
            metadata[name] = value

    diff = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, metadata,
                              gt_mask_path=None, out_rdsm_path=None,
                              out_err_path=None)

    return np.nanmean(abs(diff.ravel()))


##########################################################################################################""

##### UTILS FUNCTIONS #####

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
    x, y, w, h = common.bounding_box2D(img_pts)
    return {'x': x, 'y': y, 'w': w, 'h': h}

def get_rmse(output_dir):
    stats_json = os.path.join(output_dir, 'comparison', 'stats', 'alti-diff', 'global', 'stats_results.json')
    with open(stats_json) as f:
        stats = json.load(f)
    rmse = stats['0']['rmse']
    return rmse


##### FUNCTIONS TO RUN PIPELINES #####

def run_s2p(path_to_config, path_to_logfile=None):
    if path_to_logfile is not None:
        # Write logs to the specified file
        with open(path_to_logfile, 'w') as f:
            subprocess.run(['s2p', path_to_config], stdout=f, stderr=f)
    else:
        # Write logs to stdout
        subprocess.run(['s2p', path_to_config], stdout=sys.stdout, stderr=sys.stderr)


def run_cars(path_to_config, path_to_logfile=None):
    if path_to_logfile is not None:
        # Write logs to the specified file
        with open(path_to_logfile, 'w') as f:
            subprocess.run(['cars', path_to_config], stdout=f, stderr=f)
    else:
        # Write logs to stdout
        subprocess.run(['cars', path_to_config], stdout=sys.stdout, stderr=sys.stderr)


def run_asp(path_img1, path_img2, output_dir, path_to_logfile,
            path_rpc_1=None, path_rpc_2=None, roi_left=None, roi_right=None,
            matching_algo='asp_mgm'):

    output_dir = os.path.join(output_dir, 'run')

    if not (path_rpc_1 is None or path_rpc_1 == ''):
        if roi_left is None and roi_right is None:
            with open(path_to_logfile, 'w') as f:
                subprocess.run([
                    'parallel_stereo',
                    '-t', 'rpc',
                    '--stereo-algorithm', matching_algo,
                    '--alignment-method', 'homography',
                    path_img1, path_img2,
                    path_rpc_1, path_rpc_2,
                    os.path.join(output_dir, 'run')
                ], stdout=f, stderr=f
                )
        else:
            with open(path_to_logfile, 'w') as f:
                subprocess.run([
                    'parallel_stereo',
                    '-t', 'rpc',
                    '--stereo-algorithm', matching_algo,
                    '--alignment-method', 'homography',
                    '--left-image-crop-win', str(roi_left['x']), str(roi_left['y']), str(roi_left['w']), str(roi_left['h']),
                    '--right-image-crop-win', str(roi_right['x']), str(roi_right['y']), str(roi_right['w']), str(roi_right['h']),
                    path_img1, path_img2,
                    path_rpc_1, path_rpc_2,
                    os.path.join(output_dir, 'run')
                ], stdout=f, stderr=f
                )

    else:
        if roi_left is None and roi_right is None:
            with open(path_to_logfile, 'w') as f:
                subprocess.run([
                    'parallel_stereo',
                    '-t', 'rpc',
                    '--stereo-algorithm', matching_algo,
                    '--alignment-method', 'homography',
                    path_img1, path_img2,
                    os.path.join(output_dir, 'run')
                ], stdout=f, stderr=f
                )
        else:
            with open(path_to_logfile, 'w') as f:
                subprocess.run([
                    'parallel_stereo',
                    '-t', 'rpc',
                    '--stereo-algorithm', matching_algo,
                    '--alignment-method', 'homography',
                    '--left-image-crop-win', str(roi_left['x']), str(roi_left['y']), str(roi_left['w']), str(roi_left['h']),
                    '--right-image-crop-win', str(roi_right['x']), str(roi_right['y']), str(roi_right['w']), str(roi_right['h']),
                    path_img1, path_img2,
                    os.path.join(output_dir, 'run')
                ], stdout=f, stderr=f
                )

    with open(path_to_logfile, 'a') as f:
        subprocess.run([
            'point2dem',
            # '--stereographic',
            '--t_srs', 'auto',
            os.path.join(output_dir, 'run-PC.tif')
        ], stdout=f, stderr=f
        )

# def build_parallel_stereo_command(path_img1, path_img2, output_dir, matching_algo,
#                                   path_rpc_1=None, path_rpc_2=None, roi_left=None, roi_right=None):
#     """Constructs the parallel_stereo command based on given parameters."""
#     command = [
#         'parallel_stereo',
#         '-t', 'rpc',
#         '--stereo-algorithm', matching_algo,
#         '--alignment-method', 'homography'
#     ]

#     # Add cropping windows if provided
#     if roi_left and roi_right:
#         command.extend([
#             '--left-image-crop-win', str(roi_left['x']), str(roi_left['y']), str(roi_left['w']), str(roi_left['h']),
#             '--right-image-crop-win', str(roi_right['x']), str(roi_right['y']), str(roi_right['w']), str(roi_right['h'])
#         ])

#     # Add images and RPCs if available
#     command.extend([path_img1, path_img2])
#     if path_rpc_1 and path_rpc_2:
#         command.extend([path_rpc_1, path_rpc_2])

#     command.append(os.path.join(output_dir, 'run'))

#     return command


# def run_asp(path_img1, path_img2, output_dir, path_to_logfile,
#             path_rpc_1=None, path_rpc_2=None, roi_left=None, roi_right=None,
#             matching_algo='asp_mgm'):
#     """Runs parallel_stereo and point2dem with logging."""
    
#     output_dir = os.path.join(output_dir, 'run')
#     os.makedirs(output_dir, exist_ok=True) 

#     # Construct and execute parallel_stereo command
#     stereo_command = build_parallel_stereo_command(
#         path_img1, path_img2, output_dir, matching_algo, path_rpc_1, path_rpc_2, roi_left, roi_right
#     )

#     with open(path_to_logfile, 'w') as f:
#         subprocess.run(stereo_command, stdout=f, stderr=f)

#         # Run point2dem
#         subprocess.run([
#             'point2dem',
#             # '--stereographic',
#             '--t_srs', 'auto',
#             os.path.join(output_dir, 'run-PC.tif')
#           ], stdout=f, stderr=f
#         )


def run_comparison(path_to_config, path_to_logfile=None):
    print('Running Comparison')
    if path_to_logfile is not None:
        # Write logs to the specified file
        with open(path_to_logfile, 'w') as f:
            subprocess.run(['demcompare', path_to_config], stdout=f, stderr=f)
    else:
        # Write logs to stdout
        subprocess.run(['demcompare', path_to_config], stdout=sys.stdout, stderr=sys.stderr)


### RUN PIPELINES AND COMPARISON ###

def eval_cars(path_img1, path_img2, path_gt_dsm, output_dir, roi=None,
              path_rpc_1=None, path_rpc_2=None, logfile=True,
              initial_elevation=False, pct_max_processes=0.9, 
              matching_algo='census_sgm', run_comparison_=True, compute_ambiguity=False,
              path_to_dsm=None, return_pct_nodata=True):

    cfg = dict()
    cfg['inputs'] = dict()
    cfg['inputs']['sensors'] = dict()
    cfg['inputs']['sensors']['one'] = dict()
    cfg['inputs']['sensors']['two'] = dict()
    cfg['applications'] = dict()
    cfg['applications']['dense_matching'] = dict()
    cfg['output'] = dict()
    cfg['orchestrator'] = dict()

    cfg['inputs']['sensors']['one']['image'] = path_img1
    cfg['inputs']['sensors']['two']['image'] = path_img2
    cfg['applications']['dense_matching']['method'] = matching_algo
    cfg['output']['directory'] = output_dir

    if compute_ambiguity:
        cfg['output']['auxiliary'] = {"mask": True, "classification": True, "performance_map": True, "contributing_pair": True, "filling": True, "ambiguity": True}

    if path_rpc_1 is not None:
        cfg['inputs']['sensors']['one']['geomodel'] = {
            "path": path_rpc_1,
            "model_type": "RPC"
        }
    if path_rpc_2 is not None:
        cfg['inputs']['sensors']['two']['geomodel'] = {
            "path": path_rpc_2,
            "model_type": "RPC"
        }

    if roi is not None:
        cfg['inputs']['roi'] = roi

    cfg['orchestrator']['mode'] = 'multiprocessing'
    cpu_count = multiprocessing.cpu_count()
    cfg['orchestrator']['nb_workers'] = int(pct_max_processes*cpu_count)

    if initial_elevation:
        lon, lat = rasterio.open(path_gt_dsm).lnglat()
        dem_init_url = get_srtm_tif_name(lat, lon)
        output_directory_dem_init = os.path.join(output_dir, 'init_dem')
        init_dem_path = download_and_extract_dem(dem_init_url, output_directory_dem_init)
        cfg['inputs']['initial_elevation'] = dict()
        cfg['inputs']['initial_elevation']['dem'] = init_dem_path

    tmp_path_to_config = os.path.join(FILE_DIR, 'configs/cars_tmp.json')
    with open(tmp_path_to_config, 'w') as f:
        json.dump(cfg, f)

    if path_to_dsm is None:
    # DSM to compute or re-compute
        if logfile:
            # path_to_logfile = os.path.join(FILE_DIR, 'configs/cars_logfile.txt')
            path_to_logfile = os.path.join(output_dir, 'logfile.txt')
            os.makedirs(output_dir, exist_ok=True)
            print('Logfile at', path_to_logfile)
        else:
            path_to_logfile = None

        start = time.time()
        run_cars(tmp_path_to_config, path_to_logfile)
        stop = time.time()
        total_time = stop - start
    else:
        total_time = None

    if path_to_dsm is None:
        path_to_pred = os.path.join(output_dir, 'dsm', 'dsm.tif')
    else:
        path_to_pred = path_to_dsm

    if return_pct_nodata:
        dsm = rasterio.open(path_to_pred).read()
        nb_nodata = np.isnan(dsm).sum()
        pct_nodata = nb_nodata / dsm.size
    else:
        pct_nodata = None

    if path_gt_dsm is None or not run_comparison_:
        return total_time, pct_nodata

    path_to_cmp = os.path.join(FILE_DIR, 'configs/compare_default.json')
    with open(path_to_cmp, 'r') as f:
        compare_cfg = json.load(f)

    compare_output_dir = os.path.join(output_dir, 'comparison')
    compare_cfg['output_dir'] = compare_output_dir

    compare_cfg['input_ref']['path'] = path_to_pred
    compare_cfg['input_sec']['path'] = path_gt_dsm

    tmp_path_to_cmp_config = os.path.join(FILE_DIR, 'configs/compare_tmp.json')
    with open(tmp_path_to_cmp_config, 'w') as f:
        json.dump(compare_cfg, f)

    run_comparison(tmp_path_to_cmp_config)

    os.remove(tmp_path_to_config)
    os.remove(tmp_path_to_cmp_config)
    # if initial_elevation:
    #     # remove the downloaded srtm initial elevation dem
    #     os.system(f'rm -rf {init_dem_path}')
    rmse = get_rmse(output_dir)
    return total_time, rmse, pct_nodata



def eval_s2p(path_img1, path_img2, path_gt_dsm, output_dir, roi=None,
             use_srtm=False, dsm_resolution=0.5, gpu_total_memory=None, 
             rectification_method='rpc', census_ncc_win=5, logfile=True,
             tile_size=200, pct_max_processes=0.9, matching_algo='mgm',
             timeout=600, postprocess_stereosgm_gpu=True, vertical_margin=5, **kwargs):

    cfg = dict()
    cfg['out_dir'] = output_dir
    cfg['use_srtm'] = use_srtm
    cfg['matching_algorithm'] = matching_algo
    cfg['images'] = [dict(), dict()]
    cfg['images'][0]['img'] = path_img1
    cfg['images'][1]['img'] = path_img2
    cfg['dsm_resolution'] = dsm_resolution
    cfg['rectification_method'] = rectification_method
    cfg['census_ncc_win'] = census_ncc_win
    cfg['timeout'] = timeout
    cfg['mgm_timeout'] = timeout

    cfg['postprocess_stereosgm_gpu'] = postprocess_stereosgm_gpu
    cfg['vertical_margin'] = vertical_margin

    # add remaining parameters
    cfg.update(kwargs)

    if roi is not None:
        cfg['roi'] = dict()
        cfg['roi']['x'] = roi['x']
        cfg['roi']['y'] = roi['y']
        cfg['roi']['w'] = roi['w']
        cfg['roi']['h'] = roi['h']
        cfg['full_img'] = False
    else:
        cfg['full_img'] = True

    if 'gpu' in matching_algo:
        print('Using GPU')
        if gpu_total_memory is None:
            gpu_total_memory = get_gpu_memory()
        cfg['gpu_total_memory'] = gpu_total_memory
    
    cfg['tile_size'] = tile_size
    cpu_count = multiprocessing.cpu_count()
    # cfg['max_processes_stereo_matching'] = int(pct_max_processes*cpu_count)
    cfg['max_processes'] = int(pct_max_processes*cpu_count)

    tmp_path_to_config = os.path.join(FILE_DIR, 'configs/s2p_tmp.json')
    with open(tmp_path_to_config, 'w') as f:
        json.dump(cfg, f)

    path_to_logfile = os.path.join(output_dir, 'logfile.txt') if logfile else None
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    run_s2p(tmp_path_to_config, path_to_logfile)
    stop = time.time()
    total_time = stop-start
    
    dsm = rasterio.open(os.path.join(output_dir, 'dsm.tif')).read()
    nb_nodata = np.isnan(dsm).sum()
    pct_nodata = nb_nodata / dsm.size

    if os.path.exists(os.path.join(output_dir, 'dsm-filtered.tif')):
        dsm = rasterio.open(os.path.join(output_dir, 'dsm-filtered.tif')).read()
        nb_nodata_filtered = np.isnan(dsm).sum()
        pct_nodata_filtered = nb_nodata_filtered / dsm.size
    else:
        pct_nodata_filtered = None

    if path_gt_dsm is None:
        return total_time, pct_nodata, pct_nodata_filtered

    path_to_cmp = os.path.join(FILE_DIR, 'configs/compare_default.json')
    with open(path_to_cmp, 'r') as f:
        compare_cfg = json.load(f)

    compare_output_dir = os.path.join(output_dir, 'comparison')
    compare_cfg['output_dir'] = compare_output_dir

    path_to_pred = os.path.join(output_dir, 'dsm.tif')
    print('path_to_pred comparison', path_to_pred)

    nodata_gt_dsm = rasterio.open(path_gt_dsm).nodata
    if nodata_gt_dsm is None:
        nodata_gt_dsm = np.nan

    nodata_pred_dsm = rasterio.open(path_to_pred).nodata
    if nodata_pred_dsm is None:
        nodata_pred_dsm = np.nan

    compare_cfg['input_ref']['path'] = path_gt_dsm
    compare_cfg['input_ref']['nodata'] = nodata_gt_dsm
    compare_cfg['input_sec']['path'] = path_to_pred
    compare_cfg['input_sec']['nodata'] = nodata_pred_dsm

    tmp_path_to_cmp_config = os.path.join(FILE_DIR, 'configs/compare_tmp.json')
    with open(tmp_path_to_cmp_config, 'w') as f:
        json.dump(compare_cfg, f)

    run_comparison(tmp_path_to_cmp_config)

    if os.path.exists(os.path.join(output_dir, 'dsm-filtered.tif')):
        path_to_cmp = os.path.join(FILE_DIR, 'configs/compare_default.json')
        with open(path_to_cmp, 'r') as f:
            compare_cfg = json.load(f)

        compare_output_dir = os.path.join(output_dir, 'comparison_filtered')
        compare_cfg['output_dir'] = compare_output_dir

        path_to_pred = os.path.join(output_dir, 'dsm-filtered.tif')
        print('path_to_pred comparison', path_to_pred)
        compare_cfg['input_ref']['path'] = path_gt_dsm
        compare_cfg['input_ref']['nodata'] = nodata_gt_dsm
        compare_cfg['input_sec']['path'] = path_to_pred
        compare_cfg['input_sec']['nodata'] = nodata_pred_dsm

        tmp_path_to_cmp_config = os.path.join(FILE_DIR, 'configs/compare_tmp.json')
        with open(tmp_path_to_cmp_config, 'w') as f:
            json.dump(compare_cfg, f)

        run_comparison(tmp_path_to_cmp_config)


    os.remove(tmp_path_to_config)
    os.remove(tmp_path_to_cmp_config)

    rmse = get_rmse(output_dir)
    return total_time, rmse, pct_nodata, pct_nodata_filtered


def eval_asp(path_img1, path_img2, path_gt_dsm, output_dir, 
             roi_left=None, roi_right=None,
             path_rpc_1=None, path_rpc_2=None,
             matching_algo='asp_mgm'):
    
    """
    Matching algorithms available: asp_bm, asp_sgm, asp_mgm, asp_final_mgm, mgm, opencv_sgbm, libelas, msmw, msmw2, opencv_bm
    """

    # path_to_logfile = os.path.join(FILE_DIR, 'configs/asp_logfile.txt')
    path_to_logfile = os.path.join(output_dir, 'logfile.txt')
    os.makedirs(output_dir, exist_ok=True)
    print('logs', path_to_logfile)
    start = time.time()
    run_asp(path_img1, path_img2, output_dir, path_to_logfile,
            path_rpc_1=path_rpc_1, path_rpc_2=path_rpc_2, roi_left=roi_left, 
            roi_right=roi_right, matching_algo=matching_algo)
    stop = time.time()
    total_time = stop-start

    dsm = rasterio.open(os.path.join(output_dir, 'run', 'run-DEM.tif')).read()
    nb_nodata = np.isnan(dsm).sum()
    pct_nodata = nb_nodata / dsm.size

    if path_gt_dsm is None:
        return total_time, pct_nodata

    path_to_cmp = os.path.join(FILE_DIR, 'configs/compare_default.json')
    with open(path_to_cmp, 'r') as f:
        compare_cfg = json.load(f)

    compare_output_dir = os.path.join(output_dir, 'comparison')
    compare_cfg['output_dir'] = compare_output_dir

    dem_name = glob(os.path.join(output_dir, 'run', '*DEM.tif'))[0]
    compare_cfg['input_ref']['path'] = dem_name
    compare_cfg['input_sec']['path'] = path_gt_dsm

    tmp_path_to_cmp_config = os.path.join(FILE_DIR, 'configs/compare_tmp.json')
    with open(tmp_path_to_cmp_config, 'w') as f:
        json.dump(compare_cfg, f)

    run_comparison(tmp_path_to_cmp_config)

    os.remove(tmp_path_to_cmp_config)

    rmse = get_rmse(output_dir)
    return total_time, rmse, pct_nodata
