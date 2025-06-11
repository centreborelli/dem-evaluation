import numpy as np
import subprocess
import os
import json
import time
from glob import glob
import sys
import multiprocessing
import rasterio

from utils import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))



##### FUNCTIONS TO RUN PIPELINES #####

def run_command(tool_name, config_path, log_path=None):
    """
    Runs a subprocess command with optional logging.

    Args:
        tool_name (str): The name of the tool to run.
        config_path (str): Path to the configuration file.
        log_path (str, optional): Path to a log file. If None, logs are printed to stdout.
    """
    command = [tool_name, config_path]

    if log_path:
        with open(log_path, 'w') as log_file:
            subprocess.run(command, stdout=log_file, stderr=log_file)
    else:
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)

def run_comparison(path_to_config, path_to_logfile=None):
    run_command('demcompare', path_to_config, path_to_logfile)

def run_s2p(path_to_config, path_to_logfile=None):
    run_command('s2p', path_to_config, path_to_logfile)

def run_cars(path_to_config, path_to_logfile=None):
    run_command('cars', path_to_config, path_to_logfile)


# functions to run asp #
def _build_parallel_stereo_command(img1, img2, out_dir, rpc1=None, rpc2=None,
                                   roi_left=None, roi_right=None, matching_algo='asp_mgm'):
    cmd = [
        'parallel_stereo',
        '-t', 'rpc',
        '--stereo-algorithm', matching_algo,
        '--alignment-method', 'homography'
    ]

    if roi_left and roi_right:
        cmd += [
            '--left-image-crop-win', str(roi_left['x']), str(roi_left['y']),
            str(roi_left['w']), str(roi_left['h']),
            '--right-image-crop-win', str(roi_right['x']), str(roi_right['y']),
            str(roi_right['w']), str(roi_right['h'])
        ]

    cmd += [img1, img2]

    if rpc1 and rpc2:
        cmd += [rpc1, rpc2]

    cmd.append(os.path.join(out_dir, 'run'))
    return cmd

def _run_command_asp(command, log_path, mode='w'):
    if log_path:
        with open(log_path, mode) as f:
            subprocess.run(command, stdout=f, stderr=f)
    else:
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)

def run_asp(path_img1, path_img2, output_dir, path_to_logfile,
            path_rpc_1=None, path_rpc_2=None, roi_left=None, roi_right=None,
            matching_algo='asp_mgm'):
    """
    Runs the ASP stereo pipeline with optional RPCs and ROIs.
    """
    run_dir = os.path.join(output_dir, 'run')
    os.makedirs(run_dir, exist_ok=True)

    # Run parallel_stereo
    stereo_cmd = _build_parallel_stereo_command(
        path_img1, path_img2, output_dir,
        rpc1=path_rpc_1, rpc2=path_rpc_2,
        roi_left=roi_left, roi_right=roi_right,
        matching_algo=matching_algo
    )
    _run_command_asp(stereo_cmd, path_to_logfile)

    # Run point2dem
    point2dem_cmd = [
        'point2dem',
        '--t_srs', 'auto',
        os.path.join(run_dir + '-PC.tif')
    ]
    _run_command_asp(point2dem_cmd, path_to_logfile, mode='a')


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

def compare_dsms(output_dir, path_to_pred, path_gt_dsm, return_pct_nodata, run_comparison_=True):

    if return_pct_nodata:
        dsm = rasterio.open(path_to_pred).read()
        nb_nodata = np.isnan(dsm).sum()
        pct_nodata = nb_nodata / dsm.size
    else:
        pct_nodata = None

    if path_gt_dsm is None or not run_comparison_:
        return None, pct_nodata

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

    os.remove(tmp_path_to_cmp_config)
    rmse = get_rmse(output_dir)
    return rmse, pct_nodata


def eval_cars(path_img1, path_img2, path_gt_dsm, output_dir, roi=None,
              path_rpc_1=None, path_rpc_2=None, logfile=True,
              initial_elevation=False, pct_max_processes=0.9, 
              matching_algo='census_sgm', run_comparison_=True, compute_ambiguity=False,
              path_to_dsm=None, return_pct_nodata=True, **kwargs):

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
        cfg['output']['auxiliary'] = {"mask": True,
                                      "classification": True,
                                      "performance_map": True,
                                      "contributing_pair": True,
                                      "filling": True,
                                      "ambiguity": True}

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

    rmse, pct_nodata = compare_dsms(output_dir, path_to_pred, path_gt_dsm, return_pct_nodata, run_comparison_=run_comparison_)

    if path_gt_dsm is None or not run_comparison_:
        return total_time, pct_nodata

    os.remove(tmp_path_to_config)

    return total_time, rmse, pct_nodata



def eval_s2p(path_img1, path_img2, path_gt_dsm, output_dir, roi=None,
             use_srtm=False, dsm_resolution=0.5, gpu_total_memory=None, 
             rectification_method='rpc', census_ncc_win=5, logfile=True,
             tile_size=200, pct_max_processes=0.9, matching_algo='mgm',
             timeout=600, postprocess_stereosgm_gpu=True, vertical_margin=5, 
             return_pct_nodata=True, **kwargs):
    
    output_dir = os.path.abspath(output_dir)
    print('outpur_dir', output_dir)

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

    # return_pct_nodata = os.path.exists(os.path.join(output_dir, 'dsm-filtered.tif'))
    if os.path.exists(os.path.join(output_dir, 'dsm-filtered.tif')):
        path_to_pred = os.path.join(output_dir, 'dsm-filtered.tif')
    else:
        os.path.join(output_dir, 'dsm.tif')
    run_comparison_ = path_gt_dsm is not None
    rmse, pct_nodata = compare_dsms(output_dir, path_to_pred, path_gt_dsm, return_pct_nodata, run_comparison_=run_comparison_)

    os.remove(tmp_path_to_config)

    rmse = get_rmse(output_dir)
    return total_time, rmse, pct_nodata


def eval_asp(path_img1, path_img2, path_gt_dsm, output_dir, 
             roi_left=None, roi_right=None,
             path_rpc_1=None, path_rpc_2=None,
             matching_algo='asp_mgm', return_pct_nodata=True):
    
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

    path_to_pred = glob(os.path.join(output_dir, 'run', '*DEM.tif'))[0]
    run_comparison_ = path_gt_dsm is not None
    rmse, pct_nodata = compare_dsms(output_dir, path_to_pred, path_gt_dsm, return_pct_nodata, run_comparison_=run_comparison_)

    rmse = get_rmse(output_dir)
    return total_time, rmse, pct_nodata
