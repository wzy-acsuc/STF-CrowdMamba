import os
from pathlib import Path
import argparse
import multiprocessing
from utils.tester import Tester
from utils.seed import set_random_seed
import csv

"""       ig       is_train       vd
ucsd     True   train val test  'ucsd'   resize to 952 × 632
canteen  False   True   False   'None'   
canteen  False   True   False   'None'
class    False   True   False   'None'
canteen  False   True   False   'None'
canteen  False  train val test  'canteen'   resize to 640 x 360
venice   False  train val test  'venice' 
"""

parser = argparse.ArgumentParser(description='STF-Mamba multi-checkpoint evaluation')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('wights_path', metavar='WP', type=str, help='saved path folder')
parser.add_argument('-gpu', metavar='GPU', type=str, help='gpu id to use', default='0')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('mask_path', metavar='MASK', type=str, help='roi mask path')
parser.add_argument('-ig', '--is_gray', metavar='IG', type=bool, default=False, help='is gray')
parser.add_argument('-fn', '--frame_num', metavar='FN', type=int, default=6, help='frame num')
parser.add_argument('-nw', '--num_workers', metavar='NW', type=int, help='number of workers', default=6)
parser.add_argument('--seed', metavar='SEED', type=int, default=3407,
                    help='random seed for testing (default: 3407)')
parser.add_argument('--ms-stmb-depth-1', '--ms_stmb_depth_1', dest='ms_stmb_depth_1', type=int, default=2,
                    help='number of MS-STMB blocks in stage 1 (default: 2)')
parser.add_argument('--ms-stmb-depth-2', '--ms_stmb_depth_2', dest='ms_stmb_depth_2', type=int, default=2,
                    help='number of MS-STMB blocks in stage 2 (default: 2)')
parser.add_argument('--sfrb-depth-1', '--sfrb_depth_1', dest='sfrb_depth_1', type=int, default=1,
                    help='number of SFRB blocks in stage 1 (default: 1)')
parser.add_argument('--sfrb-depth-2', '--sfrb_depth_2', dest='sfrb_depth_2', type=int, default=1,
                    help='number of SFRB blocks in stage 2 (default: 1)')

if __name__ == '__main__':
    args = parser.parse_args()
    set_random_seed(args.seed)
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)

    saved_path = Path(args.wights_path)
    wights = os.listdir(saved_path)

    mae_list = []
    min_wight = ''
    min_mae = 1000
    min_rmse = 1000
    for wight in wights:
        if os.path.splitext(wight)[-1] in ['.log', '.tar']:
            pass
        else:
            wight_path = os.path.join(saved_path, wight)
            args.pth_path = wight_path
            tester = Tester(args)
            mae, r_mse = tester.execute()
            if mae.item() < min_mae:
                min_mae = mae.item()
                min_wight = wight
                min_rmse = r_mse
            mae_list.append({wight: f'mae:{mae.item()}  rmse:{r_mse}'})
    with open(f'{saved_path}/result.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        for res in mae_list:
            writer.writerow([res])
        writer.writerow(['--------------------------------------------------------------------------------------'])
        writer.writerow([f'best_result: {min_wight}:  mae:{min_mae}   rmse:{min_rmse}'])
