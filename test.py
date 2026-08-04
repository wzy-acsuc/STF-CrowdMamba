import argparse
import multiprocessing

from utils.tester import Tester
from utils.seed import set_random_seed

parser = argparse.ArgumentParser(description='STF-Mamba single-checkpoint evaluation')
parser.add_argument('-dataset_path', metavar='DATASET', help='path to dataset', default="../bccmfvu/bus")
parser.add_argument('-pth_path', metavar='PTH', help='path to pth file', default="./wight/bus_t6.pth")
parser.add_argument('-gpu', metavar='GPU', type=str, help='gpu id to use', default="0")
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path',
                    default='../bccmfvu/bus/bus_roi.npy')
parser.add_argument('-ig', '--is_gray', metavar='IG', type=bool, default=False, help='is gray')
parser.add_argument('-fn', '--frame_num', metavar='FN', type=int, default=6, help='frame num')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=4)
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
    tester = Tester(args)
    tester.execute()
