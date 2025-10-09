import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument(
        "--data_path",
        required=True,
        type=str,
    )
parser.add_argument(
        "--adaptive_mask",
        action="store_true",
        help='whether use adaptive attention reweighting',
    )
parser.add_argument(
        "--gpu_id",
        type=int, default=0,
        help="whether use ht encoder",
    )
parser.add_argument(
    "--clssname","-c",
    type=str,
    required=True,
    default="",
)
parser.add_argument(
    "--anomalyname","-a",
    type=str,
    required=True,
    default="",
)
opt=parser.parse_args()
root_dir=opt.data_path
clssname=opt.clssname
anomalyname=opt.anomalyname

os.system(
        'CUDA_VISIBLE_DEVICES=%d python generate_with_mask.py --data_root=%s --sample_name=%s --anomaly_name=%s --matching --adaptive_mask' % (
        opt.gpu_id, opt.data_path, clssname, anomalyname))

# For generating texture anomaly (e.g., color in wood), set 'adaptive_mask' to be True to use Adaptive attention reweighting by '--adaptive_mask' in generate_with_mask.py
# For generating structual anomaly (e.g., squeeze in capsule), set 'adaptive_mask' to be False by removing '--adaptive_mask' in generate_with_mask.py