import argparse

from configs import INFERENCE_CONFIG

from sap_rpt_oss.rpt import SAP_RPT_OSS_Regressor

configs = INFERENCE_CONFIG


def get_args():
    parser = argparse.ArgumentParser(description="Run inference with SAP RPT-OSS")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="选取一个checkpoint进行推理，不指定则使用configs中指定的所有checkpoint",
    )
    return parser.parse_args()


def main():
    args = get_args()
    checkpoints = configs.checkpoints_path.rglob("*.ckpt")
    inputs = configs.input_root_path.glob("*.csv")

    if args.checkpoint:
        model = SAP_RPT_OSS_Regressor(checkpoint=args.checkpoint)
    else:
        pass
