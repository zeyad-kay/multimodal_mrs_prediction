from mrs_prediction.utils import download_models_from_wandb, parse_download_args
import dotenv
import pathlib

if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_download_args()
    download_models_from_wandb(args.runs, pathlib.Path(args.save_dir))