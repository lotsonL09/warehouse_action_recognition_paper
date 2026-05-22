from utils.helper_functions import process_dataset
from pathlib import Path

if __name__=="__main__":
    root=Path.cwd()
    input_dir=root/"datasets"/"new_warehouse_actions"
    output_dir=root/"datasets"/"resized_warehouse_actions"
    output_dir.mkdir(exist_ok=True)
    process_dataset(input_dir=input_dir,output_dir=output_dir)