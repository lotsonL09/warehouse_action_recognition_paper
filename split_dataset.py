from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

root= Path.cwd()

dataset_dir=root/"datasets"/"r3d_dataset_tensor"

destiny_dir = root/"datasets"/"dataset_split_r3d_tensor"


def create_dataset_split(target_dir:Path,destiny_dir:Path):

    destiny_dir.mkdir(exist_ok=True)

    train_dir=destiny_dir/"train"
    val_dir=destiny_dir/"val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    for person in dataset_dir.iterdir():
        for activity in person.iterdir():

            video_tensors=list(activity.glob("*.pt"))

            train_videos,val_videos=train_test_split(video_tensors,test_size=0.75)

            dst_train_activity_dir=train_dir/activity.name

            dst_val_activity_dir=val_dir/activity.name

            dst_train_activity_dir.mkdir(exist_ok=True)

            dst_val_activity_dir.mkdir(exist_ok=True)

            for train_video in train_videos:

                dst_file_name=f"{person.name}_{train_video.name.split('.')[0]}.pt"

                dst_file_dir=dst_train_activity_dir/dst_file_name

                shutil.copy(train_video,dst_file_dir)             
            
            for val_video in val_videos:

                dst_file_name=f"{person.name}_{val_video.name.split('.')[0]}.pt"

                dst_file_dir=dst_val_activity_dir/dst_file_name

                shutil.copy(val_video,dst_file_dir) 
            


if __name__ == "__main__":
    create_dataset_split(dataset_dir,destiny_dir)

