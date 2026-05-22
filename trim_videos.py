import cv2
from pathlib import Path

'''
William *
Jamil *
Gabriel * ------------
alejandro * -----------
fernando
medina
karlo
'''


root = Path.cwd()

datasets_dir=root/"datasets"

dataset_dir = datasets_dir/"original_videos"

###     CAMBIAR AQUI      ###
input_video=dataset_dir/"videos_cortados_gabriel_1.mp4"
###

output_dir=datasets_dir/"trimmed_videos"/"medina"


cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_clip = int(fps * 5)

frame_count = 0
clip_index = 0

writer = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frames_per_clip == 0:

        if writer is not None:
            writer.release()

        height, width, _ = frame.shape

        output_path = output_dir / f"clip_{clip_index:03d}.mp4"

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        clip_index += 1

    writer.write(frame)

    frame_count += 1

if writer is not None:
    writer.release()

cap.release()

print("Clips generados.")

