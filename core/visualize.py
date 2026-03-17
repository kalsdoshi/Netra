import cv2
import os
import shutil

def save_clusters(cluster_dict, metadata, output_dir="outputs"):
    # delete old output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for person_id, indices in cluster_dict.items():
        cluster_dir = os.path.join(output_dir, person_id)
        os.makedirs(cluster_dir, exist_ok=True)

        for j, idx in enumerate(indices):
            face_crop = metadata[idx]["face_crop"]

            if face_crop is None or face_crop.size == 0:
                continue

            filename = f"{j}.jpg"
            path = os.path.join(cluster_dir, filename)

            cv2.imwrite(path, face_crop)