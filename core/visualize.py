import cv2
import os
import shutil

def save_clusters(cluster_dict, metadata, image_folder="data/images", output_dir="outputs"):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for person_id, indices in cluster_dict.items():
        cluster_dir = os.path.join(output_dir, person_id)
        os.makedirs(cluster_dir, exist_ok=True)

        for j, idx in enumerate(indices):
            data = metadata[idx]

            img_path = os.path.join(image_folder, data["image"])
            image = cv2.imread(img_path)

            if image is None:
                continue

            x1, y1, x2, y2 = data["bbox"]

            h, w, _ = image.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            filename = f"{j}.jpg"
            path = os.path.join(cluster_dir, filename)

            cv2.imwrite(path, face_crop)