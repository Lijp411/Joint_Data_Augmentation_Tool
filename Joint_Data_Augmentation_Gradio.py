import argparse
import sys
import gradio as gr
import os
import json
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import base64
import cv2

def parse_args(args):

    parser = argparse.ArgumentParser(description="Joint Image - Point Cloud Data Augmentation")
    # For pipeline 
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)

    return parser.parse_args(args)

def decode_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def rotate_bounding_box(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    new_x_min = y_min
    new_y_min = img_width - x_max
    new_x_max = y_max
    new_y_max = img_width - x_min
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def rotate_point_cloud(point_cloud, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    min_coords = np.min(point_cloud[:, :3], axis=0)
    max_coords = np.max(point_cloud[:, :3], axis=0)
    midpoint = (min_coords + max_coords) / 2
    translated_xyz = point_cloud[:, :3] - midpoint
    rotated_xyz = translated_xyz @ rotation_matrix.T
    rotated_xyz += midpoint

    return np.hstack((rotated_xyz, point_cloud[:, 3:]))

def rotate_polygon(points, angle, image_size):
    angle_rad = np.radians(angle)
    width, height = image_size
    center = np.array([width / 2, height / 2])
    points = np.array(points)
    relative_points = points - center
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)],
    ])
    rotated_points = relative_points @ rotation_matrix.T + center
    return rotated_points.tolist()

def rotate_augmentation(dir, output_dir, angle_list, direction):
    sample = 0
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", direction)

            for i, angle in enumerate([angle_list]):
                rotated_image = rotate_image(cv_image, angle)
                rotated_image_name = f"{base_name}_rot_{i}.png"
                rotated_image_path = os.path.join(output_dir, rotated_image_name)
                rotated_image.save(rotated_image_path)
                inst_data["imageData"] = encode_image(np.array(rotated_image)[..., ::-1])

                if "boundingBoxes" in inst_data:
                    img_width, img_height = rotated_image.size
                    rotated_bboxes = [
                        rotate_bounding_box(bbox, img_width, img_height)
                        for bbox in inst_data["boundingBoxes"]
                    ]
                    inst_data["boundingBoxes"] = rotated_bboxes

                rotated_point_cloud = rotate_point_cloud(point_cloud, angle)
                np.savetxt(os.path.join(output_dir, f"{base_name}_rot_{i}.txt"), rotated_point_cloud, fmt='%f')

                for shape in inst_data.get('shapes', []):
                    shape['points'] = rotate_polygon(shape['points'], -angle, image.size)

                rotated_json_path = os.path.join(output_dir, f"{base_name}_rot_{i}.json")
                with open(rotated_json_path, 'w') as f:
                    json.dump(inst_data, f, indent=4)

    return sample

def vertical_flip_image(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def vertical_flip_bounding_box(bbox, img_height):
    x_min, y_min, x_max, y_max = bbox
    new_y_min = img_height - y_max
    new_y_max = img_height - y_min
    return [x_min, new_y_min, x_max, new_y_max]

def vertical_flip_point_cloud(point_cloud):
    flipped_xyz = point_cloud.copy()
    flipped_xyz[:, 1] = -flipped_xyz[:, 1]  # Flip the Y coordinate
    return flipped_xyz

def vertical_flip_polygon(points, image_size):
    _, height = image_size
    flipped_points = [[x, height - y] for x, y in points]
    return flipped_points

def vertical_flip_augmentation(dir, output_dir):
    sample = 0
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "vertical_flip")

            # Flip image
            flipped_image = vertical_flip_image(cv_image)
            flipped_image_name = f"{base_name}.png"
            flipped_image_path = os.path.join(output_dir, flipped_image_name)
            flipped_image.save(flipped_image_path)
            inst_data["imageData"] = encode_image(np.array(flipped_image)[..., ::-1])

            # Flip bounding boxes
            if "boundingBoxes" in inst_data:
                img_width, img_height = flipped_image.size
                flipped_bboxes = [
                    vertical_flip_bounding_box(bbox, img_height) for bbox in inst_data["boundingBoxes"]
                ]
                inst_data["boundingBoxes"] = flipped_bboxes

            # Flip point cloud
            flipped_point_cloud = vertical_flip_point_cloud(point_cloud)
            np.savetxt(os.path.join(output_dir, f"{base_name}.txt"), flipped_point_cloud, fmt='%f')

            # Flip polygons
            for shape in inst_data.get('shapes', []):
                shape['points'] = vertical_flip_polygon(shape['points'], image.size)

            # Save flipped JSON
            flipped_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(flipped_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

    return sample

def horizontal_flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def horizontal_flip_bounding_box(bbox, img_width):
    x_min, y_min, x_max, y_max = bbox
    new_x_min = img_width - x_max
    new_x_max = img_width - x_min
    return [new_x_min, y_min, new_x_max, y_max]

def horizontal_flip_point_cloud(point_cloud):
    flipped_xyz = point_cloud.copy()
    flipped_xyz[:, 0] = -flipped_xyz[:, 0]  # Flip the X coordinate
    return flipped_xyz

def horizontal_flip_polygon(points, image_size):
    width, _ = image_size
    flipped_points = [[width - x, y] for x, y in points]
    return flipped_points

def horizontal_flip_augmentation(dir, output_dir):
    sample = 0 
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "horizontal_flip")

            # Flip image
            flipped_image = horizontal_flip_image(cv_image)
            flipped_image_name = f"{base_name}.png"
            flipped_image_path = os.path.join(output_dir, flipped_image_name)
            flipped_image.save(flipped_image_path)
            inst_data["imageData"] = encode_image(np.array(flipped_image)[..., ::-1])

            # Flip bounding boxes
            if "boundingBoxes" in inst_data:
                img_width, img_height = flipped_image.size
                flipped_bboxes = [
                    horizontal_flip_bounding_box(bbox, img_width) for bbox in inst_data["boundingBoxes"]
                ]
                inst_data["boundingBoxes"] = flipped_bboxes

            # Flip point cloud
            flipped_point_cloud = horizontal_flip_point_cloud(point_cloud)
            np.savetxt(os.path.join(output_dir, f"{base_name}.txt"), flipped_point_cloud, fmt='%f')

            # Flip polygons
            for shape in inst_data.get('shapes', []):
                shape['points'] = horizontal_flip_polygon(shape['points'], image.size)

            # Save flipped JSON
            flipped_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(flipped_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

    return sample

def blur_image(image, kernel_size=(5, 5)):
    img_array = np.array(image)
    blurred_array = cv2.blur(img_array, kernel_size)
    return Image.fromarray(blurred_array)

def blur_augmentation(dir, output_dir, kernel_size=(5, 5)):
    sample = 0 
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "blur")

            # Blur image
            blurred_image = blur_image(cv_image, kernel_size)
            blurred_image_name = f"{base_name}.png"
            blurred_image_path = os.path.join(output_dir, blurred_image_name)
            blurred_image.save(blurred_image_path)
            inst_data["imageData"] = encode_image(np.array(blurred_image)[..., ::-1])

            # Save unchanged point cloud
            np.savetxt(os.path.join(output_dir, f"{base_name}.txt"), point_cloud, fmt='%f')

            # Save unchanged JSON
            blurred_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(blurred_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

    return sample

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def brightness_augmentation(dir, output_dir, min_factor=0.9, max_factor=1.1):
    sample = 0
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "brightness")

            # Random brightness factor
            factor = np.random.uniform(min_factor, max_factor)
            adjusted_image = adjust_brightness(cv_image, factor)

            # Save adjusted image
            adjusted_image_name = f"{base_name}.png"
            adjusted_image_path = os.path.join(output_dir, adjusted_image_name)
            adjusted_image.save(adjusted_image_path)
            inst_data["imageData"] = encode_image(np.array(adjusted_image)[..., ::-1])

            # Save point cloud (unchanged)
            np.savetxt(os.path.join(output_dir, f"{base_name}.txt"), point_cloud, fmt='%f')

            # Save JSON file (unchanged)
            adjusted_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(adjusted_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

    return sample

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def contrast_augmentation(dir, output_dir, min_factor=0.7, max_factor=1.3):
    sample = 0
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            img_path = os.path.join(dir, img_file)
            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            inst_path = os.path.join(dir, f"{base_name}.json")

            # Load image
            image = Image.open(img_path)

            # Load corresponding files (point cloud and JSON)
            point_cloud = np.loadtxt(pc_path)
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "contrast")

            # Generate a random contrast factor
            factor = np.random.uniform(min_factor, max_factor)

            # Adjust image contrast
            contrast_image = adjust_contrast(cv_image, factor)

            # Save augmented image
            contrast_image_name = f"{base_name}.png"
            contrast_image_path = os.path.join(output_dir, contrast_image_name)
            contrast_image.save(contrast_image_path)

            # Update JSON with the new image data
            inst_data["imageData"] = encode_image(np.array(contrast_image)[..., ::-1])

            # Save augmented JSON and point cloud (unchanged)
            contrast_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(contrast_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

            contrast_pc_path = os.path.join(output_dir, f"{base_name}.txt")
            np.savetxt(contrast_pc_path, point_cloud, fmt='%f')

    return sample

def farthest_point_sampling(point_cloud, num_samples):
    num_points = point_cloud.shape[0]
    if num_points <= num_samples:
        return point_cloud

    sampled_idx = [np.random.randint(num_points)]
    distances = np.full(num_points, np.inf)

    for _ in range(num_samples - 1):
        last_sampled_point = point_cloud[sampled_idx[-1], :3]
        dist_to_last_point = np.linalg.norm(point_cloud[:, :3] - last_sampled_point, axis=1)
        distances = np.minimum(distances, dist_to_last_point)
        sampled_idx.append(np.argmax(distances))

    return point_cloud[sampled_idx]

    return 1

def random_sampling_point_cloud(point_cloud, drop_ratio):
    total_points = point_cloud.shape[0]
    sampled_indices = np.random.choice(total_points, int(total_points * (1 - drop_ratio)), replace=False)
    return point_cloud[sampled_indices]

def random_sampling_augmentation(dir, output_dir, drop_ratio):
    sample = 0
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            # Load image
            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            # Load point cloud
            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            # Load instance JSON
            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "random_sampling")

            # Save the original image (unchanged)
            sampling_image_name = f"{base_name}.png"
            sampling_image_path = os.path.join(output_dir, sampling_image_name)
            cv_image.save(sampling_image_path)
            inst_data["imageData"] = encode_image(np.array(cv_image)[..., ::-1])

            # Perform random sampling on the point cloud
            sampled_point_cloud = random_sampling_point_cloud(point_cloud, drop_ratio)
            sampled_pc_path = os.path.join(output_dir, f"{base_name}.txt")
            np.savetxt(sampled_pc_path, sampled_point_cloud, fmt='%f')

            # Save the instance JSON (unchanged)
            sampled_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(sampled_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

    return sample

def random_sampling_FPS_augmentation(dir, output_dir, drop_ratio, number_samples):
    sample = 0
    for img_file in os.listdir(dir):
        if img_file.endswith(".png"):
            sample += 1
            base_name = os.path.splitext(img_file)[0]

            # Load image
            img_path = os.path.join(dir, img_file)
            image = Image.open(img_path)

            # Load point cloud
            pc_path = os.path.join(dir, f"{base_name.split('_img')[0]}_pc.txt")
            point_cloud = np.loadtxt(pc_path)

            # Load instance JSON
            inst_path = os.path.join(dir, f"{base_name}.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)

            if "imageData" in inst_data:
                base64_image = inst_data["imageData"]
                decoded_image = decode_image(base64_image)
                cv_image = Image.fromarray(decoded_image[..., ::-1])
            else:
                cv_image = image

            base_name = base_name.replace("img", "resampling")

            # Save the original image (unchanged)
            sampling_image_name = f"{base_name}.png"
            sampling_image_path = os.path.join(output_dir, sampling_image_name)
            cv_image.save(sampling_image_path)
            inst_data["imageData"] = encode_image(np.array(cv_image)[..., ::-1])

            # Perform random sampling on the point cloud
            sampled_point_cloud = random_sampling_point_cloud(point_cloud, drop_ratio)
            sampled_point_cloud = farthest_point_sampling(sampled_point_cloud, number_samples)
            sampled_pc_path = os.path.join(output_dir, f"{base_name}.txt")
            np.savetxt(sampled_pc_path, sampled_point_cloud, fmt='%f')

            # Save the instance JSON (unchanged)
            sampled_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(sampled_json_path, 'w') as f:
                json.dump(inst_data, f, indent=4)

    return sample

def data_augmentation(data_dir, output_dir, augmentations, contrast_range, brightness_range, dropout_ratio):
    results = []
    total_sample = 0
    if "rotate_left" in augmentations:
        total_sample += rotate_augmentation(data_dir, output_dir, 90, "left")
    if "rotate_right" in augmentations:
        total_sample += rotate_augmentation(data_dir, output_dir, -90, "right")
    if "vertical_flip" in augmentations:
        total_sample += vertical_flip_augmentation(data_dir, output_dir)
    if "horizontal_flip" in augmentations:
        total_sample += horizontal_flip_augmentation(data_dir, output_dir)
    if "blur" in augmentations:
        total_sample += blur_augmentation(data_dir, output_dir, kernel_size=(5, 5))
    if "brightness" in augmentations:
        total_sample += brightness_augmentation(data_dir, output_dir, min_factor=(1.0 - brightness_range), max_factor=(1.0 + brightness_range))
    if "contrast" in augmentations:
        total_sample += contrast_augmentation(data_dir, output_dir, 1.0 - contrast_range, 1.0 + contrast_range)
    if "resampling" in augmentations:
        total_sample += random_sampling_FPS_augmentation(data_dir, output_dir, drop_ratio=0.95, number_samples=10000)
    if "random_sampling" in augmentations:
        total_sample += random_sampling_augmentation(data_dir, output_dir, drop_ratio=dropout_ratio)
    
    return f"\nThe total sample after augmentation is {total_sample}!"

def start_augmentation(data_dir, output_dir, augmentations, contrast_range, brightness_range, dropout_ratio):
    if not os.path.exists(data_dir):
        return "Input folder does not exist."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return data_augmentation(data_dir, output_dir, augmentations, contrast_range, brightness_range, dropout_ratio)

if __name__ == "__main__":

  global args
  args = parse_args(sys.argv[1:])

  with gr.Blocks(
    css="#pc_scene { height: 400px; width: 100%; } .button { height: 300px; width: 100px; }"
  ) as demo:

    gr.Markdown(
            """
            ## Joint Data Augmentation Tool: Mitigating the availability challenges for bimodal image-point cloud data 🚀
            ## If you found this demo useful, please consider starring 🌟 our github repo and open-source platform
            """
            )

    gr.Markdown(
            """
            ## Usage:
            1. Specify the path of input data and output data.
            2. Select the data augmentation method and adjust the corresponding parameters.
            3. Click **Start!** to obtain data augmentation results.
            """)


    data_dir_input = gr.Textbox(label="data_path", placeholder="e.g. /content/drive/MyDrive/IPCE-Net/test_tool/original_data")
    output_dir_input = gr.Textbox(label="output_path", placeholder="e.g. /content/drive/MyDrive/IPCE-Net/test_tool/output")

    augmentations_input = gr.CheckboxGroup(
        choices=[
            "rotate_left",
            "rotate_right",
            "vertical_flip",
            "horizontal_flip",
            "blur",
            "contrast",
            "brightness",
            "resampling",
            "random_sampling",
        ],
        label="Please select data augmentation method (multiple choices are supported)",
        value=["rotate_left", "contrast", "random_sampling"],
    )
    
    contrast_range = gr.Slider(
        label="Contrast stretch range", minimum=0.0, maximum=0.5, value=0.2, step=0.01
    )

    brightness_range = gr.Slider(
        label="Brightness stretch range", minimum=0.0, maximum=0.3, value=0.2, step=0.01
    )

    dropout_ratio = gr.Slider(
        label="Dropout ratio", minimum=0.0, maximum=1.0, value=0.3, step=0.01
    )

    run_button = gr.Button("Start!")

    output_text = gr.Textbox(label="Augmentation Results", interactive=False)

    run_button.click(
        fn=start_augmentation,
        inputs=[data_dir_input, output_dir_input, augmentations_input, contrast_range, brightness_range, dropout_ratio],
        outputs=output_text,
    )

    print("Completed!")

    demo.queue()
    demo.launch(share = True)

