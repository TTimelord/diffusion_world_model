import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

def extract_evenly_spaced_frames(video_path, output_folder, num_frames):
    """
    Extracts a specified number of evenly spaced frames from a video.

    :param video_path: Path to the input video file.
    :param output_folder: Directory where extracted frames will be saved.
    :param num_frames: Number of frames to extract.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames > total_frames:
        print(f"Error: Requested {num_frames} frames, but video only has {total_frames} frames.")
        cap.release()
        return

    # Calculate interval between frames to extract
    interval = total_frames // num_frames

    extracted_count = 0
    for i in range(num_frames):
        frame_number = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
            cv2.imwrite(output_path, frame)
            extracted_count += 1
            print(f"Saved frame {frame_number} at {output_path}")
        else:
            print(f"Warning: Could not read frame {frame_number}.")

    cap.release()
    print(f"Extraction completed. {extracted_count} frames saved.")




def create_composite_figure(image_folder, output_image, images_per_row=5):
    """
    Creates a composite figure from extracted frames.

    :param image_folder: Directory containing the extracted frames.
    :param output_image: Path to save the composite image.
    :param images_per_row: Number of images per row in the composite figure.
    """
    # List all image files in the directory
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    if not image_files:
        print(f"No images found in {image_folder}.")
        return

    num_images = len(image_files)
    num_rows = math.ceil(num_images / images_per_row)

    # Load images
    images = [mpimg.imread(os.path.join(image_folder, img)) for img in image_files]

    # Determine the size of the figure
    fig_width = 20  # inches
    fig_height = fig_width * (num_rows / images_per_row)
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(fig_width, fig_height))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')  # Hide any unused subplots

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Composite figure saved as {output_image}.")

# Example usage
video_dir = "/home/lma326/courses/deep_learning/dl_project/diffusion_world_model/data/videos"
video_name = "predict_final"
video_file = os.path.join(video_dir, f"{video_name}.mp4")  # Replace with your video file path
output_dir = os.path.join(video_dir, "selected_frames", video_name)      # Directory to save selected frames
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
num_frames_to_extract = 8              # Number of frames to extract

extract_evenly_spaced_frames(video_file, output_dir, num_frames_to_extract)

composite_image = os.path.join(output_dir, f"{video_name}.png")  # Output composite image file

create_composite_figure(output_dir, composite_image, images_per_row=num_frames_to_extract)