import os
import zipfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from calculate_metrics import calculate_metrics
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
SINGLE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'single')
MULTI_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'multi')
OUTPUT_DIR = 'results'

os.makedirs(SINGLE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MULTI_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SINGLE_UPLOAD_FOLDER'] = SINGLE_UPLOAD_FOLDER
app.config['MULTI_UPLOAD_FOLDER'] = MULTI_UPLOAD_FOLDER

# Load the trained model
model = load_model('../outputs/unet/unet_model.h5', compile=False)

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_outputs(image_path, output_folder):
    # Load and resize the original image
    original_image = Image.open(image_path).convert('RGB')
    original_image_resized = original_image.resize((256, 256))
    original_image_array = img_to_array(original_image_resized) / 255.0

    # Predict mask
    pred_mask = model.predict(np.expand_dims(original_image_array, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save predicted mask
    output_mask_path = os.path.join(output_folder, f'{base_name}_predicted_mask.png')
    pred_mask_resized = cv2.resize(pred_mask, (original_image.size[0], original_image.size[1]))
    cv2.imwrite(output_mask_path, pred_mask_resized * 255)

    # Convert the original image from RGB to BGR
    original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    pred_mask_bgr = np.repeat(pred_mask_resized[:, :, np.newaxis], 3, axis=2) * 255

    # Create translucent overlays
    red_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    blue_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    red_overlay[:, :, 2] = 255
    blue_overlay[:, :, 0] = 255

    color_overlay = np.where(pred_mask_resized[:, :, np.newaxis], 
                             cv2.addWeighted(original_image_bgr, 1, red_overlay, 0.5, 0),
                             cv2.addWeighted(original_image_bgr, 1, blue_overlay, 0.5, 0))

    output_overlay_path = os.path.join(output_folder, f'{base_name}_color_overlay.png')
    cv2.imwrite(output_overlay_path, color_overlay)

    # Calculate metrics
    length, width, area, perimeter, circularity = calculate_metrics(pred_mask_resized)

    # Create comparison image
    comparison_image = np.hstack((original_image_bgr, pred_mask_bgr, color_overlay))
    padding_height = 100
    blank_space = Image.new("RGB", (comparison_image.shape[1], 110), (0, 0, 0))
    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, size=11)
    draw = ImageDraw.Draw(blank_space)
    metrics_text = (f"Length: {length:.2f} mm | Width: {width:.2f} mm | Area: {area:.2f} mm² | "
                    f"Perimeter: {perimeter:.2f} mm | Circularity: {circularity:.2f}")
    
    # Calculate text size to center it
    text_bbox = draw.textbbox((0, 0), metrics_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # Width of the text
    text_height = text_bbox[3] - text_bbox[1]  # Height of the text
    
    # Calculate the center position
    position_x = (blank_space.width - text_width) // 2
    position_y = padding_height // 2  # Position text with some space above

    # Render the text onto the blank space using Pillow
    draw.text((position_x, position_y), metrics_text, font=font, fill=(255, 255, 255))

    comparison_with_metrics = np.vstack((comparison_image, np.array(blank_space)))
    output_comparison_path = os.path.join(output_folder, f'{base_name}_comparison.png')
    cv2.imwrite(output_comparison_path, comparison_with_metrics)

    return output_mask_path, output_overlay_path, output_comparison_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        upload_type = request.form.get('upload_type')  # Get the selected upload type
        file = request.files.get('file')  # Get the uploaded file
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            if upload_type == 'multi' and filename.endswith('.zip'):
                # Handle zip file upload
                file_path = os.path.join(app.config['MULTI_UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Extract files from the zip
                extracted_folder = os.path.splitext(filename)[0]
                #extraction_place = os.path.join(app.config['MULTI_UPLOAD_FOLDER'],extracted_folder)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(MULTI_UPLOAD_FOLDER)
                os.remove(file_path)  # Remove the zip file after extraction

                # Create output folder structure
                output_folder = os.path.join(OUTPUT_DIR, 'multi',extracted_folder)
                os.makedirs(output_folder, exist_ok=True)

                #extraction_path = os.path.join(extraction_place,extracted_folder)
                extraction_path = os.path.join(app.config['MULTI_UPLOAD_FOLDER'],extracted_folder)
                # Process extracted images
                image_paths = [os.path.join(extraction_path, f) for f in os.listdir(extraction_path) if f.endswith(('png', 'jpg', 'jpeg'))]

                # Count the number of images in the zip file
                num_images = len(image_paths)
                
                # Create a folder for each image's results
                output_paths = []
                for image_path in image_paths:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    image_output_folder = os.path.join(output_folder, base_name)
                    os.makedirs(image_output_folder, exist_ok=True)  # Create folder for each image's results
                    output_paths.append(generate_outputs(image_path, image_output_folder))  # Save outputs to the image's output folder

                return render_template('results.html', message='Multiple images processed successfully!', folder_type = 'multi',outputs=output_paths, folder_name=extracted_folder)

            elif upload_type == 'single':
                # Handle single image upload
                file_path = os.path.join(app.config['SINGLE_UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Create a new folder for the single output
                picture_name = os.path.splitext(filename)[0]
                single_output_folder = os.path.join(OUTPUT_DIR, 'single', picture_name)
                os.makedirs(single_output_folder, exist_ok=True)

                # Generate outputs and save in the specific folder
                output_paths = generate_outputs(file_path, single_output_folder)
                
                return render_template('results.html', message='Single image processed successfully!', folder_type = 'single',outputs=[output_paths], folder_name=picture_name)

@app.route('/download/<folder_type>/<folder_name>', methods=['GET'])
def download(folder_type, folder_name):
    if folder_type == 'multi':
        folder_path = os.path.join(OUTPUT_DIR, 'multi', folder_name)
        zip_dir = os.path.join(OUTPUT_DIR, 'downloads', 'multi')
    elif folder_type == 'single':
        folder_path = os.path.join(OUTPUT_DIR, 'single', folder_name)
        zip_dir = os.path.join(OUTPUT_DIR, 'downloads', 'single')
    else:
        return "Invalid folder type.", 400

    zip_filename = f"{folder_name}.zip"
    
    # Ensure the download directory exists
    os.makedirs(zip_dir, exist_ok=True)

    # Create a zip file of the results
    zip_filepath = os.path.join(zip_dir, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

    return send_file(zip_filepath, as_attachment=True)

@app.route('/results/<folder_name>/<path:filename>', methods=['GET'])
def serve_image(folder_name, filename):
    file_path = os.path.join(OUTPUT_DIR, 'single', folder_name, filename.replace('\\', '/'))
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True)
