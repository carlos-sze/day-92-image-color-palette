from flask import Flask, render_template, request
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import webcolors
import base64

app = Flask(__name__)

def get_top_colors(image, num_colors=10):
    # Open the image and convert it to RGB
    img = Image.open(image).convert('RGB')

    # Resize the image to reduce processing time (optional)
    img = img.resize((100, 100))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Flatten the image array
    img_flat = img_array.reshape(-1, 3)

    # Perform K-means clustering to find the dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_flat)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    # Convert RGB values to hex codes and get the counts
    hex_codes = []
    counts = []
    for color in colors:
        hex_code = webcolors.rgb_to_hex(color.astype(int))
        hex_codes.append(hex_code)
        counts.append(list(kmeans.labels_).count(kmeans.predict([color.round()])))

    # Normalize the counts
    counts = np.array(counts) / np.sum(counts)

    # Sort the colors by counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    sorted_hex_codes = [hex_codes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # Return the top colors as a list of tuples
    return list(zip(sorted_hex_codes[:num_colors], sorted_counts[:num_colors]))

# Custom filter to format percentage
@app.template_filter('percent')
def format_percent(value):
    return "{:.2%}".format(value)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return render_template('index.html', error='No image file')

        file = request.files['image']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', error='No selected image file')

        # Check if the file is an image
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return render_template('index.html', error='Invalid image file')

        # Read the image file
        image = file.read()

        # Open the image using PIL
        pil_image = Image.open(io.BytesIO(image))

        # Create a thumbnail version of the image
        pil_image.thumbnail((300, 300))

        # Convert the thumbnail image to bytes
        thumbnail_data = io.BytesIO()
        pil_image.save(thumbnail_data, format='JPEG')
        thumbnail_data.seek(0)

        # Get the top colors in the image
        color_data = get_top_colors(thumbnail_data)

        # Convert thumbnail image data to base64
        thumbnail_base64 = base64.b64encode(thumbnail_data.getvalue()).decode('utf-8')

        # Render the template with the color data and thumbnail image
        return render_template('index.html', color_data=color_data, thumbnail_data=thumbnail_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)