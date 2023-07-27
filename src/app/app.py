from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash, send_file
import os
import base64
import uuid
from werkzeug.utils import secure_filename
from io import BytesIO
import pandas as pd
import numpy as np
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from ultralytics import YOLO
import sys
sys.path.append('/Users/matt/Desktop/graphs-capstone/src')
from extract_data.functions import classify_chart, get_scatter_data, get_bar_data, get_lineplot_data


app = Flask(__name__)
app.secret_key = "graphs" 



# Function to check if the uploaded file is allowed
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Read the image from the file without saving it to the server
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Classify the uploaded image
            chart_type = classify_chart(image)

            # Redirect based on chart type
            if chart_type == 'scatter':
                # Pass the image as a base64 encoded string to the scatter_forms template
                image_base64 = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
                return render_template('scatter_forms.html', chart_type=chart_type, image_base64=image_base64)
            
            elif chart_type == 'vertical_bar':  # Handle 'vertical_bar' chart type
                # Pass the image as a base64 encoded string to the bar_forms template
                image_base64 = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
                return render_template('bar_forms.html', chart_type=chart_type, image_base64=image_base64)
            
            elif chart_type == 'line':  # Handle 'line' chart type
                # Pass the image as a base64 encoded string to the bar_forms template
                image_base64 = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
                return render_template('line_forms.html', chart_type=chart_type, image_base64=image_base64)

            else:
                # Redirect to other pages for different chart types
                # For example, if chart_type == 'bar', use url_for('bar_forms', chart_type=chart_type)
                # Add more routes and corresponding templates for other chart types as needed
                pass

    return render_template('index.html')



@app.route('/line_forms', methods=['GET', 'POST'])
def line_forms():
    if request.method == 'POST':
        # Get the chart type and base64 encoded image data from the query parameters
        chart_type = request.args.get('chart_type')
        image_base64 = request.args.get('image_base64')

        # Get the form inputs (ymin, ymax, xtick_labels) from the POST request
        ymin = request.form['ymin']
        ymax = request.form['ymax']
        xtick_labels = request.form['xtick_labels'].split(',')

        # Convert the base64 encoded image data to a NumPy array
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Get bar data using the provided inputs
        line_data = get_lineplot_data(image, float(ymin), float(ymax), xtick_labels)

        # Convert the DataFrame to HTML table for rendering in the template
        line_data_table = line_data.to_html(classes='table table-striped')
        return render_template('line_graph.html', chart_type=chart_type, image_base64=image_base64,
                               line_data_table=line_data_table)
    
    # If it's a GET request, simply display the bar plot image
    chart_type = request.args.get('chart_type')
    image_base64 = request.args.get('image_base64')
    return render_template('line_forms.html', chart_type=chart_type, image_base64=image_base64)




@app.route('/line_graph', methods=['GET', 'POST'])
def line_graph():
    if request.method == 'POST':

        # Get the chart type and base64 encoded image data from the form data
        chart_type = request.form['chart_type']
        image_base64 = request.form['image_base64']

        # Convert the base64 encoded image data to a NumPy array
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Get bar data using the provided inputs (ymin, ymax, xtick_labels)
        ymin = request.form['ymin']
        ymax = request.form['ymax']
        xtick_labels = request.form['xtick_labels'].split(',')
        line_data = get_lineplot_data(image, float(ymin), float(ymax), xtick_labels)

        # Convert the DataFrame to HTML table for rendering in the template
        line_data_table = line_data.to_html(classes='table table-striped')

        # Get xlabel, ylabel, and title from the form data
        xlabel = request.form['xlabel']
        ylabel = request.form['ylabel']
        title = request.form['title']

        return render_template('line_graph.html', chart_type=chart_type, image_base64=image_base64,
                               line_data_table=line_data_table, line_data=line_data, xlabel=xlabel, ylabel=ylabel, title=title)
    
    # If it's a GET request, simply display the bar plot image
    chart_type = request.args.get('chart_type')
    image_base64 = request.args.get('image_base64')
    return render_template('line_forms.html', chart_type=chart_type, image_base64=image_base64)






@app.route('/bar_forms', methods=['GET', 'POST'])
def bar_forms():
    if request.method == 'POST':
        # Get the chart type and base64 encoded image data from the query parameters
        chart_type = request.args.get('chart_type')
        image_base64 = request.args.get('image_base64')

        # Get the form inputs (ymin, ymax, xtick_labels) from the POST request
        ymin = request.form['ymin']
        ymax = request.form['ymax']
        xtick_labels = request.form['xtick_labels'].split(',')

        # Convert the base64 encoded image data to a NumPy array
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Get bar data using the provided inputs
        bar_data = get_bar_data(image, float(ymin), float(ymax), xtick_labels)

        # Convert the DataFrame to HTML table for rendering in the template
        bar_data_table = bar_data.to_html(classes='table table-striped')
        return render_template('bar_graph.html', chart_type=chart_type, image_base64=image_base64,
                               bar_data_table=bar_data_table)
    
    # If it's a GET request, simply display the bar plot image
    chart_type = request.args.get('chart_type')
    image_base64 = request.args.get('image_base64')
    return render_template('bar_forms.html', chart_type=chart_type, image_base64=image_base64)




@app.route('/bar_graph', methods=['GET', 'POST'])
def bar_graph():
    if request.method == 'POST':

        # Get the chart type and base64 encoded image data from the form data
        chart_type = request.form['chart_type']
        image_base64 = request.form['image_base64']

        # Convert the base64 encoded image data to a NumPy array
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Get bar data using the provided inputs (ymin, ymax, xtick_labels)
        ymin = request.form['ymin']
        ymax = request.form['ymax']
        xtick_labels = request.form['xtick_labels'].split(',')
        bar_data = get_bar_data(image, float(ymin), float(ymax), xtick_labels)

        # Convert the DataFrame to HTML table for rendering in the template
        bar_data_table = bar_data.to_html(classes='table table-striped')

        # Get xlabel, ylabel, and title from the form data
        xlabel = request.form['xlabel']
        ylabel = request.form['ylabel']
        title = request.form['title']

        return render_template('bar_graph.html', chart_type=chart_type, image_base64=image_base64,
                               bar_data_table=bar_data_table, bar_data=bar_data, xlabel=xlabel, ylabel=ylabel, title=title)
    
    # If it's a GET request, simply display the bar plot image
    chart_type = request.args.get('chart_type')
    image_base64 = request.args.get('image_base64')
    return render_template('bar_forms.html', chart_type=chart_type, image_base64=image_base64)




















@app.route('/scatter_forms', methods=['GET', 'POST'])
def scatter_forms():
    if request.method == 'POST':
        # Get the chart type and base64 encoded image data from the query parameters
        chart_type = request.args.get('chart_type')
        image_base64 = request.args.get('image_base64')

        # Get the form inputs (xmin, xmax, ymin, ymax) from the POST request
        xmin = request.form['xmin']
        xmax = request.form['xmax']
        ymin = request.form['ymin']
        ymax = request.form['ymax']

        # Convert the base64 encoded image data to a NumPy array
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Get scatter data using the provided inputs
        scatter_data = get_scatter_data(image, float(xmin), float(xmax), float(ymin), float(ymax))

        # Convert the DataFrame to HTML table for rendering in the template
        scatter_data_table = scatter_data.to_html(classes='table table-striped')

        return render_template('scatter_graph.html', chart_type=chart_type, image_base64=image_base64,
                               scatter_data_table=scatter_data_table)

    # If it's a GET request, simply display the scatter plot image
    chart_type = request.args.get('chart_type')
    image_base64 = request.args.get('image_base64')

    return render_template('scatter_forms.html', chart_type=chart_type, image_base64=image_base64)






@app.route('/scatter_graph', methods=['GET', 'POST'])
def scatter_graph():
    if request.method == 'POST':
        # Get the chart type and base64 encoded image data from the form data
        chart_type = request.form['chart_type']
        image_base64 = request.form['image_base64']

        # Convert the base64 encoded image data to a NumPy array
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Get scatter data using the provided inputs (xmin, xmax, ymin, ymax)
        xmin = request.form['xmin']
        xmax = request.form['xmax']
        ymin = request.form['ymin']
        ymax = request.form['ymax']
        scatter_data = get_scatter_data(image, float(xmin), float(xmax), float(ymin), float(ymax))

        # Convert the DataFrame to HTML table for rendering in the template
        scatter_data_table = scatter_data.to_html(classes='table table-striped')

        # Get xlabel, ylabel, and title from the form data
        xlabel = request.form['xlabel']
        ylabel = request.form['ylabel']
        title = request.form['title']

        return render_template('scatter_graph.html', chart_type=chart_type, image_base64=image_base64,
                               scatter_data_table=scatter_data_table, scatter_data=scatter_data,
                               xlabel=xlabel, ylabel=ylabel, title=title)

    # If it's a GET request, simply display the scatter plot image
    chart_type = request.args.get('chart_type')
    image_base64 = request.args.get('image_base64')

    return render_template('scatter_forms.html', chart_type=chart_type, image_base64=image_base64)






# Run the app if executed directly
if __name__ == '__main__':
    app.run(debug=True)