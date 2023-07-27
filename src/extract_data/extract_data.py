import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from ultralytics import YOLO


classify_model = pickle.load(open('/Users/matt/Desktop/graphs-capstone/models/subset_with_augs_model.pkl','rb'))
scatter_model = YOLO('/Users/matt/Desktop/graphs-capstone/models/scatterplot_model.pt')
bar_model = YOLO('/Users/matt/Desktop/graphs-capstone/models/barplot_model.pt')
dot_model = YOLO('/Users/matt/Desktop/graphs-capstone/models/dotplot_model.pt')
line_model = YOLO('/Users/matt/Desktop/graphs-capstone/models/lineplot_model.pt')
barh_model = YOLO('/Users/matt/Desktop/graphs-capstone/models/barHplot_model.pt')
data = pd.read_csv('/Users/matt/Desktop/graphs-capstone/data/interim/annots_imgs_merged.csv')
labelencoder=LabelEncoder()
labels = to_categorical(labelencoder.fit_transform(data['chart-type']))
labels = list(labelencoder.classes_)


def classify_chart(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    img = np.expand_dims(img, axis=0)
    pred = classify_model.predict(img, verbose=False)
    pred = np.argmax(pred,axis=1)
    pred = labelencoder.inverse_transform(pred) 
    pred = pred[0]
    return pred

def get_scatter_graph(path, xmin, xmax, ymin, ymax, xlabel, ylabel, title):
    results = scatter_model(path, verbose=False)
    res = results[0].boxes.data
    bbox_pred = []
    for bbox in res:
        x1, y1, x2, y2, probs, pred = bbox
        bbox_pred.append((int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())))

    rescaled_data = rescale_scatter_points(path, bbox_pred, xmin, xmax, ymin, ymax)

    fig = px.scatter(rescaled_data, x="x_pred", y="y_pred", width=800, height=500)
    fig.update_layout(xaxis= dict(tick0=0, dtick=20, tickmode='linear',title='{}'.format(xlabel)),
                      yaxis= dict(tick0=0, dtick=10, tickmode='linear',title='{}'.format(ylabel)),
                      title = '{}'.format(title))
    fig.update_layout(yaxis_range=[ymin,ymax],
                     xaxis_range=[xmin,xmax])
    fig.show()
    return rescaled_data

def get_bar_graph(path, ymin, ymax, xtick_labels, xlabel, ylabel, title):
    results = bar_model(path, verbose=False)
    res = results[0].boxes.data
    bbox_pred = []
    for bbox in res:
        x1, y1, x2, y2, probs, pred = bbox
        bbox_pred.append((int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())))

    rescaled_data = rescale_bar_heights(path, bbox_pred, ymin, ymax, xtick_labels)

    # fig = px.bar(rescaled_data, x="xlabel", y="bar_height_pred", width=800, height=500)
    # fig.update_layout(xaxis= dict(title='{}'.format(xlabel)),
                      # yaxis= dict(title='{}'.format(ylabel)),
                      # title = '{}'.format(title))
    # fig.show()
    return rescaled_data

def get_dotplot(path, xtick_labels, xlabel, ylabel, title):
    results = dot_model(path, verbose=False)
    bbox_pred = []
    res = results[0].boxes.data
    for bbox in res:
        x1, y1, x2, y2, probs, pred = bbox
        bbox_pred.append((int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())))

    dots = []
    for box in bbox_pred:
        dot = box[0]
        dots.append(dot)

    # Count how many times does each value occur
    unique_values, counts = np.unique(dots, return_counts=True)
    
    # Convert 1D input into 2D array
    scatter_x = [] # x values 
    scatter_y = [] # corresponding y values
    for idx, value in enumerate(unique_values):
        for counter in range(1, counts[idx]+1):
            scatter_x.append(value)
            scatter_y.append(counter)

    # draw dot plot using scatter() 
    fig, ax = plt.subplots()
    ax.scatter(scatter_x, scatter_y, s=200)
    ax.set_xticks(unique_values)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()

def get_lineplot(path, ymin, ymax, xtick_labels, xlabel, ylabel,title):
    results = line_model(path, verbose=False)
    bbox_pred = []
    res = results[0].boxes.data
    for bbox in res:
        x1, y1, x2, y2, probs, pred = bbox
        bbox_pred.append((int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())))

    line_df = rescale_lineplot_points(path,bbox_pred, ymin, ymax, xtick_labels)

    # fig = px.line(line_df, x="xlabel", y="point_pred", width=900, height=400)
    # fig.update_layout(yaxis_range=[ymin,ymax],
                      # xaxis= dict(title='{}'.format(xlabel)),
                      # yaxis= dict(title='{}'.format(ylabel)),
                      # title = '{}'.format(title))
    # fig.show()
    return line_df








def has_dark_background(image_path, brightness_threshold=50):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness of the image
    average_brightness = np.mean(gray) * 100 / 255

    # Compare the average brightness with the threshold
    if average_brightness <= brightness_threshold:
        return True  # Dark background
    else:
        return False  # Bright background
    
def detect_graph_space(image_path, show_graph=False):
    # Load the image
    image = cv2.imread(image_path)

    dark_image = has_dark_background(image_path, 50)
    if dark_image:
        image = cv2.bitwise_not(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]
    
    if len(valid_contours) == 0:
        return None
    else:
        # Sort contours by area in descending order
        largest_contour = max(valid_contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        image_area = image.shape[0] * image.shape[1]
        area_ratio = contour_area / image_area
        min_area_ratio = 0.001    # Minimum threshold ratio
        max_area_ratio = 0.70    # Maximum threshold ratio
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return None
        

        if show_graph==True:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(image)
        
    return largest_contour

def get_axes(image_path, show_plot=False):
    """
    Detect and extract the axes lines from a scatterplot image.

    Args:
        image_path (str): The path to the scatterplot image file.
        show_plot (bool, optional): Whether to display the image with the detected axes lines. 
                                    Defaults to True.

    Returns:
        tuple: A tuple containing the x-axis and y-axis lines as numpy arrays.
    """

    # Load the scatterplot image
    scatter_image = cv2.imread(image_path) 
    
    # If the image has a dark background, we will apply bitwise_not
    dark_image = has_dark_background(image_path, 50)
    if dark_image == True:
        scatter_image = cv2.bitwise_not(scatter_image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(scatter_image, cv2.COLOR_BGR2GRAY)   

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(gray_image, 50, 150) 

    # Perform Hough line detection to detect the lines in the image
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Initialize variables to store the x_axis and highest y coordinates
    # The x_axis bbox should have the largest y-coordinate
    x_axis = None
    highest_y = 0

    # Initialize variables to store the y_axis and smallest x coordinate
    # The y_axis bbox should have the smallest x-coordinate
    y_axis = None
    smallest_x = float('inf')

    # Iterate over the detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract line coordinates

        # Calculate the length of the line
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # The x-axis should be 4/5 the length of the image x length
        x_axis_line_thresh = 4*(scatter_image.shape[1]/5)

        # The y-axis should be 4/5 the length of the image y length
        y_axis_line_thresh = 4*(scatter_image.shape[0]/5)

        # Restricting the line length and location of the line to above x-label and to the right of the y-label
        if line_length > 150 and y2 < scatter_image.shape[0]-20 and x2 > 20:

            # Check if the line is horizontal
            if abs(y2 - y1) < 5 and y1 > highest_y and line_length > x_axis_line_thresh:
                highest_y = y1
                x_axis = line

            # Check if the line is vertical
            if abs(x2 - x1) < 5 and x1 < smallest_x and y_axis_line_thresh:
                smallest_x = x1
                y_axis = line

    if x_axis is None:
        height, width = scatter_image.shape[:2]
        x_axis = np.array([[0, height // 2, width, height // 2]], dtype=np.int32)

    if y_axis is None:
        height, width = scatter_image.shape[:2]
        y_axis = np.array([[width // 2, 0, width // 2, height]], dtype=np.int32)

    # If the x-axis extends past the y-axis, we will trim a piece off
    if x_axis[0][0] < y_axis[0][0]:
        x1, _, _, _ = y_axis[0]
        _, y1, x2, y2 = x_axis[0]
        line = np.array([[x1, y1, x2, y2]])
        x_axis[0] = line

    if y_axis[0][1] > x_axis[0][1]:
        _, y1, _, _ = x_axis[0]
        x1, _, x2, y2 = y_axis[0]
        line = np.array([[x1, y1, x2, y2]])
        y_axis[0] = line

    if show_plot == True:
        scatter_image = cv2.imread(image_path)
        # Extract coordinates for x_axis
        x1, y1, x2, y2 = x_axis[0]

        # Draw the x_axis on the scatterplot image
        cv2.line(scatter_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Extract the coordinates of the y_axis 
        x1, y1, x2, y2 = y_axis[0]

        # Draw the y_axis on the scatterplot image
        cv2.line(scatter_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(scatter_image)
        plt.show
    
    return x_axis, y_axis

def rescale_scatter_points(image_path, points, target_x_min, target_x_max, target_y_min, target_y_max):
    
    
    grid = detect_graph_space(image_path)
    x, y, w, h = cv2.boundingRect(grid)

    if grid is not None:
        x_min = x
        x_max = x + w

        y_min = y + h
        y_max = y
    
    else:
        x_axis, y_axis = get_axes(image_path)
        # Define the original range of coordinates
        x_min = x_axis[0][0]
        x_max = x_axis[0][2]

        y_min = y_axis[0][1]
        y_max = y_axis[0][3]

    # Iterate over the points
    scaled_points = []
    for point in points:
        if len(point) == 4:
            x1, y1, x2, y2 = point
            x_mean = np.mean([x1, x2])
            y_mean = np.mean([y1, y2])
            scaled_x = int((x_mean - x_min) * (target_x_max - target_x_min) / (x_max - x_min) + target_x_min)
            scaled_y = int((y_mean - y_min) * (target_y_max - target_y_min) / (y_max - y_min) + target_y_min)

            # Append the scaled bounding box to the list
            scaled_points.append((scaled_x, scaled_y))

        elif len(point) == 2:
            x_point = point[0]
            y_point = point[1]    
            scaled_x = int((x_point - x_min) * (target_x_max - target_x_min) / (x_max - x_min) + target_x_min)
            scaled_y = int((y_point - y_min) * (target_y_max - target_y_min) / (y_max - y_min) + target_y_min)
            scaled_points.append((scaled_x, scaled_y))

    x_scaled = []
    y_scaled = []    
    for point in scaled_points:
        x, y = point
        x_scaled.append(x)
        y_scaled.append(y)

        
    
    df = pd.DataFrame({'point_pred':scaled_points, 'x_pred':x_scaled,'y_pred':y_scaled})
    df.sort_values(by='point_pred', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
       
    return df

def rescale_bar_heights(path, bboxes, target_ymin, target_ymax, xlabels):
    
    
    grid = detect_graph_space(path)
    x, y, w, h = cv2.boundingRect(grid)

    if grid is not None:
        y_min = y + h
        y_max = y
    
    else:
        x_axis, y_axis = get_axes(path)
        y_min = y_axis[0][1]
        y_max = y_axis[0][3]
    
    bboxes = sorted(bboxes)
    scaled_heights = []
    for bbox in bboxes:
        _, y1, _, y2 = bbox
        height = y1
        y_min = y2
        scaled_height = int((height - y_min) * (target_ymax - target_ymin) / (y_max - y_min) + target_ymin)
        scaled_heights.append(scaled_height)

    labels = xlabels
    df = pd.DataFrame({'xlabel':labels,'bar_height_pred':scaled_heights})
    return df

def rescale_lineplot_points(path, bboxes, target_ymin, target_ymax, xlabels):
    
    grid = detect_graph_space(path)
    x, y, w, h = cv2.boundingRect(grid)

    if grid is not None:
        y_min = y + h
        y_max = y
    
    else:
        x_axis, y_axis = get_axes(path)
        y_min = y_axis[0][1]
        y_max = y_axis[0][3]
    
    bboxes = sorted(bboxes)
    scaled_points = []
    for bbox in bboxes:
        _, y1, _, y2 = bbox
        point = (y1 + y2)/2
        scaled_point = int((point - y_min) * (target_ymax - target_ymin) / (y_max - y_min) + target_ymin)
        scaled_points.append(scaled_point)

    labels = xlabels
    df = pd.DataFrame({'xlabel':labels,'point_pred':scaled_points})
    return df

