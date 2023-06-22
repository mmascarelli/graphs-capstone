import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import easyocr
from scipy import stats

def clean_data_series(plot_index):
       
    """
    Clean the data series from the specified plot index.

    Args:
        plot_index (int): The index of the plot to clean the data series from.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned x and y coordinates.
    
    """
    img_df = pd.read_csv('../data/processed/subset_data.csv')
    scatter_data = img_df.iloc[plot_index][2]

    row_data = scatter_data
    fixed_row_data = row_data.replace("'", "\"")

     # Parse the string into a list of dictionaries
    data_list = json.loads(fixed_row_data)

    # Extract the data from each dictionary
    x_coords = []
    y_coords = []

    for data_point in data_list:
        x_coords.append(data_point['x'])
        y_coords.append(data_point['y'])

    df = pd.DataFrame({'x_actual':x_coords,'y_actual':y_coords})
    return df




def get_lines(image_path):
    """
    Detect and plot lines in a scatterplot image.

    Args:
        image_path (str): The path to the scatterplot image file.

    Returns:
        np.ndarray: An array containing the detected lines.
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


    scatter_rgb = cv2.cvtColor(scatter_image, cv2.COLOR_BGR2RGB)
    plt.imshow(scatter_rgb)

    # Iterate over the lines and plot each line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)

    plt.show()
    return lines



reader = easyocr.Reader(['en'])
def get_title_bbox(image_path):
    """
    Extract the bounding box of the title from a scatterplot image.
    The bbox is used as an upper bound for finding the bboxes of scatterplot points.

    Args:
        image_path (str): The path to the scatterplot image file.

    Returns:
        float: The y-coordinate of the upper bound of the title bounding box.
    """
    scatter_image = cv2.imread(image_path) 
    gray_image = cv2.cvtColor(scatter_image, cv2.COLOR_BGR2GRAY)
    x_axis, y_axis = get_axes(image_path, show_plot=False) 
    results = reader.readtext(gray_image[:y_axis[0][3], :]) 
    upper_bound = results[-1][0][-1][1]
    return upper_bound




def get_yticks(image_path):
    scatter_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(scatter_image, cv2.COLOR_BGR2GRAY)
    
    # Take 1/5 of the image on the left side
    left_region = int(gray_image.shape[1]/5)
    results = reader.readtext(gray_image[:,:left_region])
    
    # Only take results with more 90% confidence
    filtered_results = [result for result in results if result[2] > 0.70]
    
    # Greatest x-coordinate for the text, will be used as a left bound for axis/point detection
    left_bound = filtered_results[0][0][1][0]

    # Loop through results to extract predicted text
    y_ticks = []
    for i in range(len(filtered_results)):
        y = filtered_results[i][1]
        if y.find(',') > 0:
            y = y.replace(',','.')
        y_ticks.append(y)

    # Convert the text to integers
    y_ticks = [float(num) for num in y_ticks]

    # Calculate the first and third quartiles
    q1, q3 = np.percentile(y_ticks, [25, 75])

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the threshold as a certain percentage (e.g., 1.5 times) of the IQR
    threshold = 1.5  # Adjust this value as needed

    # Filter out numbers beyond the threshold
    y_ticks = [num for num in y_ticks if q1 - threshold * iqr <= num <= q3 + threshold * iqr]

    y_ticks = sorted(y_ticks)

    return y_ticks, left_bound





def get_xticks(image_path):

    scatter_image = cv2.imread(image_path) 
    gray_image = cv2.cvtColor(scatter_image, cv2.COLOR_BGR2GRAY)
    lower_region = int(gray_image.shape[0] - gray_image.shape[0]/5)
    results = reader.readtext(gray_image[lower_region:,:])

    filtered_results = [result for result in results if result[2] > 0.9] 
    
    lower_bound = lower_region + results[0][0][0][1]

    x_ticks = []
    for i in range(len(filtered_results)):
        x = filtered_results[i][1]
        x_ticks.append(x)

    xticks = []
    for num in x_ticks:
        try:
            xticks.append(int(num))
        except ValueError:
            pass

    return xticks, lower_bound







def get_axes(image_path, show_plot=True):
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








def get_points(image_path, x_axis, y_axis, num_points, show_plot=True, show_transformations=False):
    """
    Extract the bounding boxes of the data points from a scatterplot image.

    Args:
        image_path (str): The path to the scatterplot image file.
        x_axis (numpy.ndarray): The x-axis line as a numpy array.
        y_axis (numpy.ndarray): The y-axis line as a numpy array.
        show_plot (bool, optional): Whether to display the image with the detected bounding boxes. 
                                    Defaults to True.
        show_transformations (bool, optional): Whether to display the intermediate image transformations. 
                                               Defaults to False.

    Returns:
        list: A list of bounding boxes, each represented as (xmin, ymin, xmax, ymax).
    """
    
    # Load the scatterplot image
    scatter_image = cv2.imread(image_path)

    # If the image has a dark background, we will apply bitwise_not
    dark_image = has_dark_background(image_path, 50)
    if dark_image == True:
        scatter_image = cv2.bitwise_not(scatter_image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(scatter_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Apply morphological operations (erosion and dilation) to enhance the points
    kernel = np.ones((3, 3), np.uint8)
    erode_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(erode_image, kernel, iterations=1)

    # Find contours in the processed image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get upper bound bbox from title
    upper_bound = get_title_bbox(image_path)

    
    # Try different cutoff points for contour area
    cutoff_points = [10,20,30,40,50,60,70,80,90,100,150,200,300,350,400]

    # Iterate over each contour area cutoff point
    for cutoff in cutoff_points:
        # Initialize empty list for bboxes
        bounding_boxes = []
        # Iterate over each contour and obtain the bounding box
        for contour in contours:
            # Calculate the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
        
            # Filter out small contours (adjust the threshold as needed)
            contour_area = cv2.contourArea(contour)
            if contour_area < cutoff and w < gray_image.shape[1] and h < gray_image.shape[0]:
                # Restricting bboxes to be above x-axis, right of y-axis, and below title
                if x > y_axis[0][0]  and y < y_axis[0][1] and y < x_axis[0][1] and y > upper_bound and x > 20: 
                    bounding_boxes.append((x, y, x + w, y + h))  # (xmin, ymin, xmax, ymax)
        # Stop the loop if at least 10 points are found, but less than 30 points       
        if len(bounding_boxes) == num_points:
            break


    # For showing the plot with bboxes         
    if show_plot == True:
        scatter_image = cv2.imread(image_path)
        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(scatter_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        x_axis, y_axis = get_axes(image_path, show_plot=False)
        x1, y1, x2, y2 = x_axis[0]

        cv2.line(scatter_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        x1, y1, x2, y2 = y_axis[0]

        cv2.line(scatter_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(scatter_image)
        plt.show

    # For showing the different image transformations
    if show_transformations == True:
        scatter_image = cv2.imread(image_path)
        imgs = [scatter_image, gray_image, binary_image,erode_image, dilated_image]
        titles = ["img", "gray", "binary","erode","dilated","bbox of points"]
        i = 1
        plt.figure(figsize=(12,12))
        for each in range(len(imgs)):
            plt.subplot(4,2,i)
            plt.imshow(imgs[each])
            plt.title("{}-{}".format(each +1,titles[each]))
            plt.xticks([])
            plt.yticks([])
            i += 1

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

    return bounding_boxes

    
    

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





def rescale_points(points, x_axis, y_axis, target_x_min, target_x_max, target_y_min, target_y_max):
    """
    Rescale the coordinates of the bounding boxes to a new target range.

    Args:
        points (list): A list of bounding boxes, each represented as (xmin, ymin, xmax, ymax).
        x_axis (numpy.ndarray): The x-axis line as a numpy array.
        y_axis (numpy.ndarray): The y-axis line as a numpy array.
        target_x_min (int): The minimum value of the target x-range.
        target_x_max (int): The maximum value of the target x-range.
        target_y_min (int): The minimum value of the target y-range.
        target_y_max (int): The maximum value of the target y-range.

    Returns:
        list: A list of rescaled bounding boxes, each represented as (xmin, ymin, xmax, ymax).
    """

    # Define the original range of coordinates
    x_min = x_axis[0][0]
    x_max = x_axis[0][2]

    y_min = y_axis[0][1]
    y_max = y_axis[0][3]

    # Iterate over the bounding boxes
    scaled_bounding_boxes = []
    bboxes = points
    for box in bboxes:
        x1, y1, x2, y2 = box

        # Scale the x-coordinates of the bounding box
        scaled_x1 = int((x1 - x_min) * (target_x_max - target_x_min) / (x_max - x_min) + target_x_min)
        scaled_x2 = int((x2-x_min) * (target_x_max - target_x_min) / (x_max - x_min)+ target_x_min)

        # Scale the y-coordinates of the bounding box
        scaled_y1 = int((y1 - y_min) * (target_y_max - target_y_min) / (y_max - y_min) + target_y_min)
        scaled_y2 = int((y2 - y_min) * (target_y_max - target_y_min) / (y_max - y_min) + target_y_min)

        # Append the scaled bounding box to the list
        scaled_bounding_boxes.append((scaled_x1, scaled_y1, scaled_x2, scaled_y2))

    return scaled_bounding_boxes







def predict_points(scaled_points):
    """
    Predict the x and y coordinates based on the scaled bounding boxes.
    This is done by taking the mean coordinates: x = mean(x1,x2) and y = mean(y1,y2)

    Args:
        scaled_points (list): A list of scaled bounding boxes, each represented as (xmin, ymin, xmax, ymax).

    Returns:
        pandas.DataFrame: A DataFrame with the predicted x and y coordinates.
    """
    x_pred = []
    y_pred = []

    for box in scaled_points:
        x1, y1, x2, y2 = box
        x_mean = np.mean([x1, x2])
        x_pred.append(x_mean)
        y_mean = np.mean([y1, y2])
        y_pred.append(y_mean)

    df = pd.DataFrame({'x_pred':x_pred,'y_pred':y_pred})
    df.sort_values(by='x_pred', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
    




def compare_pred_actual(pred_df, actual_df):
    """
    Compare the predicted and actual coordinates by concatenating them side by side.

    Args:
        pred_df (pandas.DataFrame): DataFrame with predicted x and y coordinates.
        actual_df (pandas.DataFrame): DataFrame with actual x and y coordinates.

    Returns:
        pandas.DataFrame: A DataFrame with predicted and actual x and y coordinates.
    """

    pred_actual = pd.concat([pred_df,actual_df], axis=1)
    return pred_actual