# Digitizing Graph Images

  ![Picture1](https://github.com/mmascarelli/graphs-capstone/assets/116842582/6fbc55fc-4a57-4c97-ab1b-ee36b608d482)

  This project aims to address the educational barriers faced by millions of students with learning, physical, or visual disabilities preventing them from reading conventional print. Many STEM educational materials are inaccessible to these students, particularly when it comes to visuals like graphs. While technology can make the written word accessible, adapting visuals remains complex and resource intensive. Manual efforts to create accessible materials are costly and time-consuming, limiting their availability to schools without sufficient funding. To overcome these challenges, the project focuses on using machine learning to automate the process of digitizing various graph types, thereby making educational materials more accessible to learners with disabilities.

  # The Data
  The data for this project originates from Kaggle and consists of 60,578 images and their corresponding annotations in json files. There are 24942 line plots, 19189 vertical bar plots, 11243 scatterplots, 5131 dot plots, and 73 horizontal bar plots. Due to the major class imbalance, specifically with the horizontal bar plots, 200 synthetic images were created by augmenting the original 73. 

<p align="center">
  <img width="510" alt="samples" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/3b7a8f7b-5ead-464a-b6f0-a61d712adfb7">
</p>

# Project Pipeline

I.	Train a convolutional neural network to classify the chart type. Extracting data will be slightly different for each of the different chart types, so being able to correctly classify the type is a vital first step. 

II.	Train a custom object detector using the YOLOv8 architecture for each of the 5 chart types:

III.	Detect XY plane within the image and rescale the data to be within the scale of the chart. The YOLO models return bounding box coordinates of the detected objects on the scale of the image dimensions, but to successfully digitize the image it needs to be on the scale of the chart depicted in the image. 

# PART I: Classifying Charts
Subset of Images: 20,247

* Training: 13,767

* Validation: 3,442
* Testing: 3038

<p align="center">
  <img width="510" height="400" alt="samples" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/c2e349c6-3254-4857-8377-079090f2555c">
</p>

# PART II: YOLO
Using the YOLOv8 architecture, a custom model was trained for each of the 5 chart types.

Test Set Metrics:

* Vertical Bar Recall: 0.936
  
* Horizontal Bar Recall: 0.712
  
* Scatter Recall: 0.918
  
* Line Recall: 0.743
  
* Dot Recall: 0.975

# PART III: Rescaling Bounding Boxes

* Detecting the largest rectangle should result in being the total grid area of the XY plane. This can be done by using the ‘findContours’ function of cv2 and filtering out the smaller contours. However, this method does not work on all the images. There is a back-up method in place to try and detect the XY axes instead using the ‘HoughLinesP’ function of cv2. This is done by detecting all the vertical lines (y-axis) and horizontal lines (x-axis) and then filtering out the lines that are too small/large and ones that are not within a certain boundary of the images.

# PART IV: Using the algorithm
<img width="968" alt="Screenshot 2023-08-03 at 10 35 44 AM" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/f98af05f-bed8-4a66-91ea-4b7ab7f2181c">

<img width="993" alt="Screenshot 2023-08-03 at 10 37 03 AM" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/4c39ecb7-80e0-4182-859a-17c16e73b15e">

<img width="760" alt="Screenshot 2023-08-03 at 10 37 36 AM" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/297a2d7d-307a-4eee-9375-4c4897ee8754">

<img width="1033" alt="Screenshot 2023-08-03 at 10 37 47 AM" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/5268a4ad-b9d0-4c05-a5c1-0eafa9966d12">

<img width="888" alt="Screenshot 2023-08-03 at 10 41 17 AM" src="https://github.com/mmascarelli/graphs-capstone/assets/116842582/360c2f04-90b1-476e-a333-3ce40677659a">

# PART V: Conclusions and Next Steps

The aim of this project was to digitize images of different chart types in the hopes of making charts more accessible for people who have barriers that make it difficult for them to read the information portrayed in a mathematical chart. The results of this project, specifically the web application that I built, can be used by schools, students, textbook companies, math apps, and others. There is still much more work that can be done to improve these results, but it is a great first step towards digitizing charts and making them more accessible.

Next Steps:
1.	Improving accuracy by including more training data.
   
2.	Collect more horizontal bar plots and manually annotate the bar bounding boxes.
   
3.	Automating the rescaling process by using OCR algorithms to extract xy tick labels.
   
4.	Currently, the web app only works for scatter, vertical bar, and line plots. I would like to update it to work for horizontal bar and dot plots.
