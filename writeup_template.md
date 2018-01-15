## Writeup Template
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png

[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/test_example.png
[image51]: ./output_images/initial_frames.png
[image52]: ./output_images/identified_windows.png
[image53]: ./output_images/heat.png
[image54]: ./output_images/identified_cars.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here]

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

![alt text][image1]

For this I used the find_cars function discussed in the lesson. The main reason to use this was the reuse of the hog values calculated once for all the windows.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

I treid different combination using the color spaces

RGB, YUV, HSV and YCrCb. However I found the best performance with YCrCb so I decided to stick with it.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.

Color Spaces :  RGB, HSV, HLS, YUV, YCrCb

Orient : 9, 11

WIndow size: 8, 16

Cells per block : 2

Bin Spatial Parameters : True | False

Color Histogram features : True | False

However the best configuration which worked for me was

Color Space: YCrCb
Orient: 9
Window Size: 8
Cells per blocm : 2
Bin Spatial parameters : True
Color Histogram Features: True
Hog Features : True
Hog Channels : ALL

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default settings.

The feature vector was of length 8460

It took 48.74 Seconds to train SVC

Test Accuracy of SVC =  0.9896

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window was done by the find_cars method.

I used four diffrent scales to capture different size of classifications. I used
1 for smaller window.
1.5, 2 and 2.5 for medium and large windows. The start stop configuration was different for each of them. 

1 - Start: 400 Stop: 480 
1.5 - Start: 400 Stop: 530 
2 - Start: 400 Stop: 560 
2.5 - Start: 400 Stop: 650 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I had tried various configurations, however I used the combination as I mentioned above. It required a lot of experimentation to appear at the right settings.



![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./Video.mp4)

Here's a [link to combined video with advanced lane finding](./CombinedVideo.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image51]

![alt text][image52]

![alt text][image53]

![alt text][image54]

Another optimization I ded was, I created a bounding box class like this

```
from sklearn import linear_model

class BoundingBox(object):
    def __init__(self, bbox, center):
        self.boxes = [bbox]
        self.identification_count = 1
        self.successvive_non_identified_frame_count = 0
        self.enabled = True
        self.centers = [center]
    
    def check_enabled(self):
        if self.successvive_non_identified_frame_count > 3:
            self.enabled = False
    def check_closeness(self, center):
        last_center = self.centers[-1]
        distance = (last_center[0] - center[0]) ** 2 + (last_center[0] - center[0]) ** 2
        if distance < 200:
            return True
        return False
    def add_identification(self, bbox, center): 
        self.boxes.append(bbox)
        self.identification_count += 1
        self.centers.append(center)
        self.successvive_non_identified_frame_count = 0
    
    def predict_identification(self):
        if self.successvive_non_identified_frame_count > 3:
            self.enabled = False
            return
        total_centers = len(self.centers)
        if total_centers < 3:
            self.enabled = False
            return
        self.successvive_non_identified_frame_count += 1
        X = np.arange(total_centers).reshape(-1,1)
        Y = np.array(self.centers)
        regr = linear_model.LinearRegression()
        regr.fit(X,Y)
        new_center = regr.predict(np.array([[total_centers]]))[0].astype(np.int)
        old_center = self.centers[-1]
        self.centers.append(new_center)
        current_bbox = self.boxes[-1]
        ctop_left = current_bbox[0]
        cbottom_right = current_bbox[1]
        top_left = (new_center[0] - old_center[0] + ctop_left[0] , new_center[1] - old_center[1] + ctop_left[1])
        bottom_right = (new_center[0] - old_center[0] + cbottom_right[0] , new_center[1] - old_center[1] + cbottom_right[1])
        bbox = (top_left, bottom_right)
        self.boxes.append(bbox)
```

The algorithm was

1. Using the heatmap identify a box.
2. In the list of existing bounding boxes, check closeness with an existing one.
3. If close, add this one as an identification.
4. Otherwise create a new bounding box instance.
5. For every bounding box for which no new instance was found either disable it or predict a new position for it. Predict for 3 times after that disable.

```
#Implemented like this

class VehicleDetector(object):
    
    def __init__(self, debug=False):
        self.boxes = []
        self.debug = debug
    
    def draw_boxes(self, img, color=(0,0,255)):
        for box in self.boxes:
            if box.enabled and box.identification_count > 3:
                bbox = box.boxes[-1]
                cv2.rectangle(img, bbox[0], bbox[1], color, 6)
        return img
    
    def process_frame(self, img):
        boxes1 = find_cars(img, 400, 480, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        boxes2 = find_cars(img, 400, 530, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        boxes3 = find_cars(img, 400, 560, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        boxes4 = find_cars(img, 400, 650, 2.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat,boxes1)
        heat = add_heat(heat,boxes2)
        heat = add_heat(heat,boxes3)
        heat = add_heat(heat,boxes4)
        total_smooth_added = 0
        for box in self.boxes:
            if box.enabled:
                old_boxes = box.boxes[-2:]
                total_smooth_added += len(old_boxes)
                heat = add_heat(heat, old_boxes)
        heat_threshold = apply_threshold(heat, 2)
        #heat_threshold = apply_threshold(heat, 1)
        labels = label(heat_threshold)
        bboxes = get_labeled_bboxes(labels)
        if self.debug:
            draw_img = self.draw_boxes(np.copy(img), color=(0,255,255))
        else:
            draw_img = img
        
        for bbox in bboxes:
            centerx = int((bbox[1][0] + bbox[0][0])/2)
            centery = int((bbox[1][1] + bbox[0][1])/2)
            found = False
            for box in self.boxes:
                if not box.enabled:
                    continue
                if box.check_closeness([centerx, centery]):
                    found = True
                    box.add_identification(bbox, [centerx, centery])
                    break
            if not found:
                if centery > 400 and centerx > 700 :
                    self.boxes.append(BoundingBox(bbox, [centerx, centery]))
        for bounding_box in self.boxes:
            if not bounding_box.enabled:
                continue
            found = False
            for bbox in bboxes:
                centerx = int((bbox[1][0] + bbox[0][0])/2)
                centery = int((bbox[1][1] + bbox[0][1])/2)
                if bounding_box.check_closeness([centerx, centery]):
                    found = True
                    break
            if not found:
                bounding_box.predict_identification()
        draw_img = self.draw_boxes(np.copy(draw_img))
        if self.debug:
            draw_img = draw_boxes(draw_img, boxes1, color=(255, 0, 0), thick=6)
            draw_img = draw_boxes(draw_img, boxes2, color=(255, 0, 0), thick=6)
            draw_img = draw_boxes(draw_img, boxes3, color=(255, 0, 0), thick=6)
            draw_img = draw_boxes(draw_img, boxes4, color=(255, 0, 0), thick=6)
            draw_img = draw_boxes(draw_img, bboxes, color=(0, 255, 0), thick=6)
        return draw_img
```

Hence  
Optimization 1 -> Identify and track bounding boxes, and mark every new bounding box as an iteration of the old one. This way we can even track the count of cars on screen, and for how long they were visible. Also in case a frame misses the identification, we just predict the position using linear regression (can definitely do better then linear). After 3 consecutive predictions without any identification we just disable the box.  

Optimization 2 -> Also applied some smoothing by adding boxes from previous 2 identifications to the heatmap.  This helps in cutting down false positives. Only good objects will be classified in consecutive frames. Also any kind of noise in classification is also smoothened, resulting in better boxes.

```
total_smooth_added = 0
for box in self.boxes:
    if box.enabled:
        old_boxes = box.boxes[-2:]
        total_smooth_added += len(old_boxes)
        heat = add_heat(heat, old_boxes)
heat_threshold = apply_threshold(heat, 2)
```

Optimization 3 -> Only add new boxes if they appear to the right of the car.

```
if not found:
    if centery > 400 and centerx > 700 :
        self.boxes.append(BoundingBox(bbox, [centerx, centery]))
```
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline fails/ wobbles when there are 2 different cars passing each other. Need to think about a strategy on that front. Also the pipeline currently will fail to pick a car in front, because of the some hardcoding (optimization 3). The hardcoding might not be needed, if we further improve the aforementioned parameters and optimizations. 

