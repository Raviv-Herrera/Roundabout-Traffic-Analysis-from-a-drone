# Roundabout-from-a-drone
Classical Optical-Flow object tracking for vehicles within a roundabout from a drone bird eye view.

## Description 
The aim of the project is to take an original video taken by a drone and generate a final result which shows the current vehicles inside the roundabout and track them.
For that purpose we need to stabilize the drone's video, track the vehicles inside and then some interesting statistic inferences can be made.

Note <!> - This project is a 100% pure classical Computer Vision and No Deep Learning techniques were invlolved.
## Pipeline

![image](https://github.com/Raviv-Herrera/Roundabout-from-a-drone/assets/136422674/d777bfaa-0468-49de-b1d3-3fafa7aff533)


## Results 

### The original Video from the drone camera  

![](https://github.com/Raviv-Herrera/Roundabout-from-a-drone/blob/main/original_video.gif)

### The Stabilized Video from the drone camera  

![](https://github.com/Raviv-Herrera/Roundabout-from-a-drone/blob/main/stabilized_video.gif)

### Apply Optical-Flow tracking to follow the vehicles and generating traffic heatmap

![](https://github.com/Raviv-Herrera/Roundabout-from-a-drone/blob/main/opf_vehicles.gif)

![](https://github.com/Raviv-Herrera/Roundabout-from-a-drone/blob/main/traffic_heatmap.gif)

### Final Results
![](https://github.com/Raviv-Herrera/Roundabout-from-a-drone/blob/main/final_results_video.gif)
