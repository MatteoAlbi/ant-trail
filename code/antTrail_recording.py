#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import cv2
import depthai as dai
import blobconverter
import numpy as np
from libraries.depthai_replay import Replay

# Compute saved fuel: first order approximation
# Data from CFD simulation
Cd1, dist1 = 0.265, 10000
Cd2, dist2 = 0.051, 300
m_Cd = (Cd1 - Cd2) / (dist1-dist2)  #[Cd/mm]
q_Cd = Cd1 - (m_Cd * dist1)              #[Cd]

# Data and hypotesis
absolute_speed = 25     #[m/s], 90 [km/h]
reference_area = 10.5   #[m^2]
air_density = 1.225     #[kg/m^3]
roll_resistance_power = 70630 #[W]
# Nominal case: power of air drag
nominal_Cd = 0.704
nominal_power = 1/2*reference_area*air_density*nominal_Cd*pow(absolute_speed,3) + roll_resistance_power  #[W]

'''
# NN for license plate detection, PROBLEM: need front facing cars
nnBlobPath=blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106", shaves=6)
frame_dimension = [300,300]
'''

# NN for vehicle detection
nnBlobPath=blobconverter.from_zoo(name="vehicle-detection-0202", shaves=6)
frame_dimension = [512,512]

labelMap = ["vehicle"]

path = "..\\recordings\\car1"
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
    args = parser.parse_args()
    path = args.path

# Create Replay object
replay = Replay(path)
# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.init_pipeline()

# Resize color frames prior to sending them to the device and 
replay.set_resize_color(frame_dimension)


# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case frame_dimension)
replay.keep_aspect_ratio(True)
# Crop window position
replay.set_crop_spacing((100,100))

#-- Def NN
nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
nn.setBlobPath(nnBlobPath)
nn.setConfidenceThreshold(0.5)
nn.input.setBlocking(False)
nn.setBoundingBoxScaleFactor(0.5)
nn.setDepthLowerThreshold(100)
nn.setDepthUpperThreshold(5000)

# Link required inputs to the Spatial detection network
nodes.color.out.link(nn.input)
nodes.stereo.depth.link(nn.inputDepth)

detOut = pipeline.create(dai.node.XLinkOut)
detOut.setStreamName("det_out")
nn.out.link(detOut.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
nodes.stereo.disparity.link(depthOut.input)

right_s_out = pipeline.create(dai.node.XLinkOut)
right_s_out.setStreamName("rightS")
nodes.stereo.syncedRight.link(right_s_out.input)

left_s_out = pipeline.create(dai.node.XLinkOut)
left_s_out.setStreamName("leftS")
nodes.stereo.syncedLeft.link(left_s_out.input)

with dai.Device(pipeline) as device:
    replay.create_queues(device)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    detQ = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)
    rightS_Q = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)
    leftS_Q = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)

    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    color = (255, 0, 0)
    
    # Previous distance detected
    previous_distance = np.zeros(3)
    # Previous timestamp
    previous_time = 0
    
    fps = 30
    time_interval = 1/fps
        
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.send_frames():
        rgbFrame = replay.lastFrame['color']

        # if mono:
        cv2.imshow("left", leftS_Q.get().getCvFrame())
        cv2.imshow("right", rightS_Q.get().getCvFrame())

        depthFrame = depthQ.get().getFrame()
        depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        current_time = previous_time + time_interval
        
        inDet = detQ.tryGet()
        if inDet is not None:
            # Display (spatial) object detections on the color frame
            for detection in inDet.detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * frame_dimension[0])
                x2 = int(detection.xmax * frame_dimension[0])
                y1 = int(detection.ymin * frame_dimension[1])
                y2 = int(detection.ymax * frame_dimension[1])
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(rgbFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            # Just for first detection...
            if(len(inDet.detections) != 0):
                detection = inDet.detections[0]
                x = int(detection.xmin * frame_dimension[0])
                y = int(detection.ymin * frame_dimension[1])
                           
                # Current distance
                current_distance = np.zeros(3)
                current_distance[0] = detection.spatialCoordinates.x
                current_distance[1] = detection.spatialCoordinates.y
                current_distance[2] = detection.spatialCoordinates.z
                # Compute relative speed [m/s] 
                try:
                    speed = (current_distance-previous_distance)/(current_time-previous_time)/1000
                except:
                    speed = (current_distance-previous_distance)/1000
                # Display speed
                cv2.putText(rgbFrame, f"V_X: {float(round(speed[0],3))} m/s", (x + 10, y + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rgbFrame, f"V_Y: {float(round(speed[1],3))} m/s", (x + 10, y + 125), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rgbFrame, f"V_Z: {float(round(speed[2],3))} m/s", (x + 10, y + 140), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                
                # Save previous location and timestamp
                previous_distance = current_distance
                previous_time = current_time
                
                
                # Compute saved fuel, same hp as row 16
                if current_distance[2] != 0:
                    Cd = m_Cd * current_distance[2] + q_Cd
                    power = 1/2*reference_area*air_density*Cd*pow(absolute_speed,3)+roll_resistance_power
                    saved_fuel = (nominal_power-power) / nominal_power * 100
                else:
                    saved_fuel = 0
                # Display saved fuel
                cv2.putText(rgbFrame, f"saved fuel (hp: speed 25[m/s]): {float(round(saved_fuel,3))}%", (10, frame_dimension[0]-15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))
                
            
        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')

