# Ant-trail

Project developed during the Transatlantic AI Hackathon 2022, Sustainable [Sustainable Supply Chain DeepHack](https://ultrahack.org/sustainable-supply-chain).  
The idea has been previously studied by team TruckPooling in the Start-Up lab 2022 edition, held by [CLab Trento](https://clabtrento.it/en), and presented in the [Demo day](https://clabtrento.it/en/events/2022/demoday-2022) on 26th May 2022.

## Description

The idea is to reduce green house gases emission of trucks implementing a semi-autonomous driving system, allowing the vehicles to drive close to each other and exploting a *wind tunnel* effect. This would lead to an heavy reduction of the air drag affecting the truck, reducing consumption and emisisons. Thus, this system would rise the efficiency of the vehicle, resulting in lower costs, and reducing pollution (for combustion-engine equipped trucks).

## Implementation

This project aims to build a simple plug-and-run system able to detect vehicle in front of the truck, exploting the [OAK-D-Lite](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html) kindly provided by [Luxonis](https://www.luxonis.com/).  
Thanks to this device, we were able to rapidly combine a vehicle detection system based on artificial intelligence, and a stereo-camera system to detect distance and relative velocity with respect to the truck on which is mounted on.  
- [real-time](/code/antTrail.py) run the algorithm in real time.
- [recording](/code/antTrail_recording.py) run the algorithm on a registration done with OAK-D-Lite.
- [truckpooling](/code/truckpooling.py) demo used during the Start-Up lab 2022 Demo day. It runs the same algorithm, but using a model to detect persons instead of vehicles.

### Reference code

- [record and replay](https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay)
- [spatial detection](https://github.com/luxonis/depthai-python/blob/main/examples/spatialDetection)

### Pre-trained models

- [vehicle-detection-0202](https://docs.openvino.ai/latest/omz_models_model_vehicle_detection_0202.html)
- [vehicle-license-plate-detection-barrier-0106](https://docs.openvino.ai/latest/omz_models_model_vehicle_license_plate_detection_barrier_0106.html)
- [person-detection-retail-0013](https://docs.openvino.ai/latest/omz_models_model_person_detection_retail_0013.html)

## Setup

First, you need to install the [DepthAI](https://docs.luxonis.com/projects/api/en/latest/install/#installation) library and its dependencies.

Install the requirements for the recording example:

```
python3 -m pip install -r requirements.txt
```

For detecting vehicle from a recorded video  (place the video in the recordings folder)
```
python3 antTrail_recording.py

```

For detecting vehicle from from live camera stream

```
python3 antTrail.py

```


## UI Dashboard
![Alt text](UI_dashboard.jpg?raw=true "UI Dashboard available to truck drivers and stakeholders")
