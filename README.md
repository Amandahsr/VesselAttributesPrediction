# Vessel Type and Cargo Status Predictor
A multi-model workflow that detects vessel objects in optical satellite imagery, and predicts vessel type and cargo-bearing status of detected vessel. 

## Dataset
Leveraging on optical satellite imagery data from open-sourced [xView 2018 object detection challenge](https://challenge.xviewdataset.org/welcome), a total of 8571 vessel objects across 846 images was used in a 70:15:15 split for training, validation and testing. Annotations relate to a total of 10 vessel types:
<img width="237" height="277" alt="VesselClasses" src="https://github.com/user-attachments/assets/b847250b-0fb4-4067-ad02-25c1c6f09c8e" />

## Multi-model workflow
Transfer learning was first used on a pretrained YOLOv8 model to fine-tune for vessel detection and vessel type classification. Additional techniques including tiling and NMS is used to improve small vessel object detection and reduce redundant overlapping bounding box estimations: 
<img width="689" height="383" alt="DetectorModelExample" src="https://github.com/user-attachments/assets/49f67535-5253-43d4-a92f-000079929ebd" />

A pretrained CLIP model is then used for zero-shot cargo-bearing status prediction on detected vessel objects. The final prediction outputs bounding box dimensions of detected vessel objects, the cargo status and CLIP's cargo status confidence scores for each detected vessel object in an image:
<img width="836" height="420" alt="FinalPredictionExample" src="https://github.com/user-attachments/assets/dacca07b-e076-4958-80ce-f53c2c0f019d" />

## Considerations and Limitations
This model workflow is specifically trained on optical satellite imagery data, and can only be used for RGB satellite images in TIFF format. It is unadvisable to use this workflow on other satellite data modalities (e.g. SAR imagery).

This multi-model workflow is fine-tuned on a small set of images, and can be further improved by leveraging on data augmentation techniques. Additionally, ground-truth vessel type labels provided in the xView data is noisy and lacks the granularity required for vessel type classification. 

For instance, a) all vessel types can be correctly considered as "Maritime Vessel". Ground-truth annotations for vessel objects labelled as "Maritime Vessel" in the xView dataset actually comprises of different vessel classes, making it difficult for the YOLO model to differentiate between other maritime vessels and other vessel subtypes:
<img width="359" height="294" alt="Screenshot 2026-06-24 at 4 40 52 PM" src="https://github.com/user-attachments/assets/ff670da0-796b-46aa-8970-35873cc930c9" />

b) Many visible vessel objects are not annotated in the xView dataset, making evaluation of model performance difficult:
<img width="906" height="433" alt="FalsePostiveExample" src="https://github.com/user-attachments/assets/2103a02a-ef3e-4a16-b0bb-47e20dd36ebb" />
