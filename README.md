This is FastAPI application which is Dockerised.

This API was created for GUI developement of Detectron2 - Mask RCNN-missile parts detection custom trained model.  

Upon setting up, this creates a API endpoint that takes image of missile as an input and gives JSON output which contains: 
Image data of masked/segmented image, names of detected classes, info about which pixel belongs to which detected class. 

When uploaded on AWS -EC2 cloud service, the service creates a public port which can be connected to the website backend for GUI developement of computer vision model. 