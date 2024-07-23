# traffic-net-situation-classification-prediction
Traffic-Net is a dataset of traffic images, collected in order to ensure that machine learning systems can be trained to detect traffic conditions and provide real-time monitoring, analytics and alerts.    

Background

As traffic congestion has spread world-wide, this is basically about traffic classification model based on migration learning; i.e trained model on an old domain to be used on a new domain due to lacking of the other models, in terms of expense, time consumption and difficulty. They chose fine-tuning method when dataset adapts a pre-training model

Impact

If we can collect enough data to accurately describe and classify urban traffic conditions and analyze the main causes of traffic congestion and secondary accidents, especially when traffic emergencies occur, rapid access to information is a key factor in organizing an optimal response; therefore, monitoring the area of the effective detection of traffic status is crucial for road traffic management 

Architecture

The model used is Simple neural network model. Input shape given was 128*128, in RGB. The used dense layers are 2, first of which is of 64 then 32 and then output is 4, which is the number of classes. Activation functions used in it are ReLU and Softmax. Optimizer used is Adam.

To fit the model we have used, batch size of 32, and epochs 30.
