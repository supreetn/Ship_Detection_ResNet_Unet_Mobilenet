# Ship_Detection_ResNet_Unet_Mobilenet
Airbus satellite ship detection Kaggle challenge using Deep Learning models. 
Here’s the backstory: Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.

Airbus offers comprehensive maritime monitoring services by building a meaningful solution for wide coverage, fine details, intensive monitoring, premium reactivity and interpretation response. Combining its proprietary-data with highly-trained analysts, they help to support the maritime industry to increase knowledge, anticipate threats, trigger alerts, and improve efficiency at sea.
https://www.kaggle.com/c/airbus-ship-detection

**Evaluation**

This competition is evaluated on the F2 Score at different intersection over union (IoU) thresholds. The metric sweeps over a range of IoU thresholds, at each point calculating an F2 Score. The threshold values range from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value , the F2 Score value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects. A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average F2 Score of a single image is then calculated as the mean of the above F2 Score values at each IoU threshold
Lastly, the score returned by the competition metric is the mean taken over the individual average F2 Scores of each image in the test dataset.
Finally the output csv has the location of pixels containing the ships in the particular image.

# Training the U-Net Model:
I set the parameters and paths, defines the data processing functions, and creates a U-Net model for image segmentation.
Later used a pre-trained ResNet-50 model for feature extraction and combines it with the U-Net architecture to create a final model.
Loss functions (Dice coefficient, binary cross-entropy) and a mean Intersection over Union (IoU) metric are defined.
The data is divided into training, validation, and test sets.
- Model Training:
  
The model is compiled with an optimizer (RectifiedAdam), loss functions, and metrics.
Training is performed using the fit method, with early stopping and model checkpointing callbacks.
The training process includes multiple epochs.
- Retraining:
  
There's a section for retraining the model from a specific epoch (e.g., epoch 4), which loads the weights from a checkpoint and continues training.

- Post this, the **Intersection over Union** is calculated along with accuracy and obtained about 82% accuracy after retraining the model.

- In summary, the code loads image and mask data, creates a U-Net model with a ResNet-50 backbone for image segmentation, and trains the model to segment objects (ships) in the images using a combination of loss functions and the mean Intersection over Union (IoU) metric. The model is trained in multiple epochs, with checkpoints to save the best model, and it can be retrained from a specific epoch if needed. The goal is to create a model that accurately segments ships in the given images.


