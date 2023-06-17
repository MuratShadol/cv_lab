UPDATED: image_classification_transfer.py. Results below
The fined_tunned file shows the neural network and its training

The file top_layer_training describes the training of the top layer

The image_classification_transfer.py file describes the program code

Found 14044 files belonging to 6 classes.
Found 3002 files belonging to 6 classes.
Model: "model_1"

_________________________________________________________________
 Layer (type)               Output Shape             Param #   
=================================================================
 input_4 (InputLayer)        [(None, 150, 150, 3)]     0         
                                                                 
 sequential_1 (Sequential)   (None, 150, 150, 3)       0         
                                                                 
 rescaling_1 (Rescaling)     (None, 150, 150, 3)       0         
                                                                 
 xception (Functional)       (None, 5, 5, 2048)        20861480  
                                                                 
 global_average_pooling2d_1   (None, 2048)             0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dropout_1 (Dropout)         (None, 2048)              0         
                                                                 
 dense_1 (Dense)             (None, 6)                 12294     
                                                                 
=================================================================
Total params: 20,873,774
Trainable params: 12,294
Non-trainable params: 20,861,480
_________________________________________________________________
Epoch 1/100
439/439 [==============================] - 1152s 3s/step - loss: 0.4520 - accuracy: 0.8357 - val_loss: 0.3177 - val_accuracy: 0.8827

Epoch 2/100
439/439 [==============================] - 33s 76ms/step - loss: 0.3466 - accuracy: 0.8735 - val_loss: 0.2946 - val_accuracy: 0.8971

Epoch 3/100
439/439 [==============================] - 32s 74ms/step - loss: 0.3147 - accuracy: 0.8825 - val_loss: 0.2830 - val_accuracy: 0.8927

Epoch 4/100
439/439 [==============================] - 37s 85ms/step - loss: 0.3087 - accuracy: 0.8884 - val_loss: 0.2899 - val_accuracy: 0.8921

Epoch 5/100
439/439 [==============================] - 33s 74ms/step - loss: 0.3126 - accuracy: 0.8849 - val_loss: 0.2787 - val_accuracy: 0.8991

Epoch 6/100
439/439 [==============================] - 33s 74ms/step - loss: 0.3032 - accuracy: 0.8891 - val_loss: 0.2753 - val_accuracy: 0.8987

Epoch 7/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2906 - accuracy: 0.8935 - val_loss: 0.3028 - val_accuracy: 0.8884

Epoch 8/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2923 - accuracy: 0.8931 - val_loss: 0.2847 - val_accuracy: 0.8994

Epoch 9/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2920 - accuracy: 0.8912 - val_loss: 0.2751 - val_accuracy: 0.8977

Epoch 10/100
439/439 [==============================] - 37s 85ms/step - loss: 0.2901 - accuracy: 0.8934 - val_loss: 0.2806 - val_accuracy: 0.8981

Epoch 11/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2812 - accuracy: 0.8960 - val_loss: 0.2763 - val_accuracy: 0.8977

Epoch 12/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2818 - accuracy: 0.8956 - val_loss: 0.2732 - val_accuracy: 0.9021

Epoch 13/100
439/439 [==============================] - 37s 85ms/step - loss: 0.2786 - accuracy: 0.8973 - val_loss: 0.2892 - val_accuracy: 0.8961

Epoch 14/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2786 - accuracy: 0.8955 - val_loss: 0.2766 - val_accuracy: 0.9027

Epoch 15/100
439/439 [==============================] - 38s 86ms/step - loss: 0.2794 - accuracy: 0.8958 - val_loss: 0.2774 - val_accuracy: 0.8971

Epoch 16/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2827 - accuracy: 0.8941 - val_loss: 0.2822 - val_accuracy: 0.8987

Epoch 17/100
439/439 [==============================] - 32s 74ms/step - loss: 0.2681 - accuracy: 0.8989 - val_loss: 0.2735 - val_accuracy: 0.9034

Epoch 18/100
439/439 [==============================] - 33s 76ms/step - loss: 0.2724 - accuracy: 0.9010 - val_loss: 0.2838 - val_accuracy: 0.9004

Epoch 19/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2771 - accuracy: 0.8997 - val_loss: 0.2664 - val_accuracy: 0.8991

Epoch 20/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2726 - accuracy: 0.8975 - val_loss: 0.2765 - val_accuracy: 0.9031

Epoch 21/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2717 - accuracy: 0.9008 - val_loss: 0.2778 - val_accuracy: 0.9011

Epoch 22/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2775 - accuracy: 0.8957 - val_loss: 0.2889 - val_accuracy: 0.8877

Epoch 23/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2685 - accuracy: 0.9003 - val_loss: 0.3056 - val_accuracy: 0.8901

Epoch 24/100
439/439 [==============================] - 38s 86ms/step - loss: 0.2764 - accuracy: 0.8964 - val_loss: 0.2816 - val_accuracy: 0.8981

Epoch 25/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2734 - accuracy: 0.8948 - val_loss: 0.2675 - val_accuracy: 0.9031

Epoch 26/100
439/439 [==============================] - 33s 75ms/step - loss: 0.2712 - accuracy: 0.8989 - val_loss: 0.2963 - val_accuracy: 0.8957

Epoch 27/100
439/439 [==============================] - 33s 74ms/step - loss: 0.2750 - accuracy: 0.8987 - val_loss: 0.2699 - val_accuracy: 0.9057

Epoch 28/100
439/439 [==============================] - 37s 85ms/step - loss: 0.2678 - accuracy: 0.9020 - val_loss: 0.2783 - val_accuracy: 0.8984

Epoch 29/100
439/439 [==============================] - 37s 85ms/step - loss: 0.2796 - accuracy: 0.8980 - val_loss: 0.2768 - val_accuracy: 0.9041

Model: "model_1"
_________________________________________________________________
 Layer (type)               Output Shape             Param #   
=================================================================
 input_4 (InputLayer)        [(None, 150, 150, 3)]     0         
                                                                 
 sequential_1 (Sequential)   (None, 150, 150, 3)       0         
                                                                 
 rescaling_1 (Rescaling)     (None, 150, 150, 3)       0         
                                                                 
 xception (Functional)       (None, 5, 5, 2048)        20861480  
                                                                 
 global_average_pooling2d_1   (None, 2048)             0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dropout_1 (Dropout)         (None, 2048)              0         
                                                                 
 dense_1 (Dense)             (None, 6)                 12294     
                                                                 
=================================================================
Total params: 20,873,774
Trainable params: 20,819,246
Non-trainable params: 54,528
_________________________________________________________________
Epoch 1/1000
439/439 [==============================] - 142s 248ms/step - loss: 0.2357 - accuracy: 0.9104 - val_loss: 0.2361 - val_accuracy: 0.9154

Epoch 2/1000
439/439 [==============================] - 105s 240ms/step - loss: 0.2015 - accuracy: 0.9275 - val_loss: 0.2101 - val_accuracy: 0.9177

Epoch 3/1000
439/439 [==============================] - 105s 239ms/step - loss: 0.1697 - accuracy: 0.9371 - val_loss: 0.2013 - val_accuracy: 0.9277

Epoch 4/1000
439/439 [==============================] - 105s 240ms/step - loss: 0.1524 - accuracy: 0.9433 - val_loss: 0.1969 - val_accuracy: 0.9277

Epoch 5/1000
439/439 [==============================] - 110s 250ms/step - loss: 0.1332 - accuracy: 0.9516 - val_loss: 0.2058 - val_accuracy: 0.9284

Epoch 6/1000
439/439 [==============================] - 105s 239ms/step - loss: 0.1204 - accuracy: 0.9554 - val_loss: 0.1974 - val_accuracy: 0.9290

Epoch 7/1000
439/439 [==============================] - 105s 239ms/step - loss: 0.1096 - accuracy: 0.9575 - val_loss: 0.2040 - val_accuracy: 0.9280

Epoch 8/1000
439/439 [==============================] - 105s 240ms/step - loss: 0.0932 - accuracy: 0.9645 - val_loss: 0.2182 - val_accuracy: 0.9314

Epoch 9/1000
439/439 [==============================] - 105s 239ms/step - loss: 0.0917 - accuracy: 0.9675 - val_loss: 0.2205 - val_accuracy: 0.9287

Epoch 10/1000
439/439 [==============================] - 110s 250ms/step - loss: 0.0798 - accuracy: 0.9709 - val_loss: 0.2148 - val_accuracy: 0.9310

Epoch 11/1000
439/439 [==============================] - 110s 250ms/step - loss: 0.0721 - accuracy: 0.9719 - val_loss: 0.2378 - val_accuracy: 0.9317

Epoch 12/1000
439/439 [==============================] - 110s 251ms/step - loss: 0.0647 - accuracy: 0.9773 - val_loss: 0.2337 - val_accuracy: 0.9257

Epoch 13/1000
439/439 [==============================] - 105s 240ms/step - loss: 0.0568 - accuracy: 0.9786 - val_loss: 0.2425 - val_accuracy: 0.9247

Epoch 14/1000
439/439 [==============================] - 105s 240ms/step - loss: 0.0517 - accuracy: 0.9791 - val_loss: 0.2453 - val_accuracy: 0.9317


