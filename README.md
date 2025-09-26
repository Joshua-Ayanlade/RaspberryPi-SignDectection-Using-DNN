# Sign Detection on Raspberry Pi using TensorFlow and OpenCV
<p align="justify"> This project implements real-time sign detection on a Raspberry Pi 4 using a custom-trained <b>TensorFlow Convolutional Neural Network (CNN)</b> model. The model classifies images captured from the Raspberry Pi camera into two categories: <b>Detected</b> and <b>Not Detected</b>.

The TensorFlow model is loaded using the <code>readNet</code> function from the <code>cv::dnn</code> namespace in OpenCV’s DNN module. The <code>cv::dnn::blobFromImage</code> function from the <b>DNN module</b> of the <b>OpenCV library</b> preprocesses the frames captured by the Raspberry Pi camera by resizing them to <b>100 x 100</b> pixels, normalizing them <b>(/255.0)</b>, and converting them into a 4D tensor (called a "blob"), which is the expected input format for the TensorFlow frozen model. The loaded network then uses this blob as input via <code>net.setInput(blob)</code> and performs inference with <code>net.forward()</code>. The confidence value obtained from <code>output.at<float>(0, 0)</code> is used to determine the model’s prediction.

</p>

---

### ✨ Features
✅ Real-time sign detection at high FPS on Raspberry Pi.  
✅ Binary classification ("Detected" vs "Not Detected").   
✅ Live performance monitoring (FPS, inference time).  
✅ Simple keyboard controls for image capture and quitting.  
✅ Optimized for embedded systems with limited resources.

---

### ✨ Requirements
### Hardware
- Raspberry Pi (tested on Raspberry Pi 4)
- Raspberry Pi Camera

### Software
### OpenCV Installation
#### Increase the Raspberry Pi swap size (for smooth OpenCV compilation)
 ```bash
sudo nano /etc/dphys-swapfile
set CONF_SWAPSIZE=2048

sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
```

#### Clone OpenCV Repositories
```bash
mkdir ~/opencv && cd ~/opencv
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

#### Build and Install OpenCV with contrib modules
```bash
cd ~/opencv/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
      -DBUILD_opencv_java=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=OFF \
      ..
```

#### Compile and Install
```bash
make -j4         # Adjust '4' according to your CPU cores
sudo make install
sudo ldconfig
```

#### Reset Swap size
```bash
sudo nano /etc/dphys-swapfile
set vCONF_SWAPSIZE = 100

sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
```

### RaspiCam Installation
#### Clone RaspiCam Repository
 ```bash
git clone https://github.com/cedricve/raspicam.git
```

#### Build the library
 ```bash
cd raspicam
mkdir build
cd build
cmake ..
make
```

#### Install the library
 ```bash
sudo make install
sudo ldconfig
```
---
### 📊 Results


<p align="center">
  <table>
    <!-- First Row -->
    <tr>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/cf245d02-4eba-4fcb-bc03-9c3c97859652" alt="Test Result 1" style="width: 100%;" />
        <br>
        <strong>Test Result 1</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/f6849fcf-5200-43c8-94f2-bf79ce15c092" alt="Test Result 2" style="width: 100%;" />
        <br>
        <strong>Test Result 2</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/6cd03f89-ebea-4deb-b1a2-ad941f8eedea" alt="Test Result 3" style="width: 100%;" />
        <br>
        <strong>Test Result 3</strong>
      </td>
    </tr>
    <!-- Second Row -->
    <tr>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/b2ee9e6b-200d-48b2-a0a7-eb460df845ed" alt="Test Result 4" style="width: 100%;" />
        <br>
        <strong>Test Result 4</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/6c928b23-c205-400c-af56-3c07b414261f" alt="Test Result 5" style="width: 100%;" />
        <br>
        <strong>Test Result 5</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/346bce75-0ecf-4740-a41a-1a80ad9944ba" alt="Test Result 6" style="width: 100%;" />
        <br>
        <strong>Test Result 6</strong>
      </td>
    </tr>
  </table>
</p>

---

### 🛠️ Customization
[Check out the TensorFlow File](https://github.com/Joshua-Ayanlade/sign_identification_tensorflow)
#### Sequential API
 ```bash
#Classifier Model
dropout_rate = 0.2
regularization_rate = 0.01
model = tf.keras.Sequential([InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)), #Specifies the shape input data shape

                             #Feature Extraction Layer
                             Conv2D(filters = 6, kernel_size = 3, strides = 1, padding='valid', activation='relu', kernel_regularizer= l2(regularization_rate)),      #ConvNet layer with 6 outputs(neurons)- each contains the kernel and bias
                             BatchNormalization(),
                             MaxPool2D(pool_size = 2,strides = 2),    #max pooling layer - selects the maximum value from each kernel window
                             Dropout(dropout_rate),

                             Conv2D(filters = 10, kernel_size = 3, strides = 1, padding='valid', activation='relu', kernel_regularizer= l2(regularization_rate)),      #ConvNet layer
                             BatchNormalization(),
                             MaxPool2D(pool_size = 2,strides = 2),    #max pooling layer
                             Dropout(dropout_rate),

                             
                             Flatten(), #flatten layer - converts 2D feature to 1D vector

                             #Classification Layer
                             Dense(15, activation = 'relu', kernel_regularizer= l2(regularization_rate)),      #dense layer with 10 neurons
                             Dropout(dropout_rate),
                             
                             Dense(1, activation = 'sigmoid'),      #Output layer (for 2-class classification)
]) 
```
