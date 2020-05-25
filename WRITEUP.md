# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

=> First of all openvino Toolkit support various different various framework i.e tensorflow, caffe models,ONNX
=> You need to check for supported layes for a model when running the model optimizer by adding a an extension to both model optimizer and inference engine
=> To add custom layer:
1. For tensor flow and caffe models: Register an extensions to the to their model optimizer and inference engine.
2. Caffe only: Use caffe to calculate the output shape
3. Tensorflow only: Replace the subgraph with another
4. Offload computation to tensorflow for tensorflow model

i.e cpu extn for linux is:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_xxxx.so 

you can also follow the steps at github for your model specifics:-https://github.com/david-drew/OpenVINO-Custom-Layers/blob/master/2019.r2.0/ReadMe.Linux.2019.r2.md

Some of the potential reasons for handling custom layers are...

=> Custom layers are used when certain layers in a model are not supported in the OpenVino Toolkit
=> By not handling your custom layers your program may crush since the model optimizer has no idea about the unsupported layers hence its upto you to make sure you handle them first

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

=> The pre-conversion accuracy for all the models i used were quite good but after conversion to IR the accuracy for post-coversion model was high compared to pre-conversion



=> The size of the model pre-conversion model was 142mb while the size after conversion,
post-conversion is the .bin file plus the .xml file was only 50.925mb

=> The minimum inference time for the post-converted model was abit high at about 567ms while the maximum was about 933ms

=> The CPU Overhead of the model pre- and post-conversion was...

=>The cpu overhead for pre-conversion was at 65% while the cpu overhead of the post-conversion was around 40% per core

=> It is to run your edge app in local network compared to running in at the server since you need to have a host to connet to.
=> The cost for hiring a server to deploy your app is much expensive and also you may incurr network fluctuations compare to to deploying your app at the edge which requires only a local connection

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1. Can be used in banks,supermarkets,malls etc to monitor the number of people in the banking halls
2. Can be used to detect if the limit of number of people required is reach in a certain place
3. Can allow only specific gender to enter a certain area
4. Can be used to detect intrusion of thieves or un authorised people to enter a certain area with certain gardgets.



Each of these use cases would be useful because...

=> we can set the total number required to enter acertain area, and when that number is reached we send an alert message or an alarm - this can be useful especially this time of COVID-19 restrictions to number of people at a certain place
=> we can set only specic gender to enter a certain place, when unauthorised gender is detected it sends notification or an alarm
=> protect intrusion since it allows only authorised personel

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

=> Lighting is essential since it allows to provide clear picture or image since its difficult to detect image at the dark
=> The more the model accuracy the better the perfomance at the edge,the poor the accuracy it leads faulty of the app.
=> Camera focal length is useful since poor quality camera with low focal legth and a better model with the best accuracy it leads to poor perfomance since it covers a small area. For bigger spaces,a camera with high focal length is recomended at all times for the best output
- camera with high focal length can be able tomagnify a small image size to give distinctive features


## Submission Details

# Generating IR files

# Step 1: Download model
Download the pre-trained model from here:- http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

by command:- wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

# Step 2: Extract files
Extract the files:-
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

# Step 3: Convert to Intermediate Presentation
cd faster_rcnn_inception_v2_coco_2018_01_28

then run the model optimizer
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --reverse_input_channel
```

# Step 4
You can create a new file directory in the app directory and move the generated .xml and .bin file into created model directory.or just leave the .xml file and .bin file in the
faster_rcnn_inception_v2_coco_2018_01_28 folder and redirect the file path directory to it

# Setting up for UI app
**To install the user interface 
=> run on the worspace
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

**Running the app for CPU
```
python3 main.py -i resources/Pedestrain_Detect_2_1_1.mp4 -m ./faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```
To see the output on a web based interface, open the link [http://localhost:8080](http://localhost:8080/) in a browser.

or click open open if on udacity workspace



