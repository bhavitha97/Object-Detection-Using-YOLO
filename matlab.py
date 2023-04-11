import cv2
import numpy as np
import os

def object_detection(obj):
    # Load Yolo
    netCV = cv2.dnn.readNet("configs/yolov3-tiny.weights", "configs/yolov3-tiny.cfg")

    vidPaths = []
    for fl in os.listdir("videos"):
        if fl.endswith(".mp4"):
            path=os.path.join("videos", fl)
            vidPaths.append(path)

    cocoClasses = []
    with open("configs/coco.names", "r") as f:
        cocoClasses = [line.strip() for line in f.readlines()]


    total_frames = []
    net_layer_names = netCV.getLayerNames()
    net_output_layers = [net_layer_names[i - 1] for i in netCV.getUnconnectedOutLayers()]

    total_deduces = []
    for i in vidPaths:
        # Loading image
        cap = cv2.VideoCapture(i)
        frames = []
        urlStr = str(i)

        counter = 0
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        while cap.isOpened():
            # Extract the frame
            ret, frame1 = cap.read()
            if not ret:
                continue

            frame = cv2.rotate(frame1, cv2.ROTATE_180)
            # Write the results back to output location.
            cv2.imwrite("/deducedFrames" + urlStr + "/%#05d.jpg" % (counter + 1), frame)
            frames.append(frame)
            counter = counter + 1
            # If there are no more frames left
            if (counter > (video_length - 1)):
                # Release the feed
                cap.release()
                # Print stats
                print("Done extracting frames.\n%d frames extracted" % counter)
                break

        total_frames.append(frames)
        deduce_labels = []
        for i in frames:
            # Detecting objects
            blob = cv2.dnn.blobFromImage(i, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            height, width, channels = i.shape
            netCV.setInput(blob)
            outps = netCV.forward(net_output_layers)
            # Showing informations on the screen
            label_ids = []
            confidences_list = []
            bounding_boxes = []
            for out in outps:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[3] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 1.8)
                        y = int(center_y - h / 1.8)

                        bounding_boxes.append([x, y, w, h])
                        confidences_list.append(float(confidence))
                        label_ids.append(class_id)

            idx = cv2.dnn.NMSBoxes(bounding_boxes, confidences_list, 0.4, 0.3)

            for i in range(len(bounding_boxes)):
                if i in idx:
                    x, y, w, h = bounding_boxes[i]
                    label = str(cocoClasses[label_ids[i]])
                    if label not in deduce_labels:
                        deduce_labels.append(label)
                    confidence = confidences_list[i]

        total_deduces.append(deduce_labels)

    foundURL = []
    flag = 0
    for i in range(0, len(total_deduces)):
        for j in total_deduces[i]:
            if (obj == j and flag == 0):
                foundURL.append(vidPaths[i])
                flag = 1
        flag = 0

    return foundURL

urls = object_detection('car')

print(urls)
