import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from imutils.video import FPS

CONF_THRESH, NMS_THRESH = 0.05, 0.3

# Global state for detection/tracking
tracking = False  # Will be set to True once tracking starts
tracker = None    # Will hold the tracker object
tracked_bbox = None
clicked_point = None
b_boxes_current = []
latest_img = None
myColor = (20, 20, 230)

def detect_annotate(img, net, classes):
    """
    Run YOLO detection on the frame and annotate bounding boxes.
    Also populate the b_boxes_current list for potential tracking.
    """
    global b_boxes_current
    b_boxes_current = []

    cv2.putText(img, 'detecting...', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (192, 192), swapRB=False, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([frame_w, frame_h, frame_w, frame_h])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    if len(b_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
        for index in indices:
            x, y, w, h = b_boxes[index]
            b_boxes_current.append([x, y, w, h])
            cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 230), 2)
            cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, myColor, 2)

# TODO: Create a function called create_tracker(tracker_type="KCF") that returns an OpenCV tracker object.
#       You may use one of the following tracker types:
#       "KCF", "CSRT", "MIL", "TLD", "BOOSTING", "MEDIANFLOW", "MOSSE"
#       Use `cv2.legacy.TrackerTYPE_create()` to create the tracker.
#       Return the tracker object.

# TODO: Create a function called initialize_tracker that:
#       - Takes in the current image, list of bounding boxes, click location, and tracker type.
#       - Checks if the click is inside any box.
#       - If so, creates and initializes a tracker on that box.
#       - Returns the tracker and the bounding box selected.

def click_event(event, x, y, flags, param):
    """
    Mouse click handler: If a click lands in a bounding box, initialize tracking.
    """
    global tracking, tracker, clicked_point, tracked_bbox, latest_img

    if event == cv2.EVENT_LBUTTONDOWN and not tracking:
        clicked_point = (x, y)

        # TODO: Use your initialize_tracker function here.
        # tracker, tracked_bbox = initialize_tracker(latest_img, b_boxes_current, clicked_point, tracker_type="KCF")

        # if tracker:
        #     tracking = True
        #     print("Switched to tracking mode.")
        # else:
        #     print("Click was not inside any detected object.")

if __name__ == '__main__':
    in_weights = 'yolov4-tiny.weights'
    in_config = 'yolov4-tiny.cfg'
    name_file = 'classes.txt'

    print("Classes:")
    with open(name_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes)

    print("Loading network...")
    net = cv2.dnn.readNetFromDarknet(in_config, in_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    print("Network loaded.")

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    image_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    image_intrinsics = image_profile.get_intrinsics()
    frame_w, frame_h = image_intrinsics.width, image_intrinsics.height
    print('image: {} w  x {} h pixels'.format(frame_w, frame_h))

    # Setup window
    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", click_event)

    fps_track = FPS().start()

    try:
        while True:
            timer = cv2.getTickCount()
            fps_track.update()

            frameset = pipeline.wait_for_frames()
            frame = frameset.get_color_frame()
            if not frame:
                print('missed frame...')
                continue

            img = np.asanyarray(frame.get_data())

            # TODO: If tracking is True and tracker exists:
            #       - Call tracker.update(img)
            #       - If successful: draw green rectangle and label 'Tracking...'
            #       - If not successful: label 'Lost tracking!' and reset tracking = False

            # If not tracking, run detection
            detect_annotate(img, net, classes)

            latest_img = img.copy()

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(img, '{:.0f} fps'.format(fps), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)

            cv2.imshow("Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping RealSense pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
        if fps_track:
            fps_track.stop()
            print("Elapsed time: {:.2f}".format(fps_track.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps_track.fps()))
