import cv2
import time

def main():

    # Get all frames from video
    frames = cv2.VideoCapture('video.mp4')
    
    # Trained XML classifiers 
    car_cascade = cv2.CascadeClassifier('cars.xml')
    pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
    bus_cascade = cv2.CascadeClassifier('bus.xml')

    # Create counter variables
    total_car_count = 0
    total_pedestrian_count = 0
    total_bus_count = 0
    num_frames = 0
    total_processing_time = 0.0  
    fastest_frame_time = float('inf')  
    slowest_frame_time = 0.0

    # Process frames until the video ends
    while True:

        # get single frame from frames
        frame_exist, frame = frames.read()
        
        # Break loop no more frames are returned from frames.read()
        if not frame_exist:
            break

        # Frame counter
        num_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Start timer 
        start = time.time()

        # Detect Objects
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        pedestrian = pedestrian_cascade.detectMultiScale(gray, 1.1, 1)
        bus = bus_cascade.detectMultiScale(gray, 1.1, 1)

        # Finish timer    
        end = time.time()
        
        processing_time = end - start
        total_processing_time += processing_time  

        # Update fastest and slowest frame times
        if processing_time < fastest_frame_time:
            fastest_frame_time = processing_time
        if processing_time > slowest_frame_time:
            slowest_frame_time = processing_time

        # Total count of objects
        car_count = len(cars)
        pedestrian_count = len(pedestrian)
        bus_count = len(bus)

        # Add to total counts
        total_car_count += car_count
        total_pedestrian_count += pedestrian_count
        total_bus_count += bus_count

        # Draw rectangles around detected objects
        for (x, y, w, h) in cars:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

        for (x, y, w, h) in pedestrian:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for (x, y, w, h) in bus:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Display frames in a window 
        cv2.imshow('Video', frames)

        # Total counts of objecst per frame
        print("Frame {} - Cars detected: {}, Pedestrians detected: {}, Buses detected: {}".format(num_frames, car_count, pedestrian_count, bus_count))

    cv2.destroyAllWindows()

    print("Total # of Frames: ", num_frames)
    print("Total objects detected - Cars: {}, Pedestrians: {}, Buses: {}".format(total_car_count, total_pedestrian_count, total_bus_count))

    # Total Processing time
    print("Total Processing Time for all frames: {:.2f} seconds".format(total_processing_time))
    
    # Fastest and Slowest Frame time
    print("Fastest frame processing time: {:.2f} seconds".format(fastest_frame_time))
    print("Slowest frame processing time: {:.2f} seconds".format(slowest_frame_time))

    # Average processing time
    average_processing_time = total_processing_time / num_frames
    print("Average processing time per frame: {:.2f} seconds".format(average_processing_time))

if __name__ == "__main__":
    main()
