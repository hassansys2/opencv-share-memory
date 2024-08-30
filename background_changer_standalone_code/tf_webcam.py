import cv2
import numpy as np
import tensorflow as tf
import time

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='selfie_segmentation_landscape.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to run inference with profiling
def run_segmentation(input_frame):
    # Resize the frame to the input size expected by the model
    input_frame_resized = cv2.resize(input_frame, (256, 144))

    # Normalize the input image to [0, 1]
    input_frame_normalized = input_frame_resized.astype(np.float32) / 255.0

    # Add batch dimension
    input_data = np.expand_dims(input_frame_normalized, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Start profiling
    start_time = time.perf_counter()

    # Run inference
    interpreter.invoke()

    # End profiling
    end_time = time.perf_counter()

    # Get output tensor (segmentation mask)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Remove batch dimension and resize to original frame size
    mask = cv2.resize(output_data[0, :, :, 0], (input_frame.shape[1], input_frame.shape[0]))

    # Calculate inference time in seconds
    inference_time_sec = end_time - start_time

    return mask, inference_time_sec


# Function to apply post-processing on the mask
def post_process_mask(_mask):
    # Convert mask to binary
    binary_mask = np.where(_mask > 0.5, 1, 0).astype(np.uint8)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)

    # Dilation followed by erosion (closing) to fill small holes
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Erosion followed by dilation (opening) to remove noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Apply Gaussian blur to smooth the edges
    smoothed_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)

    return smoothed_mask


# Function to apply a new background
def apply_new_background(input_frame, mask, background):
    # Convert the mask to a 3-channel mask for RGB channels
    binary_mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Resize background to match the input frame size
    background_resized = cv2.resize(background, (input_frame.shape[1], input_frame.shape[0]))

    # Composite the new background
    output_frame = input_frame * binary_mask_3c + background_resized * (1 - binary_mask_3c)

    return output_frame


# Load the background image
background_image = cv2.imread('background_image.jpg')  # Replace with your background image path

# Open webcam stream
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# Variables for calculating running average of inference time
total_inference_time = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure the frame is 480x640 RGB
    # frame_resized = cv2.resize(frame, (640, 480))

    # Run segmentation on the current frame
    mask, inference_time = run_segmentation(frame)

    # Apply post-processing on the mask
    processed_mask = post_process_mask(mask)

    # Apply the new background
    output_frame = apply_new_background(frame, processed_mask, background_image)

    # Update the running average of inference time
    frame_count += 1
    total_inference_time += inference_time
    average_inference_time = (total_inference_time / frame_count) * 1000  # Convert to milliseconds

    # Display the running average of inference time on the output frame
    cv2.putText(output_frame, f"Avg Inference Time: {average_inference_time:.2f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert the mask to a visible format (grayscale to RGB)
    visible_mask = (mask * 255).astype(np.uint8)
    visible_mask_rgb = cv2.cvtColor(visible_mask, cv2.COLOR_GRAY2RGB)

    # Convert the processed_mask to a visible format (grayscale to RGB)
    visible_processed_mask = (processed_mask * 255).astype(np.uint8)
    visible_processed_mask_rgb = cv2.cvtColor(visible_processed_mask, cv2.COLOR_GRAY2RGB)

    # Show the output frame and the mask in separate windows
    cv2.imshow('Webcam with Background Replacement', output_frame)
    cv2.imshow('Segmentation Mask', visible_mask_rgb)
    cv2.imshow('Segmentation Processed Mask', visible_processed_mask_rgb)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
