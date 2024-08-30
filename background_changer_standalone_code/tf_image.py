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


# Define a function to run inference with profiling
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

    # Convert to milliseconds and microseconds
    inference_time_ms = inference_time_sec * 1000  # milliseconds
    inference_time_us = inference_time_sec * 1_000_000  # microseconds

    print(f"Inference time: {inference_time_ms:.4f} ms ({inference_time_us:.2f} µs)")

    return mask


# Function to apply a new background
def apply_new_background(input_frame, mask, background):
    # Threshold the mask to create a binary mask
    binary_mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

    # Create a 3-channel mask for RGB channels
    binary_mask_3c = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

    # Resize background to match the input frame size
    background_resized = cv2.resize(background, (input_frame.shape[1], input_frame.shape[0]))

    # Composite the new background
    output_frame = input_frame * binary_mask_3c + background_resized * (1 - binary_mask_3c)

    return output_frame


# Load the input image and background
input_image = cv2.imread('input_image.jpg')  # Replace with your image path
background_image = cv2.imread('background_image.jpg')  # Replace with your background image path

# Ensure the input image is 480x640 RGB
input_image = cv2.resize(input_image, (640, 480))

# Run segmentation on the input image with profiling
mask = run_segmentation(input_image)

# Apply the new background
output_image = apply_new_background(input_image, mask, background_image)

# Save or display the output image
cv2.imwrite('output_image.jpg', output_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
