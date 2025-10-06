import tensorflow as tf
import numpy as np
import cv2
import base64
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_gradcam_with_prediction(model, img_array, alpha=0.6):
    """
    Generates a Grad-CAM heatmap and prediction result for Sequential models.
    
    Args:
        model (tf.keras.Model): The trained Keras model.
        img_array (np.ndarray): The preprocessed image array ready for the model
                                (shape should be (1, H, W, C)).
        alpha (float): Transparency factor for the heatmap overlay.
    
    Returns:
        dict: A dictionary containing:
            - prediction_result (str): "Pneumonia" or "Normal"
            - confidence (float): Confidence percentage
            - gradcam_heatmap (str): Base64 encoded overlay image
    
    Raises:
        ValueError: If no Conv2D layer is found in the model
        RuntimeError: If Grad-CAM computation fails
    """
    try:
        # Step 1: Get prediction result
        logger.info("Getting prediction...")
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Use optimized threshold for better pneumonia detection (reduces false negatives)
        PNEUMONIA_THRESHOLD = 0.4
        
        if confidence > PNEUMONIA_THRESHOLD:
            prediction_result = "Pneumonia"
            confidence_percent = confidence * 100
        else:
            prediction_result = "Normal"
            confidence_percent = (1 - confidence) * 100
        
        logger.info(f"Prediction: {prediction_result} ({confidence_percent:.1f}%)")
        
        # Step 2: Find the last convolutional layer
        last_conv_layer_name = None
        last_conv_layer_index = None
        
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                last_conv_layer_index = i
        
        if last_conv_layer_name is None:
            layer_types = [f"{i}: {type(layer).__name__}" for i, layer in enumerate(model.layers)]
            raise ValueError(f"No Conv2D layer found in model. Layers: {layer_types}")
        
        logger.info(f"Using last Conv2D layer: {last_conv_layer_name} (index {last_conv_layer_index})")
        
        # Step 3: Create a new model that outputs both conv layer and final prediction
        # Build the model first by calling it
        _ = model(img_array, training=False)
        
        # Now create the grad model
        inputs = tf.keras.Input(shape=img_array.shape[1:])
        
        # Rebuild the model structure
        x = inputs
        for i in range(last_conv_layer_index + 1):
            x = model.layers[i](x)
        conv_output = x
        
        # Continue to final output
        for i in range(last_conv_layer_index + 1, len(model.layers)):
            x = model.layers[i](x)
        final_output = x
        
        grad_model = tf.keras.Model(inputs, [conv_output, final_output])
        logger.info("Grad-CAM model created successfully")
        
        # Step 4: Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Use the predicted class
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Step 5: Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise RuntimeError("Failed to compute gradients for Grad-CAM")
        
        # Step 6: Pool the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Step 7: Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Step 8: Normalize heatmap
        heatmap = tf.maximum(heatmap, 0)
        max_heat = tf.reduce_max(heatmap)
        if max_heat == 0:
            raise RuntimeError("Heatmap contains only zeros")
        
        heatmap = heatmap / max_heat
        heatmap = heatmap.numpy()
        
        # Step 9: Process original image
        original_img = np.uint8(img_array[0] * 255)
        if len(original_img.shape) == 3 and original_img.shape[-1] == 1:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
        # Step 10: Resize and apply heatmap
        target_size = (original_img.shape[1], original_img.shape[0])
        heatmap_resized = cv2.resize(heatmap, target_size)
        
        # Step 11: Apply colormap and overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed_img = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
        
        # Step 12: Convert to base64
        pil_img = Image.fromarray(superimposed_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info(f"Grad-CAM generated successfully")
        
        return {
            'prediction_result': prediction_result,
            'confidence': confidence_percent,
            'gradcam_heatmap': img_base64
        }
        
    except Exception as e:
        logger.error(f"Grad-CAM computation failed: {str(e)}")
        raise RuntimeError(f"Grad-CAM computation failed: {str(e)}") from e