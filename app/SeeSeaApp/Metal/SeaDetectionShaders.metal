#include <metal_stdlib>
using namespace metal;

// Structure to hold parameters for the sea detection computation
struct SeaDetectionParams {
    float confidence_threshold;
    uint num_classes;
    uint width;
    uint height;
};

// Structure to hold sea-related class IDs
struct SeaLabelIds {
    uint ids[8]; // Support up to 8 "sea" labels
    uint count;
};

// Structure to hold classes that should never be labeled as sea
struct PreventLabelIds {
    uint ids[8]; // Support up to 8 "prevent" labels
    uint count;
};

// This kernel function performs the sea detection computation on the GPU
// It processes a single pixel per thread, using the entire GPU for massive parallelism
kernel void seaDetectionShader(
    // The 4D tensor of logits - flattened to a 1D array for Metal compatibility
    device const float* logits [[buffer(0)]],
    // Configuration parameters
    device const SeaDetectionParams& params [[buffer(1)]],
    // Sea label IDs
    device const SeaLabelIds& seaLabels [[buffer(2)]],
    // Prevent-relabel label IDs
    device const PreventLabelIds& preventLabels [[buffer(3)]],
    // Output mask (flattened 2D array)
    device bool* mask [[buffer(4)]],
    // Current thread position (corresponds to pixel coordinates)
    uint2 position [[thread_position_in_grid]])
{
    // Check if we're within the image bounds
    if (position.x >= params.width || position.y >= params.height) {
        return;
    }
    
    // Extract coordinates for readability
    uint h = position.y;
    uint w = position.x;
    uint pixelIndex = h * params.width + w;
    
    // Find class with maximum logit value (argmax)
    float maxLogit = -INFINITY;
    uint maxClassIndex = 0;
    
    // Iterate through each class to find the maximum logit
    for (uint c = 0; c < params.num_classes; c++) {
        // Calculate the index in the flattened tensor
        // Index = batch(0) * numClasses * height * width + 
        //         class * height * width + 
        //         h * width + 
        //         w
        uint logitIndex = c * params.height * params.width + h * params.width + w;
        float logitValue = logits[logitIndex];
        
        if (logitValue > maxLogit) {
            maxLogit = logitValue;
            maxClassIndex = c;
        }
    }
    
    // Check if this pixel belongs to a prevent-relabel class
    for (uint i = 0; i < preventLabels.count; i++) {
        if (maxClassIndex == preventLabels.ids[i]) {
            // Skip this pixel, it's a class that should never be sea
            mask[pixelIndex] = false;
            return;
        }
    }
    
    // Calculate softmax denominator
    float sumExp = 0.0f;
    for (uint c = 0; c < params.num_classes; c++) {
        uint logitIndex = c * params.height * params.width + h * params.width + w;
        sumExp += exp(logits[logitIndex] - maxLogit);
    }
    
    // Calculate sea confidence as sum of probabilities of sea-related classes
    float seaConfidence = 0.0f;
    for (uint i = 0; i < seaLabels.count; i++) {
        uint labelId = seaLabels.ids[i];
        if (labelId < params.num_classes) {
            uint logitIndex = labelId * params.height * params.width + h * params.width + w;
            seaConfidence += exp(logits[logitIndex] - maxLogit) / sumExp;
        }
    }
    
    // Mark as sea if confidence exceeds threshold
    mask[pixelIndex] = seaConfidence > params.confidence_threshold;
} 