import CoreML
import Vision
import UIKit

class SeaDetector {
    // MARK: - Properties
    
    private let model: MLModel
    private let considerAsSea: [String]
    private let preventRelabelIfTop: [String]
    private let confidenceThreshold: Float
    private let labelToId: [String: Int]
    private let idToLabel: [Int: String]
    
    // Precomputed lookup values - calculated once during initialization
    private let seaLabelIds: [Int]
    private let preventLabelIdSet: Set<Int>
    
    // Input dimensions for the model - hardcoded to 224x224
    private let _inputWidth: Int = 224
    private let _inputHeight: Int = 224
    
    // Make input dimensions accessible
    var inputWidth: Int { return _inputWidth }
    var inputHeight: Int { return _inputHeight }
    
    // MARK: - Initialization
    
    init() throws {
        print("Initializing SeaDetector...")
        
        // List all resources in the bundle to help with debugging
        print("Available resources:")
        let resources = Bundle.main.paths(forResourcesOfType: "", inDirectory: nil)
        for resource in resources {
            print("   - \(resource)")
        }
        
        // First try to find the compiled model
        if let modelURL = Bundle.main.url(forResource: "sea_segmentation_base", withExtension: "mlmodelc") {
            print("Found compiled model at: \(modelURL.path)")
            do {
                self.model = try MLModel(contentsOf: modelURL)
                print("Compiled model loaded successfully")
            } catch {
                print("Error loading compiled model: \(error.localizedDescription)")
                throw error
            }
        }
        // If that fails, try to find the package
        else if let modelURL = Bundle.main.url(forResource: "sea_segmentation_base", withExtension: "mlpackage") {
            print("Found model package at: \(modelURL.path)")
            do {
                self.model = try MLModel(contentsOf: modelURL)
                print("Model package loaded successfully")
            } catch {
                print("Error loading model package: \(error.localizedDescription)")
                throw error
            }
        }
        // If we get here, we couldn't find the model
        else {
            print("Failed to find sea_segmentation_base model in the bundle")
            throw SeaDetectorError.modelNotFound
        }
        
        // No need to read input dimensions from model since we're hardcoding to 224x224
        print("Using hardcoded model input dimensions: 224x224")
        
        // Load sea configuration
        guard let configURL = Bundle.main.url(forResource: "sea_config", withExtension: "json") else {
            print("Failed to find sea_config.json in the bundle")
            throw SeaDetectorError.modelNotFound
        }
        
        print("Found config at: \(configURL.path)")
        let configData = try Data(contentsOf: configURL)
        let seaConfig = try JSONDecoder().decode(SeaConfig.self, from: configData)
        
        self.considerAsSea = seaConfig.consider_as_sea
        self.preventRelabelIfTop = seaConfig.prevent_relabel_if_top
        self.confidenceThreshold = seaConfig.confidence_threshold
        
        print("Loaded sea config: consider as sea: \(seaConfig.consider_as_sea)")
        print("Prevent relabel if top: \(seaConfig.prevent_relabel_if_top)")
        print("Confidence threshold: \(seaConfig.confidence_threshold)")
        
        // Load label mappings - these are specific to the ADE20K dataset used by SegFormer
        // The key labels we need for sea detection
        self.labelToId = [
            "sea": 26,
            "lake": 128,
            "sky": 2
        ]
        
        // Create reverse mapping
        var idToLabel = [Int: String]()
        for (label, id) in labelToId {
            idToLabel[id] = label
        }
        self.idToLabel = idToLabel
        
        // Fix 'self' captured warnings - store in local variables first
        let labelsToId = self.labelToId
        let considerSea = seaConfig.consider_as_sea
        let preventRelabel = seaConfig.prevent_relabel_if_top
        
        // Precompute label lookups at initialization time instead of every function call
        self.seaLabelIds = considerSea.compactMap { labelsToId[$0] }
        let preventLabelIds = preventRelabel.compactMap { labelsToId[$0] }
        self.preventLabelIdSet = Set(preventLabelIds)
        
        print("Precomputed sea label IDs: \(seaLabelIds)")
        print("Precomputed prevent label IDs: \(preventLabelIdSet)")
    }
    
    // MARK: - Public Methods
    
    /// Detect sea percentage using a preprocessed MLMultiArray
    /// - Parameter multiArray: The preprocessed MLMultiArray
    /// - Returns: A tuple containing the sea percentage and whether the image contains sea
    func detectSea(multiArray: MLMultiArray, minSeaFraction: Float = 0.2) throws -> (percentage: Float, containsSea: Bool) {
        // Create model input
        let input = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": multiArray])
        
        // Run inference
        let inferenceStartTime = CFAbsoluteTimeGetCurrent()
        let outputFeatures = try model.prediction(from: input)
        let inferenceEndTime = CFAbsoluteTimeGetCurrent()
        let inferenceTime = inferenceEndTime - inferenceStartTime
        print("[SeaDetector] Model inference completed in \(String(format: "%.3f", inferenceTime)) seconds")
        
        // Get logits from output
        guard let logitsMultiArray = outputFeatures.featureValue(for: "logits")?.multiArrayValue else {
            throw SeaDetectorError.invalidOutput
        }
        
        // Convert to a more usable format and apply post-processing
        let postprocessStartTime = CFAbsoluteTimeGetCurrent()
        let logits = convertMultiArrayToArray(logitsMultiArray)
        
        // Apply post-processing
        let seaPercentage = applySeaPostProcessing(logits: logits)
        let postprocessEndTime = CFAbsoluteTimeGetCurrent()
        let postprocessTime = postprocessEndTime - postprocessStartTime
        print("[SeaDetector] Post-processing completed in \(String(format: "%.3f", postprocessTime)) seconds")
        
        // Determine if the image contains sea
        let containsSea = seaPercentage > minSeaFraction
        
        return (seaPercentage, containsSea)
    }
    
    /// Generate a visualization of the sea mask
    /// - Parameter image: The input image
    /// - Returns: An image with the sea mask overlay
    func generateSeaMaskVisualization(for image: UIImage) throws -> UIImage {
        // Preprocess the image
        let preprocessStartTime = CFAbsoluteTimeGetCurrent()
        let pixelBuffer = try preprocessImage(image)
        let preprocessEndTime = CFAbsoluteTimeGetCurrent()
        let preprocessTime = preprocessEndTime - preprocessStartTime
        print("[SeaDetector] Visualization preprocessing completed in \(String(format: "%.3f", preprocessTime)) seconds")
        
        return try generateSeaMaskVisualization(multiArray: pixelBuffer, originalImage: image)
    }
    
    /// Generate a visualization of the sea mask using a preprocessed MLMultiArray
    /// - Parameters:
    ///   - multiArray: The preprocessed MLMultiArray
    ///   - originalImage: The original image to overlay the mask on
    /// - Returns: An image with the sea mask overlay
    func generateSeaMaskVisualization(multiArray: MLMultiArray, originalImage: UIImage) throws -> UIImage {
        // Create model input
        let input = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": multiArray])
        
        // Run inference
        let inferenceStartTime = CFAbsoluteTimeGetCurrent()
        let outputFeatures = try model.prediction(from: input)
        let inferenceEndTime = CFAbsoluteTimeGetCurrent()
        let inferenceTime = inferenceEndTime - inferenceStartTime
        print("[SeaDetector] Visualization model inference completed in \(String(format: "%.3f", inferenceTime)) seconds")
        
        // Get logits from output
        guard let logitsMultiArray = outputFeatures.featureValue(for: "logits")?.multiArrayValue else {
            throw SeaDetectorError.invalidOutput
        }
        
        // Post-processing steps
        let postprocessStartTime = CFAbsoluteTimeGetCurrent()
        
        // Convert to a more usable format
        let logits = convertMultiArrayToArray(logitsMultiArray)
        
        // Generate binary mask
        let binaryMask = generateSeaMask(from: logits)
        
        // Resize mask to match original image
        let resizedMask = resizeMask(binaryMask, to: originalImage.size)
        
        // Create overlay image
        let result = createOverlayImage(originalImage: originalImage, mask: resizedMask)
        
        let postprocessEndTime = CFAbsoluteTimeGetCurrent()
        let postprocessTime = postprocessEndTime - postprocessStartTime
        print("[SeaDetector] Visualization post-processing completed in \(String(format: "%.3f", postprocessTime)) seconds")
        
        return result
    }
    
    // MARK: - Private Methods
    
    /// Preprocess an image for the sea detection model
    /// - Parameter image: The input image
    /// - Returns: An MLMultiArray ready for model input
    func preprocessImage(_ image: UIImage) throws -> MLMultiArray {
        // Use the shared preprocessing method
        return try image.preprocessForML(targetSize: CGSize(width: inputWidth, height: inputHeight))
    }
    
    private func convertMultiArrayToArray(_ multiArray: MLMultiArray) -> [[[[Float]]]] {
        // Get dimensions
        let batchSize = multiArray.shape[0].intValue
        let numClasses = multiArray.shape[1].intValue
        let height = multiArray.shape[2].intValue
        let width = multiArray.shape[3].intValue
        
        // Create 4D array
        var result = [[[[Float]]]](repeating: [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: width), count: height), count: numClasses), count: batchSize)
        
        // Fill the array
        for b in 0..<batchSize {
            for c in 0..<numClasses {
                for h in 0..<height {
                    for w in 0..<width {
                        let index = b * numClasses * height * width + c * height * width + h * width + w
                        result[b][c][h][w] = multiArray[index].floatValue
                    }
                }
            }
        }
        
        return result
    }
    
    private func applySeaPostProcessing(logits: [[[[Float]]]]) -> Float {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Get dimensions (fix unused variable warning)
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Process the logits and get the binary mask using shared code
        let binaryMask = processLogitsToSeaMask(logits: logits)
        
        // Compute sea percentage
        var seaPixels = 0
        let totalPixels = height * width
        
        // Use a flattened approach to count sea pixels
        for row in binaryMask {
            // Use a faster way to count true values in a Bool array
            seaPixels += row.lazy.filter { $0 }.count
        }
        
        let seaPercentage = Float(seaPixels) / Float(totalPixels)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] Sea percentage calculation completed in \(String(format: "%.3f", endTime - startTime)) seconds")
        
        return min(max(seaPercentage, 0.0), 1.0)
    }
    
    private func generateSeaMask(from logits: [[[[Float]]]]) -> [[Bool]] {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Use the shared processing function
        let binaryMask = processLogitsToSeaMask(logits: logits)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] Mask generation completed in \(String(format: "%.3f", endTime - startTime)) seconds")
        
        return binaryMask
    }
    
    /// Shared processing function to avoid code duplication and optimize performance
    private func processLogitsToSeaMask(logits: [[[[Float]]]]) -> [[Bool]] {
        // Get dimensions from the model output tensor
        // logits shape is [batch, classes, height, width]
        // Fix unused variable warning with underscore
        _ = logits.count        // Should be 1
        let numClasses = logits[0].count    // Number of semantic classes 
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Preallocate result array only - we'll compute values on the fly
        var binaryMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        
        // Process each pixel in a single pass
        for h in 0..<height {
            for w in 0..<width {
                // Find class with maximum logit value (argmax operation)
                // Start with class 0 as the initial candidate
                var maxLogit = logits[0][0][h][w]
                var maxClassIndex = 0
                
                // Compare against all other classes
                for c in 1..<numClasses {
                    if logits[0][c][h][w] > maxLogit {
                        maxLogit = logits[0][c][h][w]
                        maxClassIndex = c
                    }
                }
                
                // Skip early if this pixel belongs to a class we don't want to relabel
                // (e.g., sky which should never be classified as sea)
                if preventLabelIdSet.contains(maxClassIndex) {
                    continue
                }
                
                // Compute softmax probabilities
                // First, calculate denominator (sum of all exp values)
                var sumExp: Float = 0
                for c in 0..<numClasses {
                    sumExp += exp(logits[0][c][h][w] - maxLogit)
                }
                
                // Compute sea confidence as the sum of probabilities of all sea-related classes
                var seaConfidence: Float = 0
                for labelId in seaLabelIds {
                    if labelId < numClasses {
                        // Calculate probability using softmax formula
                        seaConfidence += exp(logits[0][labelId][h][w] - maxLogit) / sumExp
                    }
                }
                
                // Mark this pixel as sea if confidence exceeds threshold
                if seaConfidence > confidenceThreshold {
                    binaryMask[h][w] = true
                }
            }
        }
        
        return binaryMask
    }
    
    private func resizeMask(_ mask: [[Bool]], to size: CGSize) -> [[Bool]] {
        let originalHeight = mask.count
        let originalWidth = mask[0].count
        let targetHeight = Int(size.height)
        let targetWidth = Int(size.width)
        
        // Early exit if no resizing is needed
        if originalHeight == targetHeight && originalWidth == targetWidth {
            return mask
        }
        
        var resizedMask = [[Bool]](repeating: [Bool](repeating: false, count: targetWidth), count: targetHeight)
        
        // Precompute scaling factors
        let scaleY = Float(originalHeight) / Float(targetHeight)
        let scaleX = Float(originalWidth) / Float(targetWidth)
        
        // Precompute source coordinates mapping to avoid redundant calculations
        var srcYMap = [Int](repeating: 0, count: targetHeight)
        var srcXMap = [Int](repeating: 0, count: targetWidth)
        
        for y in 0..<targetHeight {
            srcYMap[y] = min(Int(Float(y) * scaleY), originalHeight - 1)
        }
        
        for x in 0..<targetWidth {
            srcXMap[x] = min(Int(Float(x) * scaleX), originalWidth - 1)
        }
        
        // Apply the mapping efficiently
        for y in 0..<targetHeight {
            let srcY = srcYMap[y]
            for x in 0..<targetWidth {
                let srcX = srcXMap[x]
                resizedMask[y][x] = mask[srcY][srcX]
            }
        }
        
        return resizedMask
    }
    
    private func createOverlayImage(originalImage: UIImage, mask: [[Bool]]) -> UIImage {
        let startTime = CFAbsoluteTimeGetCurrent()
        let width = Int(originalImage.size.width)
        let height = Int(originalImage.size.height)
        
        // Use a more efficient approach with UIGraphicsImageRenderer
        let renderer = UIGraphicsImageRenderer(size: originalImage.size)
        
        let resultImage = renderer.image { context in
            // Draw original image
            originalImage.draw(at: .zero)
            
            // Set the overlay color
            context.cgContext.setFillColor(UIColor.red.withAlphaComponent(0.5).cgColor)
            
            // Prepare path for more efficient drawing
            let path = CGMutablePath()
            
            // Process mask row by row and create paths for efficiency
            for y in 0..<min(height, mask.count) {
                // Track consecutive true pixels for batching
                var currentRun: (start: Int, length: Int)? = nil
                
                for x in 0..<min(width, mask[y].count) {
                    if mask[y][x] {
                        if let run = currentRun {
                            // Extend current run
                            currentRun = (run.start, run.length + 1)
                        } else {
                            // Start new run
                            currentRun = (x, 1)
                        }
                    } else if let run = currentRun {
                        // End of a run, add rectangle to path
                        path.addRect(CGRect(x: run.start, y: y, width: run.length, height: 1))
                        currentRun = nil
                    }
                }
                
                // Handle run that extends to the end of the row
                if let run = currentRun {
                    path.addRect(CGRect(x: run.start, y: y, width: run.length, height: 1))
                }
            }
            
            // Draw all sea areas at once
            context.cgContext.addPath(path)
            context.cgContext.fillPath()
        }
        
        let endTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] Overlay creation completed in \(String(format: "%.3f", endTime - startTime)) seconds")
        
        return resultImage
    }
}

// MARK: - Supporting Types

struct SeaConfig: Decodable {
    let consider_as_sea: [String]
    let prevent_relabel_if_top: [String]
    let confidence_threshold: Float
    
    // Map snake_case JSON keys to camelCase properties
    enum CodingKeys: String, CodingKey {
        case consider_as_sea
        case prevent_relabel_if_top
        case confidence_threshold
    }
    
    // Computed properties for more Swift-like naming
    var considerAsSea: [String] { return consider_as_sea }
    var preventRelabelIfTop: [String] { return prevent_relabel_if_top }
    var confidenceThreshold: Float { return confidence_threshold }
}

enum SeaDetectorError: Error {
    case modelNotFound
    case preprocessingFailed
    case invalidOutput
} 
