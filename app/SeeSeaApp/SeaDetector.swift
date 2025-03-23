import CoreML
import Vision
import UIKit

class SeaDetector {
    // Configuration
    static var enableVisualization: Bool = true  // Set to false to disable visualization
    
    // MARK: - Properties
    
    private let model: MLModel
    private let confidenceThreshold: Float
    
    // Precomputed lookup values - calculated once during initialization
    private let seaLabelIds: [Int]
    private let preventLabelIdSet: Set<Int>
    
    // Input dimensions for the model - hardcoded to 224x224
    private let inputWidth: Int = 224
    private let inputHeight: Int = 224
    
    // Store visualization image
    private var visualizationImage: UIImage?
    
    // MARK: - Initialization
    
    init(modelPath: URL, config: SeaConfig) throws {
        // Load model
        guard let model = try? MLModel(contentsOf: modelPath) else {
            throw SeaDetectorError.modelNotFound
        }
        self.model = model
        
        // Initialize configuration
        self.confidenceThreshold = config.confidenceThreshold
        
        // Load label mappings - these are specific to the ADE20K dataset used by SegFormer
        // The key labels we need for sea detection
        let labelToId = [
            "sea": 26,
            "lake": 128,
            "sky": 2
        ]
        
        // Precompute lookup values
        self.seaLabelIds = config.considerAsSea.compactMap { labelToId[$0] }
        self.preventLabelIdSet = Set(config.preventRelabelIfTop.compactMap { labelToId[$0] })
    }
    
    // MARK: - Public Methods
    
    /// Detect sea in an image using the sea detection model
    /// - Parameter multiArray: The preprocessed MLMultiArray
    /// - Returns: A binary mask of the sea
    func detectSea(multiArray: MLMultiArray) throws -> [[Bool]] {
        // Create model input
        let input = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": multiArray])
        
        // Run inference
        let inferenceStartTime = CFAbsoluteTimeGetCurrent()
        let outputFeatures = try model.prediction(from: input)
        let inferenceEndTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] Model inference completed in \(String(format: "%.3f", inferenceEndTime - inferenceStartTime)) seconds")
        
        // Get logits from output
        guard let logitsMultiArray = outputFeatures.featureValue(for: "logits")?.multiArrayValue else {
            throw SeaDetectorError.invalidOutput
        }
        
        // Convert to array and process
        let logits = Self.convertMultiArrayToArray(logitsMultiArray)
        return processLogitsToSeaMask(logits: logits)
        
    }
    
    /// Generate visualization of the sea mask
    /// - Parameters:
    ///   - mask: The binary mask
    ///   - originalImage: The original image to overlay the mask on
    /// - Returns: The visualization image if visualization is enabled
     func generateSeaMaskVisualization(mask: [[Bool]], originalImage: UIImage?) throws -> UIImage? {
        
        let resizedMask = Self.resizeMask(mask, to: originalImage?.size ?? CGSize(width: inputWidth, height: inputHeight))
        let resultImage = Self.createOverlayImage(originalImage: originalImage ?? UIImage(named: "placeholder")!, mask: resizedMask)
        
        return resultImage
    }
    
    /// Calculate the fraction of a binary mask that is true
    static func calculateSeaFraction(mask: [[Bool]]) -> Float {
        let totalPixels = mask.count * mask[0].count
        let seaPixels = mask.flatMap { $0 }.filter { $0 }.count
        return Float(seaPixels) / Float(totalPixels)
    }
    
    // MARK: - Private Methods
    
    /// Preprocess an image for the sea detection model
    /// - Parameter image: The input image
    /// - Returns: An MLMultiArray ready for model input
    private func preprocessImage(_ image: UIImage) throws -> MLMultiArray {
        return try image.preprocessForML(targetSize: CGSize(width: inputWidth, height: inputHeight))
    }
    
    /// Convert an MLMultiArray to a 4D array of floats
    static func convertMultiArrayToArray(_ multiArray: MLMultiArray) -> [[[[Float]]]] {
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
    
    /// Process logits using CPU (single-threaded version)
    private func processLogitsToSeaMask(logits: [[[[Float]]]]) -> [[Bool]] {
        let processingStartTime = CFAbsoluteTimeGetCurrent()
        
        // Get dimensions from the model output tensor
        let numClasses = logits[0].count
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Preallocate result array
        var binaryMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        
        // Process each pixel sequentially
        for h in 0..<height {
            for w in 0..<width {
                // Find class with maximum logit value (argmax operation)
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
                if preventLabelIdSet.contains(maxClassIndex) {
                    continue
                }
                
                // Compute softmax probabilities
                var sumExp: Float = 0
                for c in 0..<numClasses {
                    sumExp += exp(logits[0][c][h][w] - maxLogit)
                }
                
                // Compute sea confidence as the sum of probabilities of all sea-related classes
                var seaConfidence: Float = 0
                for labelId in seaLabelIds {
                    if labelId < numClasses {
                        seaConfidence += exp(logits[0][labelId][h][w] - maxLogit) / sumExp
                    }
                }
                
                // Mark this pixel as sea if confidence exceeds threshold
                if seaConfidence > confidenceThreshold {
                    binaryMask[h][w] = true
                }
            }
        }
        
        let processingEndTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] CPU mask processing completed in \(String(format: "%.3f", processingEndTime - processingStartTime)) seconds")
        
        return binaryMask
    }
    
    static func resizeMask(_ mask: [[Bool]], to size: CGSize) -> [[Bool]] {
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
        
        // Precompute source coordinates mapping
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
    
    static func createOverlayImage(originalImage: UIImage, mask: [[Bool]]) -> UIImage {
        let startTime = CFAbsoluteTimeGetCurrent()
        let width = Int(originalImage.size.width)
        let height = Int(originalImage.size.height)
        
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
                var currentRun: (start: Int, length: Int)? = nil
                
                for x in 0..<min(width, mask[y].count) {
                    if mask[y][x] {
                        if let run = currentRun {
                            currentRun = (run.start, run.length + 1)
                        } else {
                            currentRun = (x, 1)
                        }
                    } else if let run = currentRun {
                        path.addRect(CGRect(x: run.start, y: y, width: run.length, height: 1))
                        currentRun = nil
                    }
                }
                
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
    
    enum CodingKeys: String, CodingKey {
        case consider_as_sea
        case prevent_relabel_if_top
        case confidence_threshold
    }
    
    var considerAsSea: [String] { return consider_as_sea }
    var preventRelabelIfTop: [String] { return prevent_relabel_if_top }
    var confidenceThreshold: Float { return confidence_threshold }
}

enum SeaDetectorError: Error {
    case modelNotFound
    case preprocessingFailed
    case invalidOutput
} 
