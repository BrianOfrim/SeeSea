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
    
    // Input dimensions for the model
    private let inputWidth: Int
    private let inputHeight: Int
    
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
        
        // Get model input dimensions
        let modelDescription = model.modelDescription
        if let inputFeatureDescription = modelDescription.inputDescriptionsByName["pixel_values"],
           let multiArrayConstraint = inputFeatureDescription.multiArrayConstraint {
            self.inputHeight = Int(multiArrayConstraint.shape[2])
            self.inputWidth = Int(multiArrayConstraint.shape[3])
            print("Model input dimensions: \(inputWidth)x\(inputHeight)")
        } else {
            // Default to 512x512 if we can't determine from the model
            self.inputHeight = 512
            self.inputWidth = 512
            print("Using default dimensions: 512x512")
        }
        
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
    }
    
    // MARK: - Public Methods
    
    /// Detect sea percentage in an image
    /// - Parameter image: The input image
    /// - Returns: A tuple containing the sea percentage and whether the image contains sea
    func detectSea(in image: UIImage, minSeaFraction: Float = 0.2) throws -> (percentage: Float, containsSea: Bool) {
        // Preprocess the image
        let pixelBuffer = try preprocessImage(image)
        
        // Create model input
        let input = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": pixelBuffer])
        
        // Run inference
        let outputFeatures = try model.prediction(from: input)
        
        // Get logits from output
        guard let logitsMultiArray = outputFeatures.featureValue(for: "logits")?.multiArrayValue else {
            throw SeaDetectorError.invalidOutput
        }
        
        // Convert to a more usable format
        let logits = convertMultiArrayToArray(logitsMultiArray)
        
        // Apply post-processing
        let seaPercentage = applySeaPostProcessing(logits: logits)
        
        // Determine if the image contains sea
        let containsSea = seaPercentage > minSeaFraction
        
        return (seaPercentage, containsSea)
    }
    
    /// Generate a visualization of the sea mask
    /// - Parameter image: The input image
    /// - Returns: An image with the sea mask overlay
    func generateSeaMaskVisualization(for image: UIImage) throws -> UIImage {
        // Preprocess the image
        let pixelBuffer = try preprocessImage(image)
        
        // Create model input
        let input = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": pixelBuffer])
        
        // Run inference
        let outputFeatures = try model.prediction(from: input)
        
        // Get logits from output
        guard let logitsMultiArray = outputFeatures.featureValue(for: "logits")?.multiArrayValue else {
            throw SeaDetectorError.invalidOutput
        }
        
        // Convert to a more usable format
        let logits = convertMultiArrayToArray(logitsMultiArray)
        
        // Generate binary mask
        let binaryMask = generateSeaMask(from: logits)
        
        // Resize mask to match original image
        let resizedMask = resizeMask(binaryMask, to: image.size)
        
        // Create overlay image
        return createOverlayImage(originalImage: image, mask: resizedMask)
    }
    
    // MARK: - Private Methods
    
    private func preprocessImage(_ image: UIImage) throws -> MLMultiArray {
        // Resize image to model input dimensions
        guard let resizedImage = image.resize(to: CGSize(width: inputWidth, height: inputHeight)),
              let cgImage = resizedImage.cgImage else {
            throw SeaDetectorError.preprocessingFailed
        }
        
        // Create a multi-array with shape [1, 3, height, width]
        let multiArray = try MLMultiArray(shape: [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)],
                                         dataType: .float32)
        
        // Get pixel data
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * inputWidth
        let bitsPerComponent = 8
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        
        var pixelData = [UInt8](repeating: 0, count: inputHeight * bytesPerRow)
        
        guard let context = CGContext(data: &pixelData,
                                     width: inputWidth,
                                     height: inputHeight,
                                     bitsPerComponent: bitsPerComponent,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo) else {
            throw SeaDetectorError.preprocessingFailed
        }
        
        // Draw the image in the context
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))
        
        // Fill the multi-array with normalized pixel values
        for y in 0..<inputHeight {
            for x in 0..<inputWidth {
                let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                
                // Extract RGB values
                let r = Float(pixelData[pixelIndex]) / 255.0
                let g = Float(pixelData[pixelIndex + 1]) / 255.0
                let b = Float(pixelData[pixelIndex + 2]) / 255.0
                
                // Normalize using ImageNet mean and std (used by SegFormer)
                let normalizedR = (r - 0.485) / 0.229
                let normalizedG = (g - 0.456) / 0.224
                let normalizedB = (b - 0.406) / 0.225
                
                // Set values in the multi-array (CHW format)
                multiArray[[0, 0, y, x] as [NSNumber]] = NSNumber(value: normalizedR)
                multiArray[[0, 1, y, x] as [NSNumber]] = NSNumber(value: normalizedG)
                multiArray[[0, 2, y, x] as [NSNumber]] = NSNumber(value: normalizedB)
            }
        }
        
        return multiArray
    }
    
    private func convertMultiArrayToArray(_ multiArray: MLMultiArray) -> [[[[Float]]]] {
        // Get dimensions
        let batchSize = multiArray.shape[0].intValue
        let numClasses = multiArray.shape[1].intValue
        let height = multiArray.shape[2].intValue
        let width = multiArray.shape[3].intValue
        
        print("Converting MLMultiArray with shape: [\(batchSize), \(numClasses), \(height), \(width)]")
        
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
        // Get dimensions
        let numClasses = logits[0].count
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Compute the original prediction segmentation mask
        var originalPredSeg = [[Int]](repeating: [Int](repeating: 0, count: width), count: height)
        for h in 0..<height {
            for w in 0..<width {
                var maxClassIndex = 0
                var maxValue = logits[0][0][h][w]
                
                for c in 1..<numClasses {
                    if logits[0][c][h][w] > maxValue {
                        maxValue = logits[0][c][h][w]
                        maxClassIndex = c
                    }
                }
                
                originalPredSeg[h][w] = maxClassIndex
            }
        }
        
        // Apply softmax to get probabilities
        var probabilities = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: width), count: height), count: numClasses)
        
        for h in 0..<height {
            for w in 0..<width {
                // Find max for numerical stability
                var maxLogit = logits[0][0][h][w]
                for c in 1..<numClasses {
                    maxLogit = max(maxLogit, logits[0][c][h][w])
                }
                
                // Compute exp(logit - maxLogit)
                var sumExp: Float = 0
                for c in 0..<numClasses {
                    let expValue = exp(logits[0][c][h][w] - maxLogit)
                    probabilities[c][h][w] = expValue
                    sumExp += expValue
                }
                
                // Normalize
                for c in 0..<numClasses {
                    probabilities[c][h][w] /= sumExp
                }
            }
        }
        
        // Get top n predictions
        let nTop = 10
        var topIndices = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: width), count: height), count: nTop)
        var topProbs = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: width), count: height), count: nTop)
        
        for h in 0..<height {
            for w in 0..<width {
                // Create array of (value, index) pairs
                var valueIndexPairs = [(Float, Int)]()
                for c in 0..<numClasses {
                    valueIndexPairs.append((probabilities[c][h][w], c))
                }
                
                // Sort by value in descending order
                valueIndexPairs.sort { $0.0 > $1.0 }
                
                // Fill top n arrays
                for i in 0..<min(nTop, numClasses) {
                    topProbs[i][h][w] = valueIndexPairs[i].0
                    topIndices[i][h][w] = valueIndexPairs[i].1
                }
            }
        }
        
        // Create mask for pixels that should not be relabeled
        var preventMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        for label in preventRelabelIfTop {
            if let labelId = labelToId[label] {
                for h in 0..<height {
                    for w in 0..<width {
                        if originalPredSeg[h][w] == labelId {
                            preventMask[h][w] = true
                        }
                    }
                }
            }
        }
        
        // Calculate sum of probabilities for sea-related classes
        var seaConfidenceSum = [[Float]](repeating: [Float](repeating: 0, count: width), count: height)
        for label in considerAsSea {
            if let labelId = labelToId[label] {
                for i in 0..<nTop {
                    for h in 0..<height {
                        for w in 0..<width {
                            if topIndices[i][h][w] == labelId {
                                seaConfidenceSum[h][w] += topProbs[i][h][w]
                            }
                        }
                    }
                }
            }
        }
        
        // Create mask where sea confidence exceeds threshold
        var mask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        for h in 0..<height {
            for w in 0..<width {
                if seaConfidenceSum[h][w] > confidenceThreshold && !preventMask[h][w] {
                    mask[h][w] = true
                }
            }
        }
        
        // Build binary mask for sea pixels
        var binaryMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        for label in considerAsSea {
            if let labelId = labelToId[label] {
                for h in 0..<height {
                    for w in 0..<width {
                        if originalPredSeg[h][w] == labelId {
                            binaryMask[h][w] = true
                        }
                    }
                }
            }
        }
        
        // Combine masks
        for h in 0..<height {
            for w in 0..<width {
                if mask[h][w] {
                    binaryMask[h][w] = true
                }
            }
        }
        
        // Compute sea percentage
        var seaPixels: Int = 0
        let totalPixels = height * width
        
        for h in 0..<height {
            for w in 0..<width {
                if binaryMask[h][w] {
                    seaPixels += 1
                }
            }
        }
        
        let seaPercentage = Float(seaPixels) / Float(totalPixels)
        return min(max(seaPercentage, 0.0), 1.0)
    }
    
    private func generateSeaMask(from logits: [[[[Float]]]]) -> [[Bool]] {
        // Get dimensions
        let numClasses = logits[0].count
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Compute the original prediction segmentation mask
        var originalPredSeg = [[Int]](repeating: [Int](repeating: 0, count: width), count: height)
        for h in 0..<height {
            for w in 0..<width {
                var maxClassIndex = 0
                var maxValue = logits[0][0][h][w]
                
                for c in 1..<numClasses {
                    if logits[0][c][h][w] > maxValue {
                        maxValue = logits[0][c][h][w]
                        maxClassIndex = c
                    }
                }
                
                originalPredSeg[h][w] = maxClassIndex
            }
        }
        
        // Apply softmax to get probabilities
        var probabilities = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: width), count: height), count: numClasses)
        
        for h in 0..<height {
            for w in 0..<width {
                // Find max for numerical stability
                var maxLogit = logits[0][0][h][w]
                for c in 1..<numClasses {
                    maxLogit = max(maxLogit, logits[0][c][h][w])
                }
                
                // Compute exp(logit - maxLogit)
                var sumExp: Float = 0
                for c in 0..<numClasses {
                    let expValue = exp(logits[0][c][h][w] - maxLogit)
                    probabilities[c][h][w] = expValue
                    sumExp += expValue
                }
                
                // Normalize
                for c in 0..<numClasses {
                    probabilities[c][h][w] /= sumExp
                }
            }
        }
        
        // Get top n predictions
        let nTop = 10
        var topIndices = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: width), count: height), count: nTop)
        var topProbs = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: width), count: height), count: nTop)
        
        for h in 0..<height {
            for w in 0..<width {
                // Create array of (value, index) pairs
                var valueIndexPairs = [(Float, Int)]()
                for c in 0..<numClasses {
                    valueIndexPairs.append((probabilities[c][h][w], c))
                }
                
                // Sort by value in descending order
                valueIndexPairs.sort { $0.0 > $1.0 }
                
                // Fill top n arrays
                for i in 0..<min(nTop, numClasses) {
                    topProbs[i][h][w] = valueIndexPairs[i].0
                    topIndices[i][h][w] = valueIndexPairs[i].1
                }
            }
        }
        
        // Create mask for pixels that should not be relabeled
        var preventMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        for label in preventRelabelIfTop {
            if let labelId = labelToId[label] {
                for h in 0..<height {
                    for w in 0..<width {
                        if originalPredSeg[h][w] == labelId {
                            preventMask[h][w] = true
                        }
                    }
                }
            }
        }
        
        // Calculate sum of probabilities for sea-related classes
        var seaConfidenceSum = [[Float]](repeating: [Float](repeating: 0, count: width), count: height)
        for label in considerAsSea {
            if let labelId = labelToId[label] {
                for i in 0..<nTop {
                    for h in 0..<height {
                        for w in 0..<width {
                            if topIndices[i][h][w] == labelId {
                                seaConfidenceSum[h][w] += topProbs[i][h][w]
                            }
                        }
                    }
                }
            }
        }
        
        // Create mask where sea confidence exceeds threshold
        var mask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        for h in 0..<height {
            for w in 0..<width {
                if seaConfidenceSum[h][w] > confidenceThreshold && !preventMask[h][w] {
                    mask[h][w] = true
                }
            }
        }
        
        // Build binary mask for sea pixels
        var binaryMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        for label in considerAsSea {
            if let labelId = labelToId[label] {
                for h in 0..<height {
                    for w in 0..<width {
                        if originalPredSeg[h][w] == labelId {
                            binaryMask[h][w] = true
                        }
                    }
                }
            }
        }
        
        // Combine masks
        for h in 0..<height {
            for w in 0..<width {
                if mask[h][w] {
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
        
        var resizedMask = [[Bool]](repeating: [Bool](repeating: false, count: targetWidth), count: targetHeight)
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                // Simple nearest neighbor interpolation
                let srcY = min(Int(Float(y) * Float(originalHeight) / Float(targetHeight)), originalHeight - 1)
                let srcX = min(Int(Float(x) * Float(originalWidth) / Float(targetWidth)), originalWidth - 1)
                
                resizedMask[y][x] = mask[srcY][srcX]
            }
        }
        
        return resizedMask
    }
    
    private func createOverlayImage(originalImage: UIImage, mask: [[Bool]]) -> UIImage {
        let width = Int(originalImage.size.width)
        let height = Int(originalImage.size.height)
        
        UIGraphicsBeginImageContextWithOptions(originalImage.size, false, originalImage.scale)
        
        // Draw original image
        originalImage.draw(at: .zero)
        
        // Create overlay context
        let context = UIGraphicsGetCurrentContext()!
        context.setFillColor(UIColor.red.withAlphaComponent(0.5).cgColor)
        
        // Draw mask
        for y in 0..<height {
            for x in 0..<width {
                if y < mask.count && x < mask[y].count && mask[y][x] {
                    context.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        let resultImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
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
