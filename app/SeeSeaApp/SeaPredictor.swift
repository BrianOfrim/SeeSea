import CoreML
import Vision
import UIKit

class SeaPredictor {
    private let seeseaModel: regression_model
    
    init() throws {
        print("Initializing SeaPredictor...")
        
        // First try to find the compiled model
        if let modelURL = Bundle.main.url(forResource: "regression_model", withExtension: "mlmodelc") {
            print("Found compiled model at: \(modelURL.path)")
            self.seeseaModel = try regression_model(contentsOf: modelURL)
            print("Model loaded successfully")
            return
        }
        
        // If that fails, try to find the package
        if let modelURL = Bundle.main.url(forResource: "regression_model", withExtension: "mlpackage") {
            print("Found model package at: \(modelURL.path)")
            self.seeseaModel = try regression_model(contentsOf: modelURL)
            print("Model loaded successfully")
            return
        }
        
        // If we get here, we couldn't find the model
        print("Could not find model. Listing available resources:")
        let resources = Bundle.main.paths(forResourcesOfType: "", inDirectory: nil)
        for resource in resources {
            print("   - \(resource)")
        }
        
        throw NSError(domain: "SeaPredictor", code: -1,
                     userInfo: [NSLocalizedDescriptionKey: "Failed to find model"])
    }
    
    func predict(image: UIImage) throws -> [Float] {
        print("Starting prediction process...")
        
        // Resize image to 224x224 (matching the model's input size)
        let targetSize = CGSize(width: 224, height: 224)
        guard let resizedImage = image.resize(to: targetSize) else {
            print("Failed to resize image")
            throw NSError(domain: "SeaPredictor", code: -2, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
        }
        print("Image resized to \(targetSize.width)x\(targetSize.height)")
        
        guard let pixelBuffer = try? resizedImage.toPixelBuffer() else {
            print("Failed to convert image to pixel buffer")
            throw NSError(domain: "SeaPredictor", code: -2, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to convert to pixel buffer"])
        }
        print("Image converted to pixel buffer")
        
        // Convert pixelBuffer to MLMultiArray
        print("Converting pixel buffer to MLMultiArray...")
        let multiArray: MLMultiArray
        do {
            multiArray = try MLMultiArray(pixelBuffer: pixelBuffer)
            print("Converted to MLMultiArray with shape: \(multiArray.shape)")
        } catch {
            print("Failed to convert to MLMultiArray: \(error.localizedDescription)")
            throw error
        }
        
        // Get predictions using the model
        print("Running model prediction...")
        let output: regression_modelOutput
        do {
            output = try seeseaModel.prediction(pixel_values: multiArray)
            print("Model prediction completed")
        } catch {
            print("Model prediction failed: \(error.localizedDescription)")
            throw error
        }
        
        // Convert MLMultiArray to [Float]
        let predictions = output.predictions
        let length = predictions.count
        let result = (0..<length).map { predictions[$0].floatValue }
        
        print("Prediction results:")
        for (index, value) in result.enumerated() {
            print("   Output \(index): \(value)")
        }
        
        if result.count >= 2 {
            print("[SeaPredictor] Wind Speed: \(result[0]) m/s")
            print("[SeaPredictor] Wave Height: \(result[1]) meters")
        }
        
        return result
    }
}

extension CGImagePropertyOrientation {
    init(_ uiOrientation: UIImage.Orientation) {
        switch uiOrientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}

extension UIImage {
    func resize(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
    
    func toPixelBuffer() throws -> CVPixelBuffer {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_32BGRA,
                                       nil,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(domain: "SeaPredictor", code: -4, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

extension MLMultiArray {
    convenience init(pixelBuffer: CVPixelBuffer) throws {
        // Lock the buffer to ensure we can access its memory
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        // Get dimensions
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create MLMultiArray with shape [1, 3, height, width] for RGB (channels-first format)
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Get pixel data
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(domain: "MLMultiArray", code: -1, 
                         userInfo: [NSLocalizedDescriptionKey: "Couldn't get pixel buffer base address"])
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bufferSize = CVPixelBufferGetDataSize(pixelBuffer)
        
        // Create a buffer view
        let buffer = UnsafeBufferPointer(start: baseAddress.assumingMemoryBound(to: UInt8.self),
                                       count: bufferSize)
        
        // Copy and normalize pixels
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4 // 4 bytes per pixel (BGRA)
                
                // Get RGB values (note: pixel format is BGRA)
                let blue = Float(buffer[offset]) / 255.0
                let green = Float(buffer[offset + 1]) / 255.0
                let red = Float(buffer[offset + 2]) / 255.0
                
                // Set values in MLMultiArray - note the changed indexing for channels-first format
                // Index calculation: [batch, channel, height, width]
                mlArray[[0, 0, y, x] as [NSNumber]] = NSNumber(value: red)
                mlArray[[0, 1, y, x] as [NSNumber]] = NSNumber(value: green)
                mlArray[[0, 2, y, x] as [NSNumber]] = NSNumber(value: blue)
            }
        }
        
        // Initialize self with the shape and data from mlArray
        try self.init(shape: shape, dataType: .float32)
        
        // Copy data from mlArray to self
        for i in 0..<mlArray.count {
            self[i] = mlArray[i]
        }
    }
} 
