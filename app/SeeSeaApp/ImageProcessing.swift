import UIKit
import CoreML
import Vision

// MARK: - UIImage Extensions

extension UIImage {
    func resize(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { context in
            // Set interpolation quality to high (similar to bicubic)
            context.cgContext.interpolationQuality = .high
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
    
    func toPixelBuffer() throws -> CVPixelBuffer {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        var pixelBuffer: CVPixelBuffer?
        // Creating the pixel buffer as BGRA but we will treat it as RGBA
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_32BGRA,
                                       nil,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            let error = NSError(domain: "ImageProcessing", code: -4, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
            throw error
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
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        
        context?.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
    
    func preprocessForML(targetSize: CGSize, normalize: Bool = true) throws -> MLMultiArray {
        // Resize image to target dimensions
        guard let resizedImage = self.resize(to: targetSize),
              let pixelBuffer = try? resizedImage.toPixelBuffer() else {
            throw ImageProcessingError.preprocessingFailed
        }
        
        // Convert to MLMultiArray with normalization
        return try MLMultiArray(pixelBuffer: pixelBuffer, normalize: normalize)
    }
}

// MARK: - MLMultiArray Extensions

extension MLMultiArray {
    convenience init(pixelBuffer: CVPixelBuffer, normalize: Bool = true) throws {
        // Lock the buffer to ensure we can access its memory
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        // Get dimensions
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create MLMultiArray with shape [1, 3, height, width] for RGB (channels-first format)
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Define normalization parameters as Double
        let imageMean = [0.485, 0.456, 0.406]  // RGB order
        let imageStd = [0.229, 0.224, 0.225]   // RGB order
        
        // Get pixel data
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            let error = NSError(domain: "MLMultiArray", code: -1, 
                         userInfo: [NSLocalizedDescriptionKey: "Couldn't get pixel buffer base address"])
            throw error
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bufferSize = CVPixelBufferGetDataSize(pixelBuffer)
        
        // Create a buffer view
        let buffer = UnsafeBufferPointer(start: baseAddress.assumingMemoryBound(to: UInt8.self),
                                       count: bufferSize)
        
        // Copy and normalize pixels
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4 // 4 bytes per pixel (RGBA)
                
                // Extract RGB values (normalized to [0,1])
                let rNorm = Double(buffer[offset]) / 255.0
                let gNorm = Double(buffer[offset + 1]) / 255.0
                let bNorm = Double(buffer[offset + 2]) / 255.0
                
                // Apply ImageNet normalization if requested
                let r = normalize ? (rNorm - imageMean[0]) / imageStd[0] : rNorm
                let g = normalize ? (gNorm - imageMean[1]) / imageStd[1] : gNorm
                let b = normalize ? (bNorm - imageMean[2]) / imageStd[2] : bNorm
                
                // Set values in MLMultiArray in RGB order
                mlArray[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)  // Red
                mlArray[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)  // Green
                mlArray[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)  // Blue
            }
        }
        
        // Initialize self with the shape and data from mlArray
        do {
            try self.init(shape: shape, dataType: .float32)
            
            // Copy data from mlArray to self
            for i in 0..<mlArray.count {
                self[i] = mlArray[i]
            }
        } catch {
            print("Error initializing MLMultiArray: \(error.localizedDescription)")
            throw error
        }
    }
}

// MARK: - CGImagePropertyOrientation Extension

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

// MARK: - Error Types

enum ImageProcessingError: Error {
    case preprocessingFailed
    case pixelBufferCreationFailed
    case multiArrayConversionFailed
} 