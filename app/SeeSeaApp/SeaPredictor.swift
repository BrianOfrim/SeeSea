import CoreML
import Vision
import UIKit

class SeaPredictor {
    private let seeseaModel: regression_model
    
    init(modelPath: URL) throws {
        self.seeseaModel = try regression_model(contentsOf: modelPath)
    }
    
    func predict(image: UIImage) throws -> [Float] {
        print("Starting prediction process...")
        
        // Resize image to 224x224 (matching the model's input size)
        let targetSize = CGSize(width: 224, height: 224)
        
        // Use the shared preprocessing method
        let preprocessStartTime = CFAbsoluteTimeGetCurrent()
        let multiArray: MLMultiArray
        do {
            multiArray = try image.preprocessForML(targetSize: targetSize)
            let preprocessEndTime = CFAbsoluteTimeGetCurrent()
            let preprocessTime = preprocessEndTime - preprocessStartTime
            print("[SeaPredictor] Preprocessing completed in \(String(format: "%.3f", preprocessTime)) seconds")
        } catch {
            print("Failed to preprocess image: \(error.localizedDescription)")
            throw error
        }
        
        return try predict(multiArray: multiArray)
    }
    
    /// Predict using a preprocessed MLMultiArray
    /// This method allows reusing a preprocessed input
    func predict(multiArray: MLMultiArray) throws -> [Float] {
        // Get predictions using the model
        print("Running model prediction...")
        let inferenceStartTime = CFAbsoluteTimeGetCurrent()
        let output: regression_modelOutput
        do {
            output = try seeseaModel.prediction(pixel_values: multiArray)
            let inferenceEndTime = CFAbsoluteTimeGetCurrent()
            let inferenceTime = inferenceEndTime - inferenceStartTime
            print("[SeaPredictor] Model inference completed in \(String(format: "%.3f", inferenceTime)) seconds")
        } catch {
            print("Model prediction failed: \(error.localizedDescription)")
            throw error
        }
        
        // Convert MLMultiArray to [Float]
        let postprocessStartTime = CFAbsoluteTimeGetCurrent()
        let predictions = output.predictions
        let length = predictions.count
        let result = (0..<length).map { predictions[$0].floatValue }
        
        if result.count >= 2 {
            print("[SeaPredictor] Wind Speed: \(result[0]) m/s")
            print("[SeaPredictor] Wave Height: \(result[1]) meters")
        }
        
        let postprocessEndTime = CFAbsoluteTimeGetCurrent()
        let postprocessTime = postprocessEndTime - postprocessStartTime
        print("[SeaPredictor] Post-processing completed in \(String(format: "%.3f", postprocessTime)) seconds")
        
        return result
    }
}

