import CoreML
import Vision
import UIKit
import Metal
import MetalKit

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
    
    // Metal properties for GPU acceleration
    private let metalDevice: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private var pipelineState: MTLComputePipelineState?
    private var useGPUAcceleration: Bool = false
    
    // Flag to force using CPU even when GPU is available (for performance comparison)
    private var forceCPUProcessing: Bool = false
    
    /// Public property to check if GPU acceleration is available and enabled
    var isGPUAccelerationEnabled: Bool {
        return useGPUAcceleration && !forceCPUProcessing
    }
    
    /// Toggle between CPU and GPU processing for performance comparison
    /// - Returns: A tuple containing the current processing mode and whether GPU is available
    @discardableResult
    func toggleProcessingMode() -> (mode: String, gpuAvailable: Bool) {
        forceCPUProcessing = !forceCPUProcessing
        
        let mode = isGPUAccelerationEnabled ? "GPU" : "CPU"
        print("[SeaDetector] Switched to \(mode) processing mode")
        
        return (mode, useGPUAcceleration)
    }
    
    // MARK: - Initialization
    
    init() throws {
        print("Initializing SeaDetector...")
        
        // Initialize Metal for GPU acceleration if available
        if let device = MTLCreateSystemDefaultDevice() {
            self.metalDevice = device
            self.commandQueue = device.makeCommandQueue()
            
            // Try to create the compute pipeline
            do {
                let library = try device.makeDefaultLibrary(bundle: Bundle.main)
                if let computeFunction = library.makeFunction(name: "seaDetectionShader") {
                    self.pipelineState = try device.makeComputePipelineState(function: computeFunction)
                    self.useGPUAcceleration = true
                    print("[SeaDetector] GPU acceleration enabled")
                } else {
                    print("[SeaDetector] Compute function not found, falling back to CPU")
                }
            } catch {
                print("[SeaDetector] Failed to initialize GPU pipeline: \(error.localizedDescription)")
                print("[SeaDetector] Falling back to CPU implementation")
            }
        } else {
            print("[SeaDetector] Metal not available on this device, using CPU implementation")
            self.metalDevice = nil
            self.commandQueue = nil
        }
        
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
        
        // Create the Metal shader file if it doesn't exist
        createMetalShaderFileIfNeeded()
    }
    
    // MARK: - Public Methods
    
    /// Detect sea percentage using a preprocessed MLMultiArray
    /// - Parameters:
    ///   - multiArray: The preprocessed MLMultiArray
    ///   - minSeaFraction: Minimum fraction of sea to consider the image as containing sea
    ///   - processingMode: Optional processing mode override: "cpu", "gpu", or "compare"
    /// - Returns: A tuple containing the sea percentage, whether the image contains sea, and performance data if comparison was requested
    func detectSea(multiArray: MLMultiArray, 
                   minSeaFraction: Float = 0.2,
                   processingMode: String? = nil) throws -> (percentage: Float, containsSea: Bool, performanceData: [String: Any]?) {
        // Handle processing mode override
        let originalForceCPU = forceCPUProcessing
        var performanceData: [String: Any]? = nil
        
        defer {
            // Restore original processing mode when method exits
            forceCPUProcessing = originalForceCPU
        }
        
        if let mode = processingMode {
            switch mode.lowercased() {
            case "cpu":
                forceCPUProcessing = true
                print("[SeaDetector] Forcing CPU mode for this detection")
            case "gpu":
                forceCPUProcessing = false
                print("[SeaDetector] Forcing GPU mode for this detection (if available)")
            case "compare":
                // Will run performance comparison after getting logits
                print("[SeaDetector] Will run performance comparison")
            default:
                print("[SeaDetector] Unknown processing mode '\(mode)', using current setting")
            }
        }
        
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
        
        // Convert to a more usable format
        let postprocessStartTime = CFAbsoluteTimeGetCurrent()
        let logits = convertMultiArrayToArray(logitsMultiArray)
        
        // Run performance comparison if requested
        if processingMode?.lowercased() == "compare" {
            performanceData = runPerformanceComparison(with: logits)
        }
        
        // Apply post-processing
        let seaPercentage = applySeaPostProcessing(logits: logits)
        let postprocessEndTime = CFAbsoluteTimeGetCurrent()
        let postprocessTime = postprocessEndTime - postprocessStartTime
        print("[SeaDetector] Post-processing completed in \(String(format: "%.3f", postprocessTime)) seconds")
        
        // Determine if the image contains sea
        let containsSea = seaPercentage > minSeaFraction
        
        return (seaPercentage, containsSea, performanceData)
    }
    
    /// Generate a visualization of the sea mask
    /// - Parameters:
    ///   - image: The input image
    ///   - processingMode: Optional processing mode override: "cpu", "gpu", or "compare"
    /// - Returns: A tuple containing the visualization image and performance data if comparison was requested
    func generateSeaMaskVisualization(for image: UIImage, 
                                     processingMode: String? = nil) throws -> (UIImage, [String: Any]?) {
        // Preprocess the image
        let preprocessStartTime = CFAbsoluteTimeGetCurrent()
        let pixelBuffer = try preprocessImage(image)
        let preprocessEndTime = CFAbsoluteTimeGetCurrent()
        let preprocessTime = preprocessEndTime - preprocessStartTime
        print("[SeaDetector] Visualization preprocessing completed in \(String(format: "%.3f", preprocessTime)) seconds")
        
        return try generateSeaMaskVisualization(multiArray: pixelBuffer, originalImage: image, processingMode: processingMode)
    }
    
    /// Generate a visualization of the sea mask using a preprocessed MLMultiArray
    /// - Parameters:
    ///   - multiArray: The preprocessed MLMultiArray
    ///   - originalImage: The original image to overlay the mask on
    ///   - processingMode: Optional processing mode override: "cpu", "gpu", or "compare"
    /// - Returns: A tuple containing the visualization image and performance data if comparison was requested
    func generateSeaMaskVisualization(multiArray: MLMultiArray, originalImage: UIImage, processingMode: String? = nil) throws -> (UIImage, [String: Any]?) {
        // Handle processing mode override
        let originalForceCPU = forceCPUProcessing
        var performanceData: [String: Any]? = nil
        
        defer {
            // Restore original processing mode when method exits
            forceCPUProcessing = originalForceCPU
        }
        
        if let mode = processingMode {
            switch mode.lowercased() {
            case "cpu":
                forceCPUProcessing = true
                print("[SeaDetector] Forcing CPU mode for this visualization")
            case "gpu":
                forceCPUProcessing = false
                print("[SeaDetector] Forcing GPU mode for this visualization (if available)")
            case "compare":
                // Will run performance comparison after getting logits
                print("[SeaDetector] Will run performance comparison")
            default:
                print("[SeaDetector] Unknown processing mode '\(mode)', using current setting")
            }
        }
        
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
        
        // Run performance comparison if requested
        if processingMode?.lowercased() == "compare" {
            performanceData = runPerformanceComparison(with: logits)
        }
        
        // Generate binary mask
        let binaryMask = generateSeaMask(from: logits)
        
        // Resize mask to match original image
        let resizedMask = resizeMask(binaryMask, to: originalImage.size)
        
        // Create overlay image
        let resultImage = createOverlayImage(originalImage: originalImage, mask: resizedMask)
        
        let postprocessEndTime = CFAbsoluteTimeGetCurrent()
        let postprocessTime = postprocessEndTime - postprocessStartTime
        print("[SeaDetector] Visualization post-processing completed in \(String(format: "%.3f", postprocessTime)) seconds")
        
        return (resultImage, performanceData)
    }
    
    /// Run performance comparison between CPU and GPU implementations
    /// - Parameter logits: The logits to process
    /// - Returns: A dictionary with performance metrics
    func runPerformanceComparison(with logits: [[[[Float]]]]) -> [String: Any] {
        guard useGPUAcceleration, 
              let device = metalDevice, 
              let queue = commandQueue, 
              let pipeline = pipelineState else {
            print("[SeaDetector] GPU acceleration not available for comparison")
            return ["error": "GPU acceleration not available"]
        }
        
        print("[SeaDetector] Running performance comparison...")
        
        // Run GPU implementation
        print("[SeaDetector] Testing GPU performance...")
        let gpuStartTime = CFAbsoluteTimeGetCurrent()
        let _ = processLogitsToSeaMaskGPU(logits: logits, device: device, queue: queue, pipeline: pipeline)
        let gpuEndTime = CFAbsoluteTimeGetCurrent()
        let gpuTime = gpuEndTime - gpuStartTime
        
        // Run CPU implementation
        print("[SeaDetector] Testing CPU performance...")
        let cpuStartTime = CFAbsoluteTimeGetCurrent()
        let _ = processLogitsToSeaMaskCPU(logits: logits)
        let cpuEndTime = CFAbsoluteTimeGetCurrent()
        let cpuTime = cpuEndTime - cpuStartTime
        
        // Calculate speedup
        let speedup = cpuTime / gpuTime
        
        // Create results
        let results: [String: Any] = [
            "gpuTime": gpuTime,
            "cpuTime": cpuTime,
            "speedup": speedup,
            "gpuTimeFormatted": String(format: "%.3f seconds", gpuTime),
            "cpuTimeFormatted": String(format: "%.3f seconds", cpuTime),
            "speedupFormatted": String(format: "%.2fx", speedup)
        ]
        
        // Log results
        print("=== Performance Comparison Results ===")
        print("GPU processing time: \(results["gpuTimeFormatted"] as! String)")
        print("CPU processing time: \(results["cpuTimeFormatted"] as! String)")
        print("GPU speedup: \(results["speedupFormatted"] as! String)")
        print("====================================")
        
        return results
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
        // Only use GPU if it's available AND we're not forcing CPU processing
        if useGPUAcceleration && !forceCPUProcessing, 
           let device = metalDevice, 
           let queue = commandQueue, 
           let pipeline = pipelineState {
            print("[SeaDetector] Using GPU processing")
            return processLogitsToSeaMaskGPU(logits: logits, device: device, queue: queue, pipeline: pipeline)
        } else {
            if useGPUAcceleration && forceCPUProcessing {
                print("[SeaDetector] GPU available but using CPU processing (forced for comparison)")
            } else {
                print("[SeaDetector] Using CPU processing")
            }
            return processLogitsToSeaMaskCPU(logits: logits)
        }
    }
    
    /// Process logits using GPU acceleration via Metal
    private func processLogitsToSeaMaskGPU(logits: [[[[Float]]]],
                                          device: MTLDevice,
                                          queue: MTLCommandQueue,
                                          pipeline: MTLComputePipelineState) -> [[Bool]] {
        let processingStartTime = CFAbsoluteTimeGetCurrent()
        
        // Get dimensions from the model output tensor
        let numClasses = logits[0].count    // Number of semantic classes 
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Prepare result array 
        var binaryMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        
        // Create a flat buffer for the mask result (1D array is easier to work with in Metal)
        let maskBufferSize = width * height * MemoryLayout<Bool>.stride
        guard let maskBuffer = device.makeBuffer(length: maskBufferSize, options: .storageModeShared) else {
            print("[SeaDetector] Failed to create mask buffer, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Flatten the 4D tensor to a 1D array for Metal
        // Skip the batch dimension (assuming batch size = 1)
        let flattenedLogitsBufferSize = numClasses * height * width * MemoryLayout<Float>.stride
        guard let logitsBuffer = device.makeBuffer(length: flattenedLogitsBufferSize, options: .storageModeShared) else {
            print("[SeaDetector] Failed to create logits buffer, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Get a pointer to the buffer contents
        let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: numClasses * height * width)
        
        // Flatten the 4D tensor into the 1D buffer
        for c in 0..<numClasses {
            for h in 0..<height {
                for w in 0..<width {
                    let index = c * height * width + h * width + w
                    logitsPtr[index] = logits[0][c][h][w]
                }
            }
        }
        
        let bufferConversionEndTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] Logits buffer conversion completed in \(String(format: "%.3f", bufferConversionEndTime - processingStartTime)) seconds")
        
        // Get parameters buffer with configuration
        var params = (confidenceThreshold: Float(confidenceThreshold),
                     numClasses: UInt32(numClasses),
                     width: UInt32(width),
                     height: UInt32(height))
        guard let paramsBuffer = device.makeBuffer(bytes: &params, 
                                           length: MemoryLayout.size(ofValue: params),
                                           options: .storageModeShared) else {
            print("[SeaDetector] Failed to create params buffer, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Define Swift structures that match the Metal shader layout
        struct SeaLabelIds {
            var id0: UInt32 = 0
            var id1: UInt32 = 0
            var id2: UInt32 = 0
            var id3: UInt32 = 0
            var id4: UInt32 = 0
            var id5: UInt32 = 0
            var id6: UInt32 = 0
            var id7: UInt32 = 0
            var count: UInt32 = 0
        }
        
        struct PreventLabelIds {
            var id0: UInt32 = 0
            var id1: UInt32 = 0
            var id2: UInt32 = 0
            var id3: UInt32 = 0
            var id4: UInt32 = 0
            var id5: UInt32 = 0
            var id6: UInt32 = 0
            var id7: UInt32 = 0
            var count: UInt32 = 0
        }
        
        // Create sea label IDs buffer using the aligned structure
        var seaLabelsStruct = SeaLabelIds()
        seaLabelsStruct.count = UInt32(min(seaLabelIds.count, 8))
        for (i, id) in seaLabelIds.prefix(8).enumerated() {
            switch i {
            case 0: seaLabelsStruct.id0 = UInt32(id)
            case 1: seaLabelsStruct.id1 = UInt32(id)
            case 2: seaLabelsStruct.id2 = UInt32(id)
            case 3: seaLabelsStruct.id3 = UInt32(id)
            case 4: seaLabelsStruct.id4 = UInt32(id)
            case 5: seaLabelsStruct.id5 = UInt32(id)
            case 6: seaLabelsStruct.id6 = UInt32(id)
            case 7: seaLabelsStruct.id7 = UInt32(id)
            default: break
            }
        }
        
        guard let seaLabelsBuffer = device.makeBuffer(bytes: &seaLabelsStruct,
                                              length: MemoryLayout<SeaLabelIds>.size,
                                              options: .storageModeShared) else {
            print("[SeaDetector] Failed to create sea labels buffer, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Create prevent label IDs buffer using the aligned structure
        let preventLabelIds = Array(preventLabelIdSet)
        var preventLabelsStruct = PreventLabelIds()
        preventLabelsStruct.count = UInt32(min(preventLabelIds.count, 8))
        for (i, id) in preventLabelIds.prefix(8).enumerated() {
            switch i {
            case 0: preventLabelsStruct.id0 = UInt32(id)
            case 1: preventLabelsStruct.id1 = UInt32(id)
            case 2: preventLabelsStruct.id2 = UInt32(id)
            case 3: preventLabelsStruct.id3 = UInt32(id)
            case 4: preventLabelsStruct.id4 = UInt32(id)
            case 5: preventLabelsStruct.id5 = UInt32(id)
            case 6: preventLabelsStruct.id6 = UInt32(id)
            case 7: preventLabelsStruct.id7 = UInt32(id)
            default: break
            }
        }
        
        guard let preventLabelsBuffer = device.makeBuffer(bytes: &preventLabelsStruct,
                                                  length: MemoryLayout<PreventLabelIds>.size,
                                                  options: .storageModeShared) else {
            print("[SeaDetector] Failed to create prevent labels buffer, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Output debugging information to help diagnose the issue
        print("[SeaDetector] Struct sizes: SeaLabelIds=\(MemoryLayout<SeaLabelIds>.size) bytes, PreventLabelIds=\(MemoryLayout<PreventLabelIds>.size) bytes")
        
        // Create a command buffer for GPU work
        guard let commandBuffer = queue.makeCommandBuffer() else {
            print("[SeaDetector] Failed to create command buffer, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Create a compute command encoder
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("[SeaDetector] Failed to create compute encoder, falling back to CPU")
            return processLogitsToSeaMaskCPU(logits: logits)
        }
        
        // Set the compute pipeline state
        computeEncoder.setComputePipelineState(pipeline)
        
        // Set the buffers
        computeEncoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(seaLabelsBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(preventLabelsBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(maskBuffer, offset: 0, index: 4)
        
        // Calculate the thread group size
        let threadGroupSizeX = min(pipeline.threadExecutionWidth, width)
        let threadGroupSizeY = min(pipeline.maxTotalThreadsPerThreadgroup / threadGroupSizeX, height)
        let threadGroupSize = MTLSize(width: threadGroupSizeX, height: threadGroupSizeY, depth: 1)
        
        // Calculate the number of threadgroups needed to cover the entire grid
        // Use ceiling division to ensure we cover all pixels
        let threadgroupsX = (width + threadGroupSizeX - 1) / threadGroupSizeX
        let threadgroupsY = (height + threadGroupSizeY - 1) / threadGroupSizeY
        let threadgroupCount = MTLSize(width: threadgroupsX, height: threadgroupsY, depth: 1)
        
        // Dispatch the threadgroups instead of threads directly
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadGroupSize)
        
        // End encoding
        computeEncoder.endEncoding()
        
        // Commit the command buffer
        commandBuffer.commit()
        
        // Wait for completion
        commandBuffer.waitUntilCompleted()
        
        // Convert the results back from a 1D buffer to a 2D array
        let maskPtr = maskBuffer.contents().bindMemory(to: Bool.self, capacity: width * height)
        
        for h in 0..<height {
            for w in 0..<width {
                let index = h * width + w
                binaryMask[h][w] = maskPtr[index]
            }
        }
        
        let processingEndTime = CFAbsoluteTimeGetCurrent()
        let totalTime = processingEndTime - processingStartTime
        print("[SeaDetector] GPU processing completed in \(String(format: "%.3f", totalTime)) seconds")
        
        return binaryMask
    }
    
    /// Process logits using CPU (single-threaded version)
    private func processLogitsToSeaMaskCPU(logits: [[[[Float]]]]) -> [[Bool]] {
        let processingStartTime = CFAbsoluteTimeGetCurrent()
        
        // Get dimensions from the model output tensor
        // logits shape is [batch, classes, height, width]
        _ = logits.count        // Should be 1
        let numClasses = logits[0].count    // Number of semantic classes 
        let height = logits[0][0].count
        let width = logits[0][0][0].count
        
        // Preallocate result array
        var binaryMask = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        
        print("[SeaDetector] Processing image with single-threaded CPU implementation")
        
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
        
        let processingEndTime = CFAbsoluteTimeGetCurrent()
        print("[SeaDetector] CPU mask processing completed in \(String(format: "%.3f", processingEndTime - processingStartTime)) seconds")
        
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
    
    // MARK: - GPU Acceleration Methods
    
    /// Creates the Metal shader file if it doesn't exist in the bundle
    private func createMetalShaderFileIfNeeded() {
        // Check if we already have a Metal shader in the bundle
        guard Bundle.main.path(forResource: "SeaDetectionShaders", ofType: "metal") == nil else {
            print("[SeaDetector] Metal shader file already exists in bundle")
            return
        }
        
        print("[SeaDetector] Metal shader file not found in bundle, attempting to create it")
        
        // Define the shader content
        let shaderCode = """
        #include <metal_stdlib>
        using namespace metal;

        struct SeaDetectionParams {
            float confidence_threshold;
            uint num_classes;
            uint width;
            uint height;
        };

        struct SeaLabelIds {
            uint id0;
            uint id1;
            uint id2;
            uint id3;
            uint id4;
            uint id5;
            uint id6;
            uint id7;
            uint count;
        };

        struct PreventLabelIds {
            uint id0;
            uint id1;
            uint id2;
            uint id3;
            uint id4;
            uint id5;
            uint id6;
            uint id7;
            uint count;
        };

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
            bool shouldSkip = false;
            for (uint i = 0; i < preventLabels.count; i++) {
                uint labelId = 0;
                // Access individual elements using a switch
                switch(i) {
                    case 0: labelId = preventLabels.id0; break;
                    case 1: labelId = preventLabels.id1; break;
                    case 2: labelId = preventLabels.id2; break;
                    case 3: labelId = preventLabels.id3; break;
                    case 4: labelId = preventLabels.id4; break;
                    case 5: labelId = preventLabels.id5; break;
                    case 6: labelId = preventLabels.id6; break;
                    case 7: labelId = preventLabels.id7; break;
                }
                
                if (maxClassIndex == labelId) {
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
                uint labelId = 0;
                // Access individual elements using a switch
                switch(i) {
                    case 0: labelId = seaLabels.id0; break;
                    case 1: labelId = seaLabels.id1; break;
                    case 2: labelId = seaLabels.id2; break;
                    case 3: labelId = seaLabels.id3; break;
                    case 4: labelId = seaLabels.id4; break;
                    case 5: labelId = seaLabels.id5; break;
                    case 6: labelId = seaLabels.id6; break;
                    case 7: labelId = seaLabels.id7; break;
                }
                
                if (labelId < params.num_classes) {
                    uint logitIndex = labelId * params.height * params.width + h * params.width + w;
                    seaConfidence += exp(logits[logitIndex] - maxLogit) / sumExp;
                }
            }
            
            // Mark as sea if confidence exceeds threshold
            mask[pixelIndex] = seaConfidence > params.confidence_threshold;
        }
        """
        
        // Get the path to the Metal directory we created
        let metalDir = Bundle.main.bundlePath + "/Metal"
        let metalFilePath = metalDir + "/SeaDetectionShaders.metal"
        
        // Create the Metal directory if it doesn't exist
        do {
            if !FileManager.default.fileExists(atPath: metalDir) {
                try FileManager.default.createDirectory(atPath: metalDir, withIntermediateDirectories: true)
                print("[SeaDetector] Created Metal directory at: \(metalDir)")
            }
            
            // Write the shader code to the file
            try shaderCode.write(toFile: metalFilePath, atomically: true, encoding: .utf8)
            print("[SeaDetector] Successfully created Metal shader file at: \(metalFilePath)")
            
            // Note for the developer
            print("[SeaDetector] IMPORTANT: You need to add the Metal shader file to your Xcode project")
            print("[SeaDetector] Right-click on the project in Xcode, select 'Add Files to SeeSeaApp...'")
            print("[SeaDetector] Then select the Metal/SeaDetectionShaders.metal file")
        } catch {
            print("[SeaDetector] Failed to create Metal shader file: \(error.localizedDescription)")
        }
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
