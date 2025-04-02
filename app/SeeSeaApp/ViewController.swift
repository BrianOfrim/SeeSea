import UIKit
import AVFoundation

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var imageView: UIImageView!
    private var predictionLabel: UILabel!
    private var predictionOverlayView: UIView!
    private var activityIndicator: UIActivityIndicatorView!
    private var toggleButton: UIButton!
    private var segmentControl: UISegmentedControl!
    private var selectButton: UIButton!
    private var originalImage: UIImage?
    private var lastProcessedMode: Int = 2  // default to Live Camera mode
    
    // Camera capture properties
    private var captureSession: AVCaptureSession?
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var processingQueue = DispatchQueue(label: "videoProcessingQueue")
    
    // Frame processing queue - serializes processing to avoid race conditions
    private lazy var frameProcessingQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "FrameProcessingQueue"
        queue.maxConcurrentOperationCount = 1 // Serial queue
        return queue
    }()
    
    // Dedicated main queue work for UI updates
    private let mainQueue = DispatchQueue.main
    
    // Store last capture time
    private var lastFrameCaptureTime: CFTimeInterval = 0
    
    // Track visualization state
    private var showingVisualization = true

    // Add new property for overlay layer
    private var overlayLayer: CALayer?

    // Model properties
    private var seaPredictor: SeaPredictor?
    private var seaDetector: SeaDetector?

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set up UI components
        setupUI()
        
        // Load models
        loadModels()
        
        // Start camera session
        startCameraSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Cancel all pending frame processing
        frameProcessingQueue.cancelAllOperations()
        
        // Remove the sample buffer delegate to prevent callbacks to a deallocated instance
        videoDataOutput?.setSampleBufferDelegate(nil, queue: nil)
        
        // Stop the capture session if it's running
        if captureSession?.isRunning ?? false {
            captureSession?.stopRunning()
        }
        
        // Wait for any tasks on the processingQueue to finish
        processingQueue.sync { }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        // Update video preview layer frame when view layout changes
        videoPreviewLayer?.frame = imageView.bounds
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        
        // Remove the sample buffer delegate to be extra safe
        videoDataOutput?.setSampleBufferDelegate(nil, queue: nil)
        videoDataOutput = nil
        
        // Stop the capture session if it's running and release it
        if captureSession?.isRunning ?? false {
            captureSession?.stopRunning()
        }
        captureSession = nil
    }

    private func setupUI() {
        // Calculate safe area
        let safeArea = view.safeAreaLayoutGuide
        
        // Create and add the segmented control
        segmentControl = UISegmentedControl(items: ["Wave/Wind", "Sea Detection", "Live Camera"])
        segmentControl.translatesAutoresizingMaskIntoConstraints = false
        segmentControl.selectedSegmentIndex = 2
        segmentControl.addTarget(self, action: #selector(segmentChanged(_:)), for: .valueChanged)
        view.addSubview(segmentControl)
        
        // Image view - takes up the entire screen
        imageView = UIImageView()
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = UIColor.black.withAlphaComponent(0.1)
        imageView.clipsToBounds = true
        view.addSubview(imageView)
        
        // Toggle button for switching between original and visualization
        toggleButton = UIButton(type: .system)
        toggleButton.translatesAutoresizingMaskIntoConstraints = false
        var toggleConfig = UIButton.Configuration.filled()
        toggleConfig.title = "Camera"
        toggleConfig.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        toggleConfig.baseForegroundColor = .white
        toggleConfig.contentInsets = NSDirectionalEdgeInsets(top: 6, leading: 12, bottom: 6, trailing: 12)
        toggleConfig.cornerStyle = .medium
        toggleConfig.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer { incoming in
            var outgoing = incoming
            outgoing.font = UIFont.systemFont(ofSize: 13, weight: .medium)
            return outgoing
        }
        toggleButton.configuration = toggleConfig
        toggleButton.layer.shadowColor = UIColor.black.cgColor
        toggleButton.layer.shadowOffset = CGSize(width: 0, height: 2)
        toggleButton.layer.shadowRadius = 3
        toggleButton.layer.shadowOpacity = 0.3
        toggleButton.addTarget(self, action: #selector(toggleVisualization), for: .touchUpInside)
        view.addSubview(toggleButton)
        
        // Semi-transparent overlay for prediction text
        predictionOverlayView = UIView()
        predictionOverlayView.translatesAutoresizingMaskIntoConstraints = false
        predictionOverlayView.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        predictionOverlayView.layer.cornerRadius = 8
        predictionOverlayView.layer.shadowColor = UIColor.black.cgColor
        predictionOverlayView.layer.shadowOffset = CGSize(width: 0, height: 2)
        predictionOverlayView.layer.shadowRadius = 4
        predictionOverlayView.layer.shadowOpacity = 0.4
        view.addSubview(predictionOverlayView)
        
        predictionLabel = UILabel()
        predictionLabel.translatesAutoresizingMaskIntoConstraints = false
        predictionLabel.numberOfLines = 0
        predictionLabel.textAlignment = .left
        predictionLabel.textColor = .white
        predictionLabel.font = .systemFont(ofSize: 14, weight: .medium)
        predictionOverlayView.addSubview(predictionLabel)
        
        // Activity indicator
        activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.hidesWhenStopped = true
        activityIndicator.color = .white
        imageView.addSubview(activityIndicator)
        
        // Create and add the select image button
        selectButton = UIButton(type: .system)
        selectButton.translatesAutoresizingMaskIntoConstraints = false
        var selectConfig = UIButton.Configuration.filled()
        selectConfig.title = "Select Image"
        selectConfig.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        selectConfig.baseForegroundColor = .white
        selectConfig.contentInsets = NSDirectionalEdgeInsets(top: 10, leading: 20, bottom: 10, trailing: 20)
        selectConfig.cornerStyle = .large
        selectButton.configuration = selectConfig
        selectButton.addTarget(self, action: #selector(openPhotoLibrary), for: .touchUpInside)
        view.addSubview(selectButton)
        
        // Set up constraints
        NSLayoutConstraint.activate([
            // Segmented control at the top
            segmentControl.topAnchor.constraint(equalTo: safeArea.topAnchor, constant: 12),
            segmentControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            segmentControl.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.8),
            
            // Image view constraints - fill the entire view
            imageView.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Toggle button - positioned below segment control in the top right corner
            toggleButton.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            toggleButton.trailingAnchor.constraint(equalTo: safeArea.trailingAnchor, constant: -12),
            
            // Overlay view constraints - below segment control in the top left corner
            predictionOverlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor, constant: 12),
            predictionOverlayView.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            predictionOverlayView.widthAnchor.constraint(lessThanOrEqualTo: imageView.widthAnchor, multiplier: 0.6),
            predictionOverlayView.trailingAnchor.constraint(lessThanOrEqualTo: toggleButton.leadingAnchor, constant: -12),
            
            // Prediction label inside overlay with padding
            predictionLabel.topAnchor.constraint(equalTo: predictionOverlayView.topAnchor, constant: 6),
            predictionLabel.bottomAnchor.constraint(equalTo: predictionOverlayView.bottomAnchor, constant: -6),
            predictionLabel.leadingAnchor.constraint(equalTo: predictionOverlayView.leadingAnchor, constant: 8),
            predictionLabel.trailingAnchor.constraint(equalTo: predictionOverlayView.trailingAnchor, constant: -8),
            
            // Activity indicator center in image view
            activityIndicator.centerXAnchor.constraint(equalTo: imageView.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: imageView.centerYAnchor),
            
            // Select button at the bottom center
            selectButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            selectButton.bottomAnchor.constraint(equalTo: safeArea.bottomAnchor, constant: -20)
        ])
        
        // Initially hide the select button in camera mode
        selectButton.isHidden = true
    }

    @objc private func toggleVisualization() {
        showingVisualization = !showingVisualization
        
        // Update toggle button appearance
        updateToggleButtonState()
        
        // Remove any overlay layers when turning off visualization
        if !showingVisualization {
            clearCameraOverlays()
        }
    }

    private func updateToggleButtonState() {
        var config = UIButton.Configuration.filled()
        
        if showingVisualization {
            config.title = "Camera"
            config.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            config.title = "Camera"
            config.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        }
        
        // Apply common styling
        config.baseForegroundColor = .white
        config.contentInsets = NSDirectionalEdgeInsets(top: 6, leading: 12, bottom: 6, trailing: 12)
        config.cornerStyle = .medium
        config.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer { incoming in
            var outgoing = incoming
            outgoing.font = UIFont.systemFont(ofSize: 13, weight: .medium)
            return outgoing
        }
        
        toggleButton.configuration = config
    }

    // MARK: - Camera Mode Functions
    private func startCameraSession() {
        // Reset UI
        predictionLabel.text = "Initializing camera..."
        predictionOverlayView.isHidden = false
        
        // Initialize capture session
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .medium
        
        // Check camera authorization
        checkCameraAuthorization { [weak self] authorized in
            guard let self = self, authorized else {
                DispatchQueue.main.async {
                    self?.predictionLabel.text = "Camera access denied"
                    self?.predictionOverlayView.isHidden = false
                }
                return
            }
            
            self.setupCameraInput()
        }
    }

    private func checkCameraAuthorization(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                completion(granted)
            }
        default:
            completion(false)
        }
    }

    private func setupCameraInput() {
        guard let captureSession = captureSession else { return }
        
        // Get back camera
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            DispatchQueue.main.async {
                self.predictionLabel.text = "Back camera not available"
                self.predictionOverlayView.isHidden = false
            }
            return
        }
        
        do {
            // Create input from camera
            let input = try AVCaptureDeviceInput(device: backCamera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            } else {
                handleCameraSetupError("Could not add camera input")
                return
            }
            
            // Create and add video preview layer
            DispatchQueue.main.async {
                self.setupVideoPreviewLayer()
            }
            
            // Create and add video output
            setupVideoOutput()
            
            // Start the session
            DispatchQueue.global(qos: .userInitiated).async {
                captureSession.startRunning()
            }
            
        } catch {
            handleCameraSetupError("Error setting up camera: \(error.localizedDescription)")
        }
    }

    private func setupVideoPreviewLayer() {
        guard let captureSession = captureSession else { return }
        
        // Create preview layer
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        videoPreviewLayer?.videoGravity = .resizeAspectFill
        videoPreviewLayer?.frame = imageView.bounds
        
        // Add preview layer to image view's layer
        if let previewLayer = videoPreviewLayer {
            // Remove only existing video preview layers, not all sublayers
            imageView.layer.sublayers?.removeAll(where: { $0 is AVCaptureVideoPreviewLayer })
            imageView.layer.addSublayer(previewLayer)
        }
    }

    private func setupVideoOutput() {
        guard let captureSession = captureSession else { return }
        
        videoDataOutput = AVCaptureVideoDataOutput()
        guard let videoDataOutput = videoDataOutput else { return }
        
        // Use DispatchQueue instead of OperationQueue
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoProcessingQueue"))
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        
        // Configure for highest quality
        if let connection = videoDataOutput.connection(with: .video) {
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .auto
            }
            
            // Set the video orientation to portrait
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(0) {
                    connection.videoRotationAngle = 0
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
        }
        
        // Set video settings for better quality
        videoDataOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: 1920,
            kCVPixelBufferHeightKey as String: 1080
        ]
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
    }

    private func handleCameraSetupError(_ message: String) {
        DispatchQueue.main.async {
            self.predictionLabel.text = message
            self.predictionOverlayView.isHidden = false
        }
    }

    deinit {
        print("[DEBUG] deinit: ViewController is being deallocated: \(self)")
        // Remove the sample buffer delegate to prevent any pending callbacks
        videoDataOutput?.setSampleBufferDelegate(nil, queue: nil)
        
        // Ensure resources are properly cleaned up
        frameProcessingQueue.cancelAllOperations()
        captureSession?.stopRunning()
        processingQueue.sync { } // Wait for any pending processing to complete
    }

    // MARK: - Video Buffer Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Skip if we're currently processing a frame
        if frameProcessingQueue.operationCount > 0 {
            return
        }
        
        // Throttle based on time - process at most 1 frame per second
        let currentTime = CACurrentMediaTime()
        if currentTime - self.lastFrameCaptureTime < 1.0 {
            return
        }
        self.lastFrameCaptureTime = currentTime
        
        // Extract pixel buffer and convert to UIImage
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Create CGImage with proper orientation
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        
        // Create UIImage directly from CGImage to maintain portrait orientation
        let currentImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
        
        // Create an operation for processing this frame
        let operation = BlockOperation { [weak self] in
            self?.processFrameInBackground(currentImage)
        }
        operation.completionBlock = {
            if operation.isCancelled {
                print("[DEBUG] Frame processing operation was cancelled")
            }
        }
        frameProcessingQueue.addOperation(operation)
    }
    
    private func clearCameraOverlays() {
        // Just hide the layer instead of removing it
        overlayLayer?.isHidden = true
    }

    private func loadModels() {
        guard let detectModelURL = Bundle.main.url(forResource: "sea_segmentation_base", withExtension: "mlmodelc") else {
            print("Failed to find sea_segmentation_base.mlmodelc")
            return
        }
        guard let predictModelURL = Bundle.main.url(forResource: "regression_model", withExtension: "mlmodelc") else {
            print("Failed to find regression_model.mlmodelc")
            return
        }
        guard let configURL = Bundle.main.url(forResource: "sea_config", withExtension: "json") else {
            print("Failed to find sea_config.json")
            return
        }
        do {
            let configData = try Data(contentsOf: configURL)
            let seaConfig = try JSONDecoder().decode(SeaConfig.self, from: configData)
            seaPredictor = try SeaPredictor(modelPath: predictModelURL)
            seaDetector = try SeaDetector(modelPath: detectModelURL, config: seaConfig)
        } catch {
            print("Failed to initialize models: \(error)")
        }
    }

    private func processFrameInBackground(_ image: UIImage) {
        autoreleasepool {
            do {
                // Both models use 224x224 input size
                let modelInputSize = CGSize(width: 224, height: 224)
                
                // Rotate the image 90 degrees clockwise before processing (landscape orientation for model)
                guard let rotatedImage = rotateImage(image, byDegrees: 90) else {
                    print("[ERROR] Failed to rotate image")
                    return
                }
                
                // Preprocess the rotated image
                guard let preprocessedImage = rotatedImage.resize(to: modelInputSize),
                      let multiArray = try? preprocessedImage.preprocessForML(targetSize: modelInputSize) else {
                    print("[ERROR] Failed to preprocess image")
                    return
                }
                
                // Ensure models are loaded
                guard let detector = seaDetector, let predictor = seaPredictor else {
                    print("[ERROR] Models not loaded")
                    return
                }
                
                // Run detection
                let seaMask = try detector.detectSea(multiArray: multiArray)
                let seaPercentage = SeaDetector.calculateSeaFraction(mask: seaMask)
                let containsSea = seaPercentage > 0.2
                
                // Generate visualization if applicable
                var visualizedImage: UIImage? = nil
                if containsSea && SeaDetector.enableVisualization {
                    // Get the visualization in landscape orientation (what the model expects)
                    guard let maskVisualization = try detector.generateSeaMaskVisualization(mask: seaMask, originalImage: rotatedImage) else {
                        print("[ERROR] Failed to generate mask visualization")
                        return
                    }
                    
                    // Keep the visualization in its original orientation
                    visualizedImage = maskVisualization
                }
                
                // Run prediction if sea is detected
                var predictions: [Float]? = nil
                if containsSea {
                    predictions = try predictor.predict(multiArray: multiArray)
                }
                
                let results = CameraFrameResults(seaPercentage: seaPercentage,
                                                 containsSea: containsSea,
                                                 predictions: predictions,
                                                 visualizedImage: visualizedImage,
                                                 originalImage: image)
                
                DispatchQueue.main.async { [weak self] in
                    self?.updateUIWithCameraResults(results)
                }
            } catch {
                DispatchQueue.main.async {
                    print("[Error] Frame processing error: \(error)")
                }
            }
        }
    }

    // Helper function to rotate UIImage
    private func rotateImage(_ image: UIImage, byDegrees degrees: CGFloat) -> UIImage? {
        let radians = degrees * .pi / 180
        let rotatedSize = CGRect(origin: .zero, size: image.size)
            .applying(CGAffineTransform(rotationAngle: radians))
            .integral.size
        
        UIGraphicsBeginImageContextWithOptions(rotatedSize, false, image.scale)
        if let context = UIGraphicsGetCurrentContext() {
            context.translateBy(x: rotatedSize.width / 2, y: rotatedSize.height / 2)
            context.rotate(by: radians)
            image.draw(in: CGRect(x: -image.size.width / 2, y: -image.size.height / 2, width: image.size.width, height: image.size.height))
            
            let rotatedImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            return rotatedImage
        }
        return nil
    }

    private struct CameraFrameResults {
        let seaPercentage: Float
        let containsSea: Bool
        let predictions: [Float]?
        let visualizedImage: UIImage?
        let originalImage: UIImage
    }

    private func updateUIWithCameraResults(_ results: CameraFrameResults) {
        guard isViewLoaded, view.window != nil else {
            print("[DEBUG] Skipping UI update: View not loaded")
            return
        }
        
        // Print the results of the model
        print("[RESULTS] Sea Percentage: \(results.seaPercentage)")
        print("[RESULTS] Contains Sea: \(results.containsSea)")
        if let predictions = results.predictions {
            print("[RESULTS] Predictions: \(predictions)")
        } else {
            print("[RESULTS] No predictions available")
        }
        
        let messageText: String
        if results.containsSea {
            if let predictions = results.predictions, predictions.count >= 2 {
                messageText = String(format: "Sea: %d%%\nWind: %.1f m/s\nWave: %.1f m",
                                      Int(results.seaPercentage * 100), predictions[0], predictions[1])
            } else {
                messageText = String(format: "Sea: %d%%", Int(results.seaPercentage * 100))
            }
        } else {
            messageText = String(format: "Sea: %d%%\nNo sea detected", Int(results.seaPercentage * 100))
        }
        
        predictionLabel.text = messageText
        predictionOverlayView.isHidden = false
        
        if results.containsSea && SeaDetector.enableVisualization && showingVisualization {
            updateCameraPreviewOverlay(results.visualizedImage)
        } else {
            clearCameraOverlays()
        }
    }

    private func updateCameraPreviewOverlay(_ overlayImage: UIImage?) {
        guard let overlayImage = overlayImage, let currentImageView = imageView else { return }
        
        // If we don't have an overlay layer yet, create one
        if overlayLayer == nil {
            let newLayer = CALayer()
            newLayer.opacity = 0.7
            
            if let videoLayer = videoPreviewLayer {
                currentImageView.layer.insertSublayer(newLayer, above: videoLayer)
            } else {
                currentImageView.layer.addSublayer(newLayer)
            }
            
            overlayLayer = newLayer
        }
        
        // Now we know overlayLayer exists, so update its properties
        if let layer = overlayLayer, let videoLayer = videoPreviewLayer {
            // Important: Use the exact frame of the video preview layer
            layer.frame = videoLayer.frame
            
            // Just use the overlay image directly without rotation
            layer.contents = overlayImage.cgImage
            
            // Reset transform - no rotation needed
            layer.transform = CATransform3DIdentity
            
            // Make sure the content fills the layer the same way the video does
            layer.contentsGravity = videoLayer.videoGravity == .resizeAspectFill 
                ? .resizeAspectFill : .resizeAspect
            
            layer.isHidden = false
            
            print("[DEBUG] Overlay updated without rotation - Frame: \(layer.frame)")
        }
    }

    // Add method to handle segmented control changes
    @objc private func segmentChanged(_ sender: UISegmentedControl) {
        let selectedIndex = sender.selectedSegmentIndex
        lastProcessedMode = selectedIndex
        if selectedIndex == 2 { // Live Camera
            print("[DEBUG] Switched to Live Camera mode")
            startCameraSession()
            selectButton.isHidden = true
            toggleButton.isHidden = false
        } else {
            print("[DEBUG] Switched to static image mode")
            stopCameraSession()
            selectButton.isHidden = false
            toggleButton.isHidden = true
        }
    }

    // Add method to open photo library
    @objc private func openPhotoLibrary() {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        picker.allowsEditing = true
        present(picker, animated: true)
    }

    // UIImagePickerControllerDelegate methods
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let selectedImage = info[.editedImage] as? UIImage ?? info[.originalImage] as? UIImage {
            originalImage = selectedImage
            imageView.image = selectedImage
            // Process the selected image if needed
            processFrameInBackground(selectedImage)
        }
        dismiss(animated: true)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true)
    }

    // Add stopCameraSession() method to stop camera session for static image modes
    private func stopCameraSession() {
        captureSession?.stopRunning()
        // Optionally, remove the video preview layer
        videoPreviewLayer?.removeFromSuperlayer()
    }
}

// Helper extension to convert CIImage to UIImage
extension CIImage {
    var uiImage: UIImage? {
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(self, from: self.extent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

