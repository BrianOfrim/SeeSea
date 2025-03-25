import UIKit
import AVFoundation

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var imageView: UIImageView!
    private var seaPredictor: SeaPredictor?
    private var seaDetector: SeaDetector?
    private var predictionLabel: UILabel!
    private var predictionOverlayView: UIView!
    private var activityIndicator: UIActivityIndicatorView!
    private var segmentControl: UISegmentedControl!
    private var toggleButton: UIButton!
    
    // Camera capture properties
    private var captureSession: AVCaptureSession?
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var processingQueue = DispatchQueue(label: "videoProcessingQueue")
    private var isProcessingFrame = false
    
    // Frame processing queue - serializes processing to avoid race conditions
    private lazy var frameProcessingQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "FrameProcessingQueue"
        queue.maxConcurrentOperationCount = 1 // Serial queue
        return queue
    }()
    
    // Dedicated main queue work for UI updates
    private let mainQueue = DispatchQueue.main
    
    // Store both original and visualization images
    private var originalImage: UIImage?
    private var visualizationImage: UIImage?
    private var showingVisualization = true
    
    // Store prediction results to avoid recomputing
    private var seaPercentage: Float = 0
    private var containsSea: Bool = false
    private var waveWindPredictions: [Float]?
    
    // Track which mode was last processed
    private var lastProcessedMode: Int = -1

    // Store last capture time as a property instead of static variable
    private var lastFrameCaptureTime: CFTimeInterval = 0

    // Store a reference to the selectButton for showing/hiding
    private var selectButton: UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set up background image
        setupBackgroundImage()
        
        // Set up UI components
        setupUI()
        
        guard let detectModelURL = Bundle.main.url(forResource: "sea_segmentation_base", withExtension: "mlmodelc") else{
            print("Failed to find sea_segmentation_base.mlpackage")
            return
        }
        
        guard let predictModelURL = Bundle.main.url(forResource: "regression_model", withExtension: "mlmodelc") else{
            print("Failed to find regression_model.mlpackage")
            return
        }

        // Load sea configuration
        guard let configURL = Bundle.main.url(forResource: "sea_config", withExtension: "json") else{
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

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Resume camera session if we were in camera mode
        if lastProcessedMode == 2 && !(captureSession?.isRunning ?? false) {
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession?.startRunning()
            }
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Cancel all pending frame processing
        frameProcessingQueue.cancelAllOperations()
        
        // Pause the session but don't tear it down
        if captureSession?.isRunning ?? false {
            captureSession?.stopRunning()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        // Update video preview layer frame when view layout changes
        videoPreviewLayer?.frame = imageView.bounds
    }

    private func setupBackgroundImage() {
        // Create a background image view
        let backgroundImageView = UIImageView(frame: view.bounds)
        backgroundImageView.image = UIImage(named: "Background")
        backgroundImageView.contentMode = .scaleAspectFill
        backgroundImageView.translatesAutoresizingMaskIntoConstraints = false
        
        // Add it as the bottom-most view
        view.insertSubview(backgroundImageView, at: 0)
        
        // Make it fill the entire view
        NSLayoutConstraint.activate([
            backgroundImageView.topAnchor.constraint(equalTo: view.topAnchor),
            backgroundImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            backgroundImageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            backgroundImageView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    func setupUI() {
        // Calculate safe area
        let safeArea = view.safeAreaLayoutGuide
        
        // Image view - takes up the entire screen
        imageView = UIImageView()
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = UIColor.black.withAlphaComponent(0.1)
        imageView.clipsToBounds = true
        view.addSubview(imageView)
        
        // Segment control for switching between models
        segmentControl = UISegmentedControl(items: ["Wave/Wind", "Sea Detection", "Live Camera"])
        segmentControl.translatesAutoresizingMaskIntoConstraints = false
        segmentControl.selectedSegmentIndex = 0
        segmentControl.backgroundColor = UIColor.white.withAlphaComponent(0.8)
        segmentControl.layer.shadowColor = UIColor.black.cgColor
        segmentControl.layer.shadowOffset = CGSize(width: 0, height: 2)
        segmentControl.layer.shadowRadius = 3
        segmentControl.layer.shadowOpacity = 0.3
        segmentControl.addTarget(self, action: #selector(segmentChanged), for: .valueChanged)
        view.addSubview(segmentControl)
        
        // Toggle button for switching between original and visualization
        toggleButton = UIButton(type: .system)
        toggleButton.translatesAutoresizingMaskIntoConstraints = false
        
        var toggleConfig = UIButton.Configuration.filled()
        toggleConfig.title = "Overlay"
        toggleConfig.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        toggleConfig.baseForegroundColor = .white
        toggleConfig.contentInsets = NSDirectionalEdgeInsets(top: 6, leading: 12, bottom: 6, trailing: 12)
        toggleConfig.cornerStyle = .medium
        // Use a smaller font size
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
        toggleButton.isHidden = true // Initially hidden
        view.addSubview(toggleButton)
        
        // Semi-transparent overlay for prediction text
        predictionLabel = UILabel()
        predictionLabel.translatesAutoresizingMaskIntoConstraints = false
        predictionLabel.numberOfLines = 0
        predictionLabel.textAlignment = .left
        predictionLabel.textColor = .white
        predictionLabel.font = .systemFont(ofSize: 14, weight: .medium)
        
        // Create semi-transparent background for the label
        predictionOverlayView = UIView()
        predictionOverlayView.translatesAutoresizingMaskIntoConstraints = false
        predictionOverlayView.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        predictionOverlayView.layer.cornerRadius = 8
        predictionOverlayView.layer.shadowColor = UIColor.black.cgColor
        predictionOverlayView.layer.shadowOffset = CGSize(width: 0, height: 2)
        predictionOverlayView.layer.shadowRadius = 4
        predictionOverlayView.layer.shadowOpacity = 0.4
        
        imageView.addSubview(predictionOverlayView)
        predictionOverlayView.addSubview(predictionLabel)
        
        // Activity indicator
        activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.hidesWhenStopped = true
        activityIndicator.color = .white
        imageView.addSubview(activityIndicator)
        
        // Select image button - now floating at the bottom of the screen
        let selectButton = UIButton(type: .system)
        selectButton.translatesAutoresizingMaskIntoConstraints = false
        
        // Use UIButtonConfiguration instead of contentEdgeInsets
        var config = UIButton.Configuration.filled()
        config.title = "Select Image"
        config.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        config.baseForegroundColor = .white
        config.contentInsets = NSDirectionalEdgeInsets(top: 10, leading: 20, bottom: 10, trailing: 20)
        config.cornerStyle = .large
        selectButton.configuration = config
        selectButton.layer.shadowColor = UIColor.black.cgColor
        selectButton.layer.shadowOffset = CGSize(width: 0, height: 2)
        selectButton.layer.shadowRadius = 3
        selectButton.layer.shadowOpacity = 0.3
        
        selectButton.addTarget(self, action: #selector(openPhotoLibrary), for: .touchUpInside)
        view.addSubview(selectButton)
        
        // Store a reference to the selectButton for showing/hiding
        self.selectButton = selectButton
        
        NSLayoutConstraint.activate([
            // Image view constraints - fill the entire view, extending under safe areas
            imageView.topAnchor.constraint(equalTo: view.topAnchor),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Segment control - at the top safe area
            segmentControl.topAnchor.constraint(equalTo: safeArea.topAnchor, constant: 12),
            segmentControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            segmentControl.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.8),
            
            // Toggle button - moved to the right side of the screen
            toggleButton.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            toggleButton.trailingAnchor.constraint(equalTo: safeArea.trailingAnchor, constant: -12),
            
            // Select button constraints - floating at the bottom
            selectButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            selectButton.bottomAnchor.constraint(equalTo: safeArea.bottomAnchor, constant: -20),
            
            // Overlay view constraints - position in top left corner with padding
            // and ensure it doesn't overlap with the toggle button
            predictionOverlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor, constant: 12),
            predictionOverlayView.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            predictionOverlayView.trailingAnchor.constraint(lessThanOrEqualTo: toggleButton.leadingAnchor, constant: -12),
            predictionOverlayView.widthAnchor.constraint(lessThanOrEqualTo: imageView.widthAnchor, multiplier: 0.6),
            
            // Prediction label constraints - inside the overlay with padding
            predictionLabel.topAnchor.constraint(equalTo: predictionOverlayView.topAnchor, constant: 6),
            predictionLabel.bottomAnchor.constraint(equalTo: predictionOverlayView.bottomAnchor, constant: -6),
            predictionLabel.leadingAnchor.constraint(equalTo: predictionOverlayView.leadingAnchor, constant: 8),
            predictionLabel.trailingAnchor.constraint(equalTo: predictionOverlayView.trailingAnchor, constant: -8),
            
            // Activity indicator constraints
            activityIndicator.centerXAnchor.constraint(equalTo: imageView.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: imageView.centerYAnchor)
        ])
        
        predictionOverlayView.isHidden = false  // Initially hide the overlay
    }

    @objc func openPhotoLibrary() {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        picker.allowsEditing = true
        present(picker, animated: true)
        
        // Hide the overlay when selecting a new image
        predictionOverlayView.isHidden = false
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let selectedImage = info[.editedImage] as? UIImage ?? info[.originalImage] as? UIImage {
            // Store the original image
            self.originalImage = selectedImage
            
            // Display the original image
            imageView.image = selectedImage
            
            predictionLabel.text = "Processing..."
            activityIndicator.startAnimating()
            
            // Hide toggle button when new image is selected
            toggleButton.isHidden = true
            
            // Reset cached results when a new image is selected
            waveWindPredictions = nil
            visualizationImage = nil
            
            // Process both types of predictions in the background with truly shared preprocessing
            let processingWorkItem = DispatchWorkItem {
                // Start timing the entire process
                let totalStartTime = CFAbsoluteTimeGetCurrent()
                
                do {
                    // Both models use 224x224 input size (hardcoded)
                    let modelInputSize = CGSize(width: 224, height: 224)
                    
                    print("Preprocessing image to 224x224")
                    
                    // Preprocess the image once
                    let preprocessStartTime = CFAbsoluteTimeGetCurrent()
                    let preprocessedImage = selectedImage.resize(to: modelInputSize)
                    guard let preprocessedImage = preprocessedImage else {
                        throw NSError(domain: "ImageProcessing", code: -1, 
                                     userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
                    }
                    let preprocessEndTime = CFAbsoluteTimeGetCurrent()
                    let preprocessTime = preprocessEndTime - preprocessStartTime
                    print("[Shared] Image preprocessing completed in \(String(format: "%.3f", preprocessTime)) seconds")
                    
                    // Create a shared MLMultiArray that both models can use
                    let multiArrayStartTime = CFAbsoluteTimeGetCurrent()
                    let sharedMultiArray = try preprocessedImage.preprocessForML(targetSize: modelInputSize)
                    let multiArrayEndTime = CFAbsoluteTimeGetCurrent()
                    let multiArrayTime = multiArrayEndTime - multiArrayStartTime
                    print("[Shared] MLMultiArray conversion completed in \(String(format: "%.3f", multiArrayTime)) seconds")
                    
                    // Run both models in parallel using the shared preprocessed data
                    let group = DispatchGroup()
                    
                    // Wave/Wind prediction task
                    group.enter()
                    let waveWindWorkItem = DispatchWorkItem {
                        do {
                            let waveWindStartTime = CFAbsoluteTimeGetCurrent()
                            let predictions = try self.seaPredictor?.predict(multiArray: sharedMultiArray)
                            self.waveWindPredictions = predictions
                            let waveWindEndTime = CFAbsoluteTimeGetCurrent()
                            let waveWindTime = waveWindEndTime - waveWindStartTime
                            print("Wave/Wind prediction completed in \(String(format: "%.3f", waveWindTime)) seconds")
                            group.leave()
                        } catch {
                            print("Wave/Wind prediction failed: \(error)")
                            group.leave()
                        }
                    }
                    DispatchQueue.global(qos: .userInitiated).async(execute: waveWindWorkItem)
                    
                    // Sea detection task
                    group.enter()
                    let seaDetectionWorkItem = DispatchWorkItem {
                        do {
                            let seaDetectionStartTime = CFAbsoluteTimeGetCurrent()
                            
                            // Detection step - Fix similar issue with detectSea
                            guard let detector = self.seaDetector else {
                                throw NSError(domain: "SeaDetection", code: -1, 
                                              userInfo: [NSLocalizedDescriptionKey: "Sea detector not initialized"])
                            }
                            let seaMask = try detector.detectSea(multiArray: sharedMultiArray)

                            self.seaPercentage = SeaDetector.calculateSeaFraction(mask: seaMask)
                            self.containsSea = self.seaPercentage > 0.2
                            
                            let seaDetectionEndTime = CFAbsoluteTimeGetCurrent()
                            let seaDetectionTime = seaDetectionEndTime - seaDetectionStartTime
                            print("Sea detection completed in \(String(format: "%.3f", seaDetectionTime)) seconds")
                            
                            if(SeaDetector.enableVisualization){
                                // Visualization step
                                if let visualizedImage = try detector.generateSeaMaskVisualization(mask: seaMask, originalImage: selectedImage) {
                                    self.visualizationImage = visualizedImage
                                }
                            }

                            group.leave()
                        } catch {
                            print("Sea detection failed: \(error)")
                            group.leave()
                        }
                    }
                    DispatchQueue.global(qos: .userInitiated).async(execute: seaDetectionWorkItem)
                    
                    // Wait for both tasks to complete
                    group.wait()
                    
                    let totalEndTime = CFAbsoluteTimeGetCurrent()
                    let totalTime = totalEndTime - totalStartTime
                    print("Total processing completed in \(String(format: "%.3f", totalTime)) seconds")
                    
                    // Update UI based on the current segment
                    DispatchQueue.main.async {
                        self.activityIndicator.stopAnimating()
                        self.lastProcessedMode = self.segmentControl.selectedSegmentIndex
                        
                        // Show results based on current segment
                        if self.segmentControl.selectedSegmentIndex == 0 {
                            self.showWaveWindResults()
                        } else {
                            self.showSeaDetectionResults()
                        }
                    }
                } catch {
                    print("Image processing failed: \(error)")
                    DispatchQueue.main.async {
                        self.activityIndicator.stopAnimating()
                        self.predictionLabel.text = "Processing failed: \(error.localizedDescription)"
                        self.predictionOverlayView.isHidden = false
                    }
                }
            }
            DispatchQueue.global(qos: .userInitiated).async(execute: processingWorkItem)
        }
        dismiss(animated: true)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true)
    }

    @objc func segmentChanged(_ sender: UISegmentedControl) {
        // Only proceed if we have an original image or if camera mode is selected
        if originalImage != nil || sender.selectedSegmentIndex == 2 {
            // Capture the selected segment index on the main thread
            let selectedIndex = sender.selectedSegmentIndex
            
            // Stop camera if switching away from camera mode
            if lastProcessedMode == 2 && selectedIndex != 2 {
                stopCameraSession()
            }
            
            // Hide or show toggle button based on mode and visualization setting
            if selectedIndex == 0 {
                // Hide toggle in Wave/Wind mode
                toggleButton.isHidden = true
            } else if selectedIndex == 1 {
                // Show toggle in Sea Detection mode only if visualization is enabled
                toggleButton.isHidden = !SeaDetector.enableVisualization
            } else if selectedIndex == 2 {
                // Show toggle in Camera mode if visualization is enabled
                toggleButton.isHidden = !SeaDetector.enableVisualization
                
                // Update toggle button to show current visualization state
                updateToggleButtonState()
            }
            
            // Hide select image button in camera mode
            selectButton.isHidden = selectedIndex == 2
            
            // Update the last processed mode
            lastProcessedMode = selectedIndex
            
            // Just update the UI with existing results
            if selectedIndex == 0 {
                // Show wave/wind results
                showWaveWindResults()
            } else if selectedIndex == 1 {
                // Show sea detection results
                showSeaDetectionResults()
            } else if selectedIndex == 2 {
                // Start camera
                startCameraSession()
            }
        } else {
            // No image selected yet and not camera mode
            DispatchQueue.main.async {
                if sender.selectedSegmentIndex == 2 {
                    // Start camera mode anyway
                    self.lastProcessedMode = 2
                    self.selectButton.isHidden = true
                    
                    // Show toggle in Camera mode if visualization is enabled
                    self.toggleButton.isHidden = !SeaDetector.enableVisualization
                    self.updateToggleButtonState()
                    
                    self.startCameraSession()
                } else {
                    self.predictionLabel.text = "Please select an image first"
                    self.predictionOverlayView.isHidden = false
                }
            }
        }
    }

    @objc func toggleVisualization() {
        showingVisualization = !showingVisualization
        
        // Update toggle button appearance
        updateToggleButtonState()
        
        // If not in camera mode, handle toggle normally for static images
        if lastProcessedMode != 2 {
            if showingVisualization {
                // Show visualization
                if let visualizedImage = visualizationImage {
                    imageView.image = visualizedImage
                } else {
                    print("Warning: Visualization image is nil")
                }
            } else {
                // Show original
                if let originalImage = originalImage {
                    imageView.image = originalImage
                } else {
                    print("Warning: Original image is nil")
                }
            }
        } else {
            // In camera mode, toggle just affects overlay visualization
            // The next frame processing will update the view based on showingVisualization
            if !showingVisualization {
                // Remove any overlay layers when turning off visualization
                imageView.layer.sublayers?.forEach { layer in
                    if layer != videoPreviewLayer {
                        layer.removeFromSuperlayer()
                    }
                }
            }
        }
    }

    // Helper method to update toggle button's appearance
    private func updateToggleButtonState() {
        var config = UIButton.Configuration.filled()
        
        if showingVisualization {
            config.title = "Overlay"
            config.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            config.title = "Original"
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
        
        // Ensure toggle button is visible
        if (lastProcessedMode == 1 || lastProcessedMode == 2) && SeaDetector.enableVisualization {
            view.bringSubviewToFront(toggleButton)
        }
    }

    // Helper method to show wave/wind results without recomputing
    private func showWaveWindResults() {
        if let predictions = waveWindPredictions, predictions.count >= 2 {
            let wind_speed_mps = predictions[0]
            let wave_height_m = predictions[1]
            
            self.predictionLabel.text = String(format: "Wind Speed: %.1f m/s\nWave Height: %.1f m",
                                        wind_speed_mps, wave_height_m)
            self.predictionOverlayView.isHidden = false
            
            // Show original image if we have one
            if let originalImage = self.originalImage {
                self.imageView.image = originalImage
            }
        } else {
            self.predictionLabel.text = "Wave/Wind prediction results not available yet"
            self.predictionOverlayView.isHidden = false
        }
    }
    
    // Helper method to show sea detection results without recomputing
    private func showSeaDetectionResults() {
        // Check if we have sea detection results
        if visualizationImage == nil {
            self.predictionLabel.text = "Sea detection results not available yet"
            self.predictionOverlayView.isHidden = false
            self.toggleButton.isHidden = true
            return
        }
        
        // Update the text label with existing results
        if containsSea {
            self.predictionLabel.text = "Sea detected: \(Int(seaPercentage * 100))%"
        } else {
            self.predictionLabel.text = "No sea detected"
        }
        self.predictionOverlayView.isHidden = false
        
        // Show the toggle button only if visualization is enabled
        self.toggleButton.isHidden = !SeaDetector.enableVisualization
        if SeaDetector.enableVisualization {
            view.bringSubviewToFront(toggleButton)
        }
        
        // Update toggle button configuration
        var config = UIButton.Configuration.filled()
        
        // Show the appropriate image based on toggle state and visualization setting
        if SeaDetector.enableVisualization {
            if showingVisualization {
                if let visualizedImage = self.visualizationImage {
                    self.imageView.image = visualizedImage
                }
                config.title = "Overlay"
                config.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
            } else {
                if let originalImage = self.originalImage {
                    self.imageView.image = originalImage
                }
                config.title = "Original"
                config.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
            }
        } else {
            // If visualization is disabled, always show original image
            if let originalImage = self.originalImage {
                self.imageView.image = originalImage
            }
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
        
        // Hide select image button in camera mode
        selectButton.isHidden = true
        
        // Check if session already exists
        if captureSession != nil {
            if !(captureSession?.isRunning ?? false) {
                DispatchQueue.global(qos: .userInitiated).async {
                    self.captureSession?.startRunning()
                }
            }
            return
        }
        
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
        
        // Create video data output
        videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput?.setSampleBufferDelegate(self, queue: processingQueue)
        videoDataOutput?.alwaysDiscardsLateVideoFrames = true
        
        // Add output to session
        if let videoOutput = videoDataOutput, captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            handleCameraSetupError("Could not add video output")
        }
    }

    private func handleCameraSetupError(_ message: String) {
        DispatchQueue.main.async {
            self.predictionLabel.text = message
            self.predictionOverlayView.isHidden = false
        }
    }

    private func stopCameraSession() {
        // Stop the session
        captureSession?.stopRunning()
        
        // Cancel all processing operations
        frameProcessingQueue.cancelAllOperations()
        
        // Clear the preview layer
        mainQueue.async {
            self.videoPreviewLayer?.removeFromSuperlayer()
            self.videoPreviewLayer = nil
            
            // Show select image button when exiting camera mode
            self.selectButton.isHidden = false
        }
    }

    deinit {
        // Ensure resources are properly cleaned up
        frameProcessingQueue.cancelAllOperations()
        stopCameraSession()
        processingQueue.sync { } // Wait for any pending processing to complete
    }

    // MARK: - Video Buffer Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Skip if we're not in camera mode or we're currently processing a frame
        if lastProcessedMode != 2 || frameProcessingQueue.operationCount > 0 {
            return
        }
        
        // Throttle based on time - process at most 1 frame per second
        let currentTime = CACurrentMediaTime()
        if currentTime - self.lastFrameCaptureTime < 1.0 {
            return
        }
        self.lastFrameCaptureTime = currentTime
        
        // Safely capture the image from the buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let currentImage = ciImage.uiImage else {
            return
        }
        
        // Create an operation for processing this frame
        let operation = BlockOperation { [weak self] in
            self?.processFrameInBackground(currentImage)
        }
        
        // Add completion block to handle operation completion
        operation.completionBlock = { [weak operation] in
            if let op = operation, op.isCancelled {
                print("[Process] Frame processing operation was cancelled")
            }
        }
        
        // Add to our serial queue - only one will execute at a time
        frameProcessingQueue.addOperation(operation)
    }
    
    private func processFrameInBackground(_ image: UIImage) {
        // Skip if we're not in camera mode
        if lastProcessedMode != 2 {
            return
        }
        
        do {
            // Both models use 224x224 input size
            let modelInputSize = CGSize(width: 224, height: 224)
            
            // Preprocess the image - avoid force unwrapping
            guard let preprocessedImage = image.resize(to: modelInputSize),
                  let multiArray = try? preprocessedImage.preprocessForML(targetSize: modelInputSize) else {
                return
            }
            
            // Make defensive copies of references to avoid capturing
            guard let detector = seaDetector,
                  let predictor = seaPredictor else {
                return
            }
            
            // Detect sea in the image
            let seaMask = try detector.detectSea(multiArray: multiArray)
            let seaPercentage = SeaDetector.calculateSeaFraction(mask: seaMask)
            let containsSea = seaPercentage > 0.2
            
            // Generate visualization if sea is detected
            var visualizedImage: UIImage? = nil
            if containsSea && SeaDetector.enableVisualization {
                visualizedImage = try detector.generateSeaMaskVisualization(mask: seaMask, originalImage: image)
            }
            
            // Only predict wave/wind if sea is detected
            var predictions: [Float]? = nil
            if containsSea {
                predictions = try predictor.predict(multiArray: multiArray)
            }
            
            // Create intermediate data objects to pass to the main thread
            let frameData = FrameProcessingResult(
                seaPercentage: seaPercentage,
                containsSea: containsSea,
                waveWindPredictions: predictions,
                visualizedImage: visualizedImage,
                originalImage: image
            )
            
            // Update UI on main thread
            mainQueue.async { [weak self] in
                guard let self = self, self.lastProcessedMode == 2 else {
                    return
                }
                self.updateUIWithFrameResults(frameData)
            }
        } catch {
            print("[Error] Frame processing error: \(error)")
        }
    }
    
    // Simple struct to hold frame processing results
    private struct FrameProcessingResult {
        let seaPercentage: Float
        let containsSea: Bool
        let waveWindPredictions: [Float]?
        let visualizedImage: UIImage?
        let originalImage: UIImage
    }
    
    private func updateUIWithFrameResults(_ result: FrameProcessingResult) {
        // Verify we're on the main thread
        assert(Thread.isMainThread, "UI updates must be on main thread")
        
        // Ensure the view controller is still active and in camera mode
        guard lastProcessedMode == 2, 
              view.window != nil, // Check if view is in window hierarchy
              isViewLoaded else {
            print("[Debug] View controller is not active anymore")
            return
        }
        
        // Cache the results in instance variables
        self.seaPercentage = result.seaPercentage
        self.containsSea = result.containsSea
        self.waveWindPredictions = result.waveWindPredictions
        
        // Capture strong references to UI elements to prevent access after deallocation
        guard let label = predictionLabel,
              let overlayView = predictionOverlayView else {
            print("[Debug] UI elements are nil")
            return
        }
        
        // Ensure overlay view is visible
        overlayView.isHidden = false
        
        // Update UI based on results
        if result.containsSea {
            // Format the message
            let messageText: String
            if let predictions = result.waveWindPredictions, predictions.count >= 2 {
                messageText = String(format: "Sea: %d%%\nWind: %.1f m/s\nWave: %.1f m",
                                    Int(result.seaPercentage * 100),
                                    predictions[0],
                                    predictions[1])
            } else {
                messageText = String(format: "Sea: %d%%", Int(result.seaPercentage * 100))
            }
            
            // Update the prediction label
            label.text = messageText
            
            // Display visualization if enabled
            if let visualizedImage = result.visualizedImage, 
               SeaDetector.enableVisualization && 
               showingVisualization {
                // Display visualization overlay
                updateCameraPreviewOverlay(visualizedImage)
            } else if !showingVisualization {
                // Remove any overlay when visualization is off
                clearCameraOverlays()
            }
        } else {
            // No sea detected - prepare and set text before accessing UI
            let percentage = Int(result.seaPercentage * 100)
            let messageText = String(format: "Sea: %d%%\nNo sea detected", percentage)
            
            // Update the label text
            label.text = messageText
            
            // Remove any overlay when no sea is detected
            clearCameraOverlays()
        }
    }
    
    private func clearCameraOverlays() {
        // Must be called on main thread
        assert(Thread.isMainThread)
        
        // Safe removal of overlays
        if let videoLayer = videoPreviewLayer,
           let sublayers = imageView.layer.sublayers {
            for layer in sublayers {
                if layer != videoLayer {
                    layer.removeFromSuperlayer()
                }
            }
        }
    }

    private func updateCameraPreviewOverlay(_ overlayImage: UIImage) {
        // Create a CALayer with the overlay image
        let overlayLayer = CALayer()
        overlayLayer.contents = overlayImage.cgImage
        overlayLayer.frame = imageView.bounds
        overlayLayer.opacity = 0.7 // Semi-transparent overlay
        
        // Remove any existing overlay layers
        imageView.layer.sublayers?.forEach { layer in
            if layer != videoPreviewLayer {
                layer.removeFromSuperlayer()
            }
        }
        
        // Add the new overlay
        if let videoLayer = videoPreviewLayer {
            imageView.layer.insertSublayer(overlayLayer, above: videoLayer)
        }
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

