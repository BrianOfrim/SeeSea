import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    var imageView: UIImageView!
    private var seaPredictor: SeaPredictor?
    private var seaDetector: SeaDetector?
    private var predictionLabel: UILabel!
    private var activityIndicator: UIActivityIndicatorView!
    private var segmentControl: UISegmentedControl!
    private var toggleButton: UIButton!
    
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

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set up background image
        setupBackgroundImage()
        
        // Set up UI components
        setupUI()
        
        do {
            seaPredictor = try SeaPredictor()
            seaDetector = try SeaDetector()
        } catch {
            print("Failed to initialize models: \(error)")
        }
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
        segmentControl = UISegmentedControl(items: ["Wave/Wind", "Sea Detection"])
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
        let overlayView = UIView()
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        overlayView.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        overlayView.layer.cornerRadius = 8
        overlayView.layer.shadowColor = UIColor.black.cgColor
        overlayView.layer.shadowOffset = CGSize(width: 0, height: 2)
        overlayView.layer.shadowRadius = 4
        overlayView.layer.shadowOpacity = 0.4
        
        imageView.addSubview(overlayView)
        overlayView.addSubview(predictionLabel)
        
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
            overlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor, constant: 12),
            overlayView.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            overlayView.trailingAnchor.constraint(lessThanOrEqualTo: toggleButton.leadingAnchor, constant: -12),
            overlayView.widthAnchor.constraint(lessThanOrEqualTo: imageView.widthAnchor, multiplier: 0.6),
            
            // Prediction label constraints - inside the overlay with padding
            predictionLabel.topAnchor.constraint(equalTo: overlayView.topAnchor, constant: 6),
            predictionLabel.bottomAnchor.constraint(equalTo: overlayView.bottomAnchor, constant: -6),
            predictionLabel.leadingAnchor.constraint(equalTo: overlayView.leadingAnchor, constant: 8),
            predictionLabel.trailingAnchor.constraint(equalTo: overlayView.trailingAnchor, constant: -8),
            
            // Activity indicator constraints
            activityIndicator.centerXAnchor.constraint(equalTo: imageView.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: imageView.centerYAnchor)
        ])
        
        overlayView.isHidden = true  // Initially hide the overlay
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
    }

    @objc func openPhotoLibrary() {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        picker.allowsEditing = true
        present(picker, animated: true)
        
        // Hide the overlay when selecting a new image
        predictionLabel.superview?.isHidden = true
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
                            
                            // Detection step
                            let (percentage, containsSea, _) = try self.seaDetector?.detectSea(multiArray: sharedMultiArray, processingMode: "cpu") ?? (0, false, nil)
                            self.seaPercentage = percentage
                            self.containsSea = containsSea
                            
                            let seaDetectionEndTime = CFAbsoluteTimeGetCurrent()
                            let seaDetectionTime = seaDetectionEndTime - seaDetectionStartTime
                            print("Sea detection completed in \(String(format: "%.3f", seaDetectionTime)) seconds")
                            
                            if(SeaDetector.enableVisualization){
                                // Visualization step
                                if let visualizedImage = try self.seaDetector?.generateSeaMaskVisualization(multiArray: sharedMultiArray, originalImage: selectedImage) {
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
                        self.predictionLabel.superview?.isHidden = false
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
        // Only proceed if we have an original image
        if originalImage != nil {
            // Capture the selected segment index on the main thread
            let selectedIndex = sender.selectedSegmentIndex
            
            // Hide or show toggle button based on mode and visualization setting
            toggleButton.isHidden = selectedIndex == 0 || !SeaDetector.enableVisualization
            
            // Update the last processed mode
            lastProcessedMode = selectedIndex
            
            // Just update the UI with existing results
            if selectedIndex == 0 {
                // Show wave/wind results
                showWaveWindResults()
            } else {
                // Show sea detection results
                showSeaDetectionResults()
            }
        } else {
            // No image selected yet
            DispatchQueue.main.async {
                self.predictionLabel.text = "Please select an image first"
                self.predictionLabel.superview?.isHidden = false
            }
        }
    }

    @objc func toggleVisualization() {
        showingVisualization = !showingVisualization
        
        // Update toggle button configuration
        var config = UIButton.Configuration.filled()
        
        if showingVisualization {
            // Show visualization
            config.title = "Overlay"
            config.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
            if let visualizedImage = visualizationImage {
                imageView.image = visualizedImage
            } else {
                print("Warning: Visualization image is nil")
            }
        } else {
            // Show original
            config.title = "Original"
            config.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
            if let originalImage = originalImage {
                imageView.image = originalImage
            } else {
                print("Warning: Original image is nil")
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

    // Helper method to show wave/wind results without recomputing
    private func showWaveWindResults() {
        if let predictions = waveWindPredictions, predictions.count >= 2 {
            let wind_speed_mps = predictions[0]
            let wave_height_m = predictions[1]
            
            self.predictionLabel.text = String(format: "Wind Speed: %.1f m/s\nWave Height: %.1f m",
                                        wind_speed_mps, wave_height_m)
            self.predictionLabel.superview?.isHidden = false
            
            // Show original image if we have one
            if let originalImage = self.originalImage {
                self.imageView.image = originalImage
            }
        } else {
            self.predictionLabel.text = "Wave/Wind prediction results not available yet"
            self.predictionLabel.superview?.isHidden = false
        }
    }
    
    // Helper method to show sea detection results without recomputing
    private func showSeaDetectionResults() {
        // Check if we have sea detection results
        if visualizationImage == nil {
            self.predictionLabel.text = "Sea detection results not available yet"
            self.predictionLabel.superview?.isHidden = false
            self.toggleButton.isHidden = true
            return
        }
        
        // Update the text label with existing results
        if containsSea {
            self.predictionLabel.text = "Sea detected: \(Int(seaPercentage * 100))%"
        } else {
            self.predictionLabel.text = "No sea detected"
        }
        self.predictionLabel.superview?.isHidden = false
        
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
}

