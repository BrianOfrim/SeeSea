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

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        setupUI()
        
        do {
            seaPredictor = try SeaPredictor()
            seaDetector = try SeaDetector()
        } catch {
            print("Failed to initialize models: \(error)")
        }
    }

    func setupUI() {
        // Calculate safe area
        let safeArea = view.safeAreaLayoutGuide
        
        // Image view - takes up the entire screen
        imageView = UIImageView()
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFill
        imageView.backgroundColor = .lightGray
        imageView.clipsToBounds = true
        view.addSubview(imageView)
        
        // Segment control for switching between models
        segmentControl = UISegmentedControl(items: ["Wave/Wind", "Sea Detection"])
        segmentControl.translatesAutoresizingMaskIntoConstraints = false
        segmentControl.selectedSegmentIndex = 0
        segmentControl.backgroundColor = UIColor.white.withAlphaComponent(0.7)
        segmentControl.addTarget(self, action: #selector(segmentChanged), for: .valueChanged)
        view.addSubview(segmentControl)
        
        // Toggle button for switching between original and visualization
        toggleButton = UIButton(type: .system)
        toggleButton.translatesAutoresizingMaskIntoConstraints = false
        
        var toggleConfig = UIButton.Configuration.filled()
        toggleConfig.title = "Show Original"
        toggleConfig.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.7)
        toggleConfig.baseForegroundColor = .white
        toggleConfig.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16)
        toggleConfig.cornerStyle = .medium
        toggleButton.configuration = toggleConfig
        
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
        overlayView.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        overlayView.layer.cornerRadius = 6
        
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
        config.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.7)
        config.baseForegroundColor = .white
        config.contentInsets = NSDirectionalEdgeInsets(top: 10, leading: 20, bottom: 10, trailing: 20)
        config.cornerStyle = .large
        selectButton.configuration = config
        
        selectButton.addTarget(self, action: #selector(openPhotoLibrary), for: .touchUpInside)
        view.addSubview(selectButton)
        
        NSLayoutConstraint.activate([
            // Image view constraints - fill the entire view
            imageView.topAnchor.constraint(equalTo: view.topAnchor),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Segment control - at the top
            segmentControl.topAnchor.constraint(equalTo: safeArea.topAnchor, constant: 12),
            segmentControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            segmentControl.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.8),
            
            // Toggle button - below segment control
            toggleButton.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            toggleButton.trailingAnchor.constraint(equalTo: safeArea.trailingAnchor, constant: -12),
            
            // Select button constraints - floating at the bottom
            selectButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            selectButton.bottomAnchor.constraint(equalTo: safeArea.bottomAnchor, constant: -20),
            
            // Overlay view constraints - position in top left corner with padding
            overlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor, constant: 12),
            overlayView.topAnchor.constraint(equalTo: segmentControl.bottomAnchor, constant: 12),
            overlayView.widthAnchor.constraint(lessThanOrEqualTo: imageView.widthAnchor, multiplier: 0.7),
            
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
        // Print frames after layout
        print("Image view frame: \(imageView.frame)")
        print("Prediction label frame: \(predictionLabel.frame)")
        print("Activity indicator frame: \(activityIndicator.frame)")
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
            imageView.image = selectedImage
            predictionLabel.text = "Processing..."
            activityIndicator.startAnimating()
            
            // Hide toggle button when new image is selected
            toggleButton.isHidden = true
            
            // Capture the selected segment index on the main thread
            let selectedIndex = segmentControl.selectedSegmentIndex
            
            DispatchQueue.global(qos: .userInitiated).async {
                if selectedIndex == 0 {
                    // Wave/Wind prediction
                    self.makePrediction(for: selectedImage)
                } else {
                    // Sea detection
                    self.detectSea(in: selectedImage)
                }
            }
        }
        dismiss(animated: true)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true)
    }

    private func makePrediction(for image: UIImage) {
        do {
            let predictions = try seaPredictor?.predict(image: image)
            
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()
                
                if let predictions = predictions, predictions.count >= 2 {
                    let wind_speed_mps = predictions[0]
                    let wave_height_m = predictions[1]
                    
                    self.predictionLabel.text = String(format: "Wind Speed: %.1f m/s\nWave Height: %.1f m",
                                                wind_speed_mps, wave_height_m)
                    self.predictionLabel.superview?.isHidden = false
                } else {
                    self.predictionLabel.text = "Invalid prediction result"
                    self.predictionLabel.superview?.isHidden = false
                }
            }
        } catch {
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()
                self.predictionLabel.text = "Prediction failed"
                self.predictionLabel.superview?.isHidden = false
            }
        }
    }

    @objc func segmentChanged(_ sender: UISegmentedControl) {
        if let image = imageView.image {
            predictionLabel.text = "Processing..."
            activityIndicator.startAnimating()
            
            // Capture the selected segment index on the main thread
            let selectedIndex = sender.selectedSegmentIndex
            
            // Hide or show toggle button based on mode
            toggleButton.isHidden = selectedIndex == 0
            
            DispatchQueue.global(qos: .userInitiated).async {
                if selectedIndex == 0 {
                    // Wave/Wind prediction
                    self.makePrediction(for: image)
                } else {
                    // Sea detection
                    self.detectSea(in: image)
                }
            }
        }
    }

    private func detectSea(in image: UIImage) {
        do {
            // Store the original image
            self.originalImage = image
            
            // Get the sea percentage and containsSea boolean
            let (percentage, containsSea) = try seaDetector?.detectSea(in: image) ?? (0, false)
            
            // Generate the visualization with sea mask overlay
            if let visualizedImage = try seaDetector?.generateSeaMaskVisualization(for: image) {
                // Store the visualization image
                self.visualizationImage = visualizedImage
                
                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    
                    // Update the image view with the visualization
                    self.imageView.image = self.visualizationImage
                    
                    // Update the text label
                    if containsSea {
                        self.predictionLabel.text = "Sea detected: \(Int(percentage * 100))%"
                    } else {
                        self.predictionLabel.text = "No sea detected"
                    }
                    self.predictionLabel.superview?.isHidden = false
                    
                    // Show the toggle button
                    self.toggleButton.isHidden = false
                    self.showingVisualization = true
                    self.toggleButton.configuration?.title = "Show Original"
                    self.toggleButton.configuration?.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.7)
                }
            } else {
                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    self.predictionLabel.text = "Sea detection: \(Int(percentage * 100))%"
                    self.predictionLabel.superview?.isHidden = false
                    self.toggleButton.isHidden = true
                }
            }
        } catch {
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()
                self.predictionLabel.text = "Detection failed: \(error.localizedDescription)"
                self.predictionLabel.superview?.isHidden = false
                self.toggleButton.isHidden = true
            }
        }
    }

    @objc func toggleVisualization() {
        showingVisualization = !showingVisualization
        
        if showingVisualization {
            // Show visualization
            toggleButton.configuration?.title = "Show Original"
            toggleButton.configuration?.baseBackgroundColor = UIColor.systemGreen.withAlphaComponent(0.7)
            imageView.image = visualizationImage
        } else {
            // Show original
            toggleButton.configuration?.title = "Show Visualization"
            toggleButton.configuration?.baseBackgroundColor = UIColor.systemBlue.withAlphaComponent(0.7)
            imageView.image = originalImage
        }
    }
}
