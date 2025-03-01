import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    var imageView: UIImageView!
    private var seaPredictor: SeaPredictor?
    private var predictionLabel: UILabel!
    private var activityIndicator: UIActivityIndicatorView!

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        setupUI()
        
        do {
            seaPredictor = try SeaPredictor()
        } catch {
            print("Failed to initialize predictor: \(error)")
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
        selectButton.setTitle("Select Image", for: .normal)
        selectButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.7)
        selectButton.setTitleColor(.white, for: .normal)
        selectButton.layer.cornerRadius = 20
        selectButton.contentEdgeInsets = UIEdgeInsets(top: 10, left: 20, bottom: 10, right: 20)
        selectButton.addTarget(self, action: #selector(openPhotoLibrary), for: .touchUpInside)
        view.addSubview(selectButton)
        
        NSLayoutConstraint.activate([
            // Image view constraints - fill the entire view
            imageView.topAnchor.constraint(equalTo: view.topAnchor),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Select button constraints - floating at the bottom
            selectButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            selectButton.bottomAnchor.constraint(equalTo: safeArea.bottomAnchor, constant: -20),
            
            // Overlay view constraints - position in top left corner with padding
            overlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor, constant: 12),
            overlayView.topAnchor.constraint(equalTo: safeArea.topAnchor, constant: 12),
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
            
            DispatchQueue.global(qos: .userInitiated).async {
                self.makePrediction(for: selectedImage)
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
                    let wave_height_m = predictions[0]
                    let wind_speed_mps = predictions[1]
                    
                    self.predictionLabel.text = String(format: "Wave Height: %.1f m\nWind Speed: %.1f m/s",
                                                wave_height_m, wind_speed_mps)
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
}
