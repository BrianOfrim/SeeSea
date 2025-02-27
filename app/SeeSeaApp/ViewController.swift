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
        let padding: CGFloat = 20
        
        // Image view - takes up top portion
        imageView = UIImageView()
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .lightGray
        imageView.layer.cornerRadius = 8
        imageView.clipsToBounds = true
        view.addSubview(imageView)
        
        // Activity indicator
        activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.hidesWhenStopped = true
        view.addSubview(activityIndicator)
        
        // Prediction label
        predictionLabel = UILabel()
        predictionLabel.translatesAutoresizingMaskIntoConstraints = false
        predictionLabel.numberOfLines = 0
        predictionLabel.textAlignment = .center
        predictionLabel.backgroundColor = .systemGray6
        predictionLabel.layer.cornerRadius = 8
        predictionLabel.clipsToBounds = true
        predictionLabel.font = UIFont.systemFont(ofSize: 18, weight: .medium)
        predictionLabel.text = "Prediction will appear here"
        predictionLabel.layer.borderWidth = 1
        predictionLabel.layer.borderColor = UIColor.black.cgColor
        view.addSubview(predictionLabel)
        
        // Button
        let button = UIButton(type: .system)
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setTitle("Pick an Image", for: .normal)
        button.backgroundColor = .systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 8
        button.addTarget(self, action: #selector(openPhotoLibrary), for: .touchUpInside)
        button.layer.borderWidth = 1  // Debug border
        button.layer.borderColor = UIColor.black.cgColor
        view.addSubview(button)
        
        // Set up constraints with fixed heights
        NSLayoutConstraint.activate([
            // Image view - fixed height of 250
            imageView.topAnchor.constraint(equalTo: safeArea.topAnchor, constant: padding),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: padding),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -padding),
            imageView.heightAnchor.constraint(equalToConstant: 250),
            
            // Activity indicator centered in image view
            activityIndicator.centerXAnchor.constraint(equalTo: imageView.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: imageView.centerYAnchor),
            
            // Prediction label - fixed height of 80
            predictionLabel.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: padding),
            predictionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: padding),
            predictionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -padding),
            predictionLabel.heightAnchor.constraint(equalToConstant: 80),
            
            // Button - fixed height of 50, with bottom constraint to ensure it's visible
            button.topAnchor.constraint(equalTo: predictionLabel.bottomAnchor, constant: padding),
            button.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: padding),
            button.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -padding),
            button.heightAnchor.constraint(equalToConstant: 50),
            button.bottomAnchor.constraint(lessThanOrEqualTo: safeArea.bottomAnchor, constant: -padding)
        ])
        
        // Print view frame for debugging
        print("View frame: \(view.frame)")
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
                    print("Updated label with: \(self.predictionLabel.text ?? "nil")")
                } else {
                    self.predictionLabel.text = "Invalid prediction result"
                }
            }
        } catch {
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()
                self.predictionLabel.text = "Prediction failed: \(error.localizedDescription)"
                print("Error: \(error.localizedDescription)")
            }
        }
    }
}
