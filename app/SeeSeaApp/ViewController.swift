import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }

    func setupUI() {
        // Add an image view
        imageView = UIImageView(frame: CGRect(x: 20, y: 100, width: 300, height: 300))
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .lightGray
        view.addSubview(imageView)

        // Add a button to open the photo library
        let button = UIButton(type: .system)
        button.frame = CGRect(x: 20, y: 420, width: 300, height: 50)
        button.setTitle("Pick an Image", for: .normal)
        button.addTarget(self, action: #selector(openPhotoLibrary), for: .touchUpInside)
        view.addSubview(button)
    }

    @objc func openPhotoLibrary() {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        picker.allowsEditing = true  // Allow cropping (optional)
        present(picker, animated: true)
    }

    // Handle the selected image
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let selectedImage = info[.editedImage] as? UIImage ?? info[.originalImage] as? UIImage {
            imageView.image = selectedImage
        }
        dismiss(animated: true)
    }

    // Handle cancel button
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true)
    }
}
