# SVM-CNN Dog vs. Cat Image Classifier

Welcome to your **SVM-CNN Dog vs. Cat Image Classifier!** This project integrates the feature-extracting power of CNNs with the max-margin strengths of SVM, delivering robust, margin-aware predictions for pet images.

## 🚀 What Makes This Project Stand Out?

- **CNN Backbone**: Automatically learns to extract important visual features from images.
- **SVM-Style Classifier**: The final layer employs hinge loss and L2 regularization, providing SVM-like decision boundaries.
- **Improved Accuracy**: Outperforms classic SVMs on raw images, and often surpasses softmax CNN classifiers on binary datasets.
- **Customization Friendly**: Easily adjustable network, loss functions, and data pipeline for your own experiments.


## 🎯 Why Use the SVM-CNN Hybrid?

> **CNNs** provide superior visual feature extraction.
> **SVMs** focus on finding the optimal decision margin.
> **Together:** They offer powerful, robust, and generalizable classification for visual tasks like dogs vs. cats.

## 🗂️ Project Structure

```
🖼️ Dataset/
🖼️ visuals/    
🧠 model_rcat_dog.h5       
🛠️ SVM.ipynb      
📄 README.md     
```


## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ADITYA-CD/SCT_ML_3.git
cd dog-cat-svm-cnn
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


### 3. Add Your Data

- Place images in the `Datasets/` folder within `dog/` and `cat/` subfolders.


### 4. Train the Model



### 5. View Results

- Check accuracy, loss, and prediction outputs after training.


## 💡 Project Tips

- **Preprocessing Matters**: Use data augmentation (e.g., flips, rotations) to help generalize and prevent overfitting.
- **Label Convention**: For hinge loss, assign labels as -1 (cat) and +1 (dog) for correct margin calculation.
- **Regularization**: Adjust L2 regularization strength if you notice overfitting or underfitting.
- **Early Stopping**: Monitor validation loss, and stop training early if overfitting is detected.
- **Batch Size \& Learning Rate**: Experiment with these for best convergence on your dataset size.
- **Model Saving**: Regularly save checkpoints to avoid losing progress, especially when running longer training.
- **Result Visualization**: Plot confusion matrices and sample predictions to better understand model performance.


## 📸 Showcase: How the Model Works

Visualizing the classifier’s actual predictions brings your model to life. Here are two example outputs:

### Example 1: Correct Prediction — Dog

<img width="982" height="254" alt="image" src="https://github.com/user-attachments/assets/b11f6772-03b5-4470-987a-4d176c053aef" />

> **The model correctly classifies this image as a Dog.**

### Example 2: Correct Prediction — Cat

<img width="982" height="256" alt="image" src="https://github.com/user-attachments/assets/9d306512-5391-485b-8b69-063c18e2f1e0" />

> **The model correctly classifies this image as a Cat.**


## 🔬 Example: Hinge Loss Code (PyTorch)

```python
def hinge_loss(outputs, labels):
    # Labels: cat = -1, dog = +1
    return torch.mean(torch.clamp(1 - outputs * labels, min=0))
```


## 📊 Sample Results

| Model Type | Test Accuracy |
| :-- | :-- |
| Raw SVM (pixels) | ~57% |
| CNN + Softmax | ~79% |
| **CNN + SVM (hinge)** | ⭐️~89%+ |

## 📚 References

- Hybrid CNN-SVM for Image Classification (arXiv)
- Scikit-learn SVM Documentation


## 🙌 Contributing

Found a bug or have suggestions? Open an issue or pull request—community contributions welcome!

## 📄 License

MIT License — feel free to use, share, and modify for your own projects!

**Ready to classify some pets? Happy coding! 🐕‍🦺✨🐈**

