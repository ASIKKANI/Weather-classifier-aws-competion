# Weather Classification Challenge - 4th Place Solution
**Team Name:** P. Asik kani
**Final Score:** 0.719 (F1-Macro)

## 1. Model & Approach
**Model Architecture:** Custom SE-ResNet (Squeeze-and-Excitation Residual Network)

Instead of relying on pre-trained weights (like VGG16 or ResNet50), I designed and trained a lightweight **ResNet** from scratch, enhanced with **Squeeze-and-Excitation (SE) Attention Blocks**.

* **Residual Blocks:** Allow the network to learn deep features without the vanishing gradient problem.
* **SE Blocks:** Explicitly model the inter-dependencies between channels. This "Feature Recalibration" allows the network to focus on informative features (like rain streaks or fog density) while suppressing less useful background noise.

## 2. Why I Chose This Model
The challenge presented a significant **Domain Shift** between the training data (Earth images) and the test data (Alien/Mars-like images).

* **Avoiding Overfitting:** Standard heavy models (like DenseNet121) tended to overfit to the specific colors of the training set. A custom, smaller SE-ResNet proved more robust to the color variations in the Alien dataset.
* **Texture Priority:** The Squeeze-and-Excitation mechanism helped the model prioritize **structural patterns** (clouds, rain drops) over simple color matching, which was critical for classifying the tinted Alien images correctly.

## 3. Preprocessing & Improvements
To further address the domain shift, I implemented a strict preprocessing pipeline:

* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** The Alien images had distinct, often dark or tinted lighting. I used CLAHE to normalize the Lightness channel (L in LAB color space). This enhanced the local contrast, making texture features like "fog" and "rain" visible even in low-light conditions.
* **Label Smoothing (0.1):** Applied to the loss function to prevent the model from becoming "over-confident" on the training data, improving generalization.
* **Test-Time Augmentation (TTA):** During inference, predictions were made on both the original image and a horizontally flipped version, and then averaged to ensure stability.
