Weather Classification Challenge - 4th Place Solution
Team Name: P. Asik kani
Final Score: 0.719 (F1-Macro)

1. The Core Challenge: Generalization vs. Overfitting
The biggest hurdle in this competition wasn't just classifying weather; it was dealing with the strict constraint of "No Pre-trained Models."

Without the safety net of ImageNet weights (like VGG16 or ResNet50), I had to train a model from scratch on a relatively small dataset.

The Struggle: My initial custom CNNs suffered heavily from overfitting. They would rapidly hit 99% accuracy on the training set but perform poorly on the test set.

The Diagnosis: The models were "memorizing" the training data. For example, they learned that "Blue Background = Cloudy" or "Grey Background = Foggy." This broke down completely on the Test Set, where lighting conditions varied (e.g., a "Sunrise" image might look hazy like fog, or a "Rain" image might be dark and grainy).

2. The Solution: A "Texture-Aware" Architecture
To stop the model from cheating by looking at background colors, I needed an architecture that forced it to focus on structural details—like the vertical streaks of rain or the fluffiness of clouds.

My Choice: Custom SE-ResNet
I engineered a lightweight Residual Network (ResNet) integrated with Squeeze-and-Excitation (SE) Attention Blocks.

Why SE-ResNet?
Standard convolutions treat every pixel as equally important. The "Squeeze-and-Excitation" blocks add a layer of intelligence: they explicitly model the relationship between channels. This allows the network to say, "This feature map looks like rain streaks—pay attention to it," while ignoring irrelevant background noise.

This architecture was the turning point where my model stopped overfitting and started actually learning the weather patterns.

3. Preprocessing: Solving the "Lighting" Problem
A major issue I noticed during data analysis was the inconsistent lighting in the test images. Some were over-exposed (too bright), while others were hidden in shadows.

The Fix: CLAHE (Contrast Limited Adaptive Histogram Equalization)
Standard brightness normalization wasn't enough. I implemented a pipeline that:

Converts images to the LAB Color Space.

Applies CLAHE specifically to the 'L' (Lightness) channel.

Converts back to RGB.

Why this mattered: This effectively "equalized" the dataset. It brought out hidden details in dark, foggy images and reduced the glare in bright sunrise images, giving the model a consistent input regardless of the time of day.

4. Final Stabilizers
To secure the 0.719 score, I added two final safeguards against overfitting:

Label Smoothing (0.1): I realized my model was being "too confident" (predicting 100% probability), which is dangerous on unseen data. Label smoothing forced the model to be more conservative, improving robustness.

Test-Time Augmentation (TTA): For the final submission, I ran inference on both the original image and a horizontally flipped version. Averaging these predictions acted as a "sanity check," filtering out random errors.
