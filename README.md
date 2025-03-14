# **Tire Texture Classification Project**
This project focuses on training a computer to analyze tire images and identify whether they are cracked or intact. By leveraging a technique called computer vision, it enhances the computer's ability to recognize and differentiate between these images more effectively.

---

## **What’s the Point?**
Imagine you’re in the car business or driving around. Tires with cracks are a big no-no. They can cause accidents. This project shows how we can use technology to automatically catch these cracks before they become a problem. Sounds cool, right?

---

## **What Did We Use?**
1. **Dataset**:
   - We used a bunch of tire pictures—some cracked, some not. These were organized into folders:
     - `training_data/`: For teaching the computer.
     - `testing_data/`: For checking how well it learned.
   - The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition).

2. **Libraries**:
   - **TensorFlow/Keras**: For creating and training the model.
   - **Matplotlib**: For making nice graphs.
   - **Seaborn**: To make a confusion matrix (basically a way to see mistakes).

3. **Techniques**:
   - Image augmentation: This is like adding filters to photos. It rotates, zooms, flips, and tweaks the images so the model gets smarter.

---

## **How Did We Do It?**
1. **Load the Data**:
   - We first loaded all the tire pictures and labeled them as either `cracked` or `normal`.
   - We also resized the images to make them easier to process (128x128 pixels).

2. **Teach the Model (Training)**:
   - We used a type of AI called a **Convolutional Neural Network (CNN)**. Think of it like a child learning to identify different animals—it starts simple and gets better over time.

3. **Make It Better (Augmentation)**:
   - We used image tweaks (like zoom, flip, and rotate) to teach the computer to handle different variations of the same picture.

4. **Test the Model**:
   - After training, we tested the computer on new pictures to see how well it learned.

5. **Compare Results**:
   - We compared the computer’s results with and without image augmentation.

---

## **Results**
| Model      | Test Accuracy | Validation Gap |
|------------|---------------|----------------|
| Original   | 85%           | 10%            |
| Augmented  | 91%           | 5%             |

**What this means**:
- The computer got 6% better at spotting cracked tires with augmentation.
- The gap between training and testing also reduced, meaning it generalized better.

---

## **How Do You Run This?**
1. Clone the project:
   ```bash
   git clone https://github.com/lucienmakutano/image_aug.git
   cd image_aug
   
   pip install -r requirements.txt
   
   python main.py
   ```
