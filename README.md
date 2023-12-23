# Movie Spoiler Detection System

## Overview

Welcome to the Movie Spoiler Detection System! This project focuses on addressing the issue of spoilers in online movie and show reviews. The proliferation of spoilers can diminish the viewer's enjoyment of the content, and our goal is to develop a robust spoiler detection system using machine learning.

## Project Highlights

### Objectives

1. **Spoiler Detection System:** Develop a system capable of identifying potential spoilers within reviews.
2. **Machine Learning Model:** Build a robust model to effectively identify spoilers across various content.
3. **User Experience:** Enhance the reading experience by providing information about the presence of spoilers.
4. **Informed Choices:** Improve the selectivity of online movie reviews, enabling consumers to make informed choices while avoiding unintended spoilers.

### Dataset

We utilized the IMDB Spoiler dataset for training our model. The dataset contains meta-data about items and user reviews, with a focus on the "review_text" and "is_spoiler" columns. To access the dataset, please follow [this link](<https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset/>).

### Exploratory Data Analysis (EDA)

EDA was conducted to understand the dataset better. It included count plots, word clouds, pie charts, and exploration of the relationship between review length, ratings, and spoilers.

### Models Explored

1. **Decision Tree Classifier:**
   - TF-IDF vectorization with a maximum depth of 8.
   - Test accuracy: 70.90%, AUC: 0.62.

2. **Multinomial Naïve Bayes Classifier:**
   - TF-IDF vectorization.
   - Accuracy: 71.67%, AUC: 0.56.

3. **Custom Neural Network (BERT-based):**
   - Custom BERT model with a test accuracy of 75.42% and AUC: 0.62.

### Web Application with Flask

A user-friendly web application was built using Flask. Users can input movie reviews, and the application predicts whether the review contains spoilers using our custom BERT model.


## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone <repository_link>`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Access the IMDB Spoiler dataset from [this link](<https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset/>).
4. Combine the model checkpoints by using the following command `cat model_checkpoint_part* > combined_model_checkpoint.pth`
5. Check if the command has worked`ls -lh combined_model_checkpoint.pth`
6. If you are not able to run the above commands, you can also download the trained model from [this link](<https://drive.google.com/file/d/1ubNRipJrWL08RhR-vJ4XjXkGNh_iV_q_/view?usp=share_link>)
7. Run the application: `python app.py`

## Conclusion

Thank you for exploring the Movie Spoiler Detection System with us. We hope this system enhances your movie-watching experience. For any questions or discussions, feel free to reach out.

Happy movie watching! 🎬
