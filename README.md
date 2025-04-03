# 🌠 Star Wars: The Pulsar Chronicles 🌏

![Pulsar](image.png)

Welcome to the **HTRU2 Dataset**, where the stars themselves reveal secrets of the universe… or maybe just a bunch of radio noise! 🚀✨  
This intergalactic treasure trove of data holds the key to differentiating between real pulsars (those cool, rhythmic neutron stars that behave like cosmic DJs 🎵) and impostors (a.k.a. random noise, the universe’s equivalent of static 🔊).

---

## 🛰️ What’s in the Data?
Think of this dataset as a **cosmic recipe book**, but instead of ingredients, you have **features** that help you decode the rhythm of the stars. Here’s what’s cooking:  

🔹 **Mean Integrated Profile** – The cosmic heartbeat ❤️: How strong is the star’s signal over time?  
🔹 **SD Integrated Profile** – Even stars have their ups and downs 📉: This is the standard deviation of that signal.  
🔹 **Excess Kurtosis of Integrated Profile** – Is your star signaling in a smooth groove, or throwing wild, unexpected beats? 🎛️  
🔹 **Skewness of Integrated Profile** – Are those signals leaning left, right, or keeping it neutral? 🔄  
🔹 **Mean DM-SNR Curve** – A fancy way of saying, **"How loud is the star’s voice in the galactic chatter?"** 🔊  
🔹 **SD DM-SNR Curve** – Because even stars can’t keep their voice at a consistent volume 🔈  
🔹 **Excess Kurtosis of DM-SNR Curve** – Are the star’s tones steady, or is it a bit too dramatic? 🎭  
🔹 **Skewness of DM-SNR Curve** – Another hint at whether the star prefers to “lean” in its cosmic karaoke 🎙️  
🔹 **Target** – Is this the real deal? **(1 = Pulsar 🌟, 0 = Non-Pulsar ❌)**  

---

## 🔬 Why Is This Dataset Special?
Imagine being a **pulsar detective** 🕵️‍♂️, trying to distinguish actual celestial rock stars from a universe full of white noise.  
It’s like finding a needle in a haystack—except the needle is a **neutron star**, and the haystack is the size of a **galaxy**. 🌌  

Think you’ve got what it takes to find the real stars of the universe?  
Grab your **Logistic Regression** toolbelt 🛠️ and let’s get stargazing! 🚀✨  

---

## 📚 Assignment Tasks 🤩

### 1️⃣ Data Exploration and Preprocessing
🔹 **Load the Dataset:**  
- Load the HTRU dataset from the CSV file 📂  
- Display the first few rows to familiarize yourself with the data 👀  

🔹 **Perform Exploratory Data Analysis (EDA):**  
- Summarize the data (mean, median, standard deviation, null values, etc.) 🧐  
- Visualize feature distributions using **histograms** or **box plots** 📊  
- Check the balance of the target variable (is the dataset imbalanced?) ⚖️  

🔹 **Preprocess the Data:**  
- Scale all continuous features using `StandardScaler` 📏  
- Split the dataset into **training (80%)** and **test (20%)** subsets 🏋️‍♂️  

---

### 2️⃣ Implement Logistic Regression
- Train a **Logistic Regression** model using `sklearn`’s `LogisticRegression` 📈  
- Use it to **predict** whether a sample is a **pulsar or not** 🌟❌  
- Experiment with the **C parameter** (regularization strength) and observe its impact ⚙️  

---

### 3️⃣ Model Evaluation
Evaluate the model using these **metrics**:  
✅ **Accuracy**  
✅ **Precision**  
✅ **Recall**  
✅ **F1 Score**  

🔹 **Confusion Matrix:**  
- How many pulsars were **correctly identified**?  
- How many **false positives** and **false negatives** occurred?  

---

### 4️⃣ Visualization 📊
🔹 **Feature Relationships:**  
- Use **pairplots** or **scatter plots** to visualize how features like *Mean Integrated Profile* and *SD Integrated Profile* relate to each class.  

🔹 **Decision Boundary:**  
- Choose two features (e.g., *Mean Integrated Profile* and *SD Integrated Profile*) and plot the **decision boundary** of the Logistic Regression model.  

---

### 5️⃣ Discussion 🧠
Write a short **report** summarizing your findings:  
📌 **Which features seem most significant** in distinguishing pulsars from non-pulsars?  
📌 **How well did the Logistic Regression model perform**, based on evaluation metrics?  
📌 **Discuss any challenges** faced during preprocessing or training.  

---

## 🎯 Bonus Tasks (Optional) 💥  
🔹 **Handle Imbalanced Data:**  
- If the dataset is **imbalanced**, try techniques like **SMOTE** or class weighting in Sklearn’s `LogisticRegression`.  

🔹 **Compare Models:**  
- Train and evaluate **another classification model** (e.g., Random Forest 🌳, SVM 🕵️) and compare its performance with Logistic Regression.  

---

## 💡 How to Run the Streamlit App 🚀
1️⃣ Install the dependencies:  
   ```bash
   pip install streamlit scikit-learn pandas plotly
