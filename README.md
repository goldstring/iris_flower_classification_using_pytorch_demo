
<h1>Iris Flower Classification using PyTorch</h1>
    <p>This repository contains a simple implementation of a classification model using Artificial Neural Networks (ANN) to classify the Iris flower dataset. The dataset is used for a multi-class classification problem to predict the species of Iris flowers (Setosa, Versicolor, Virginica).</p>
    <h2>Dataset</h2>
    <p>The dataset used for this project is the <a href="https://archive.ics.uci.edu/ml/datasets/iris" target="_blank">Iris dataset</a> which includes the following features:</p>
    <ul>
        <li><strong>sepal length</strong></li>
        <li><strong>sepal width</strong></li>
        <li><strong>petal length</strong></li>
        <li><strong>petal width</strong></li>
    </ul>
    <p>The task is to classify the flowers into one of the three species:</p>
    <ul>
        <li><strong>Setosa</strong></li>
        <li><strong>Versicolor</strong></li>
        <li><strong>Virginica</strong></li>
    </ul>
    <h2>Project Overview</h2>
    <p>This project demonstrates how to use a simple Feedforward Neural Network (ANN) built using PyTorch for classification purposes.</p>
    <h3>Model Architecture:</h3>
    <p>The model consists of the following layers:</p>
    <ul>
        <li><strong>Input layer</strong>: Takes in 4 features (sepal length, sepal width, petal length, petal width).</li>
        <li><strong>Hidden layers</strong>:
            <ul>
                <li>1st Hidden layer: 64 neurons with ReLU activation.</li>
                <li>2nd Hidden layer: 32 neurons with ReLU activation.</li>
                <li>3rd Hidden layer: 16 neurons with ReLU activation.</li>
                <li>4th Hidden layer: 8 neurons with ReLU activation.</li>
            </ul>
        </li>
        <li><strong>Output layer</strong>: 3 neurons, one for each class of Iris flower, using softmax activation for multi-class classification.</li>
    </ul>
    <h3>Loss Function:</h3>
    <p><strong>CrossEntropyLoss</strong> is used for multi-class classification.</p>
    <h3>Optimizer:</h3>
    <p><strong>Adam Optimizer</strong> is used for training the model with a learning rate of 0.001.</p>
    <h2>Requirements</h2>
    <ul>
        <li>Python 3.x</li>
        <li>PyTorch</li>
        <li>Scikit-learn</li>
        <li>NumPy</li>
    </ul>
    <h2>Installation</h2>
    <p>To install the required dependencies, create a virtual environment and install the necessary libraries:</p>
    <pre><code>pip install -r requirements.txt</code></pre>
    <p>The <strong>requirements.txt</strong> file contains the following dependencies:</p>
    <pre><code>torch==1.12.1
scikit-learn==1.0.2
numpy==1.21.0</code></pre>
    <h2>How to Run</h2>
    <ol>
        <li>Clone the repository:</li>
        <pre><code>git clone https://github.com/yourusername/iris-flower-classification-pytorch.git</code></pre>
        <pre><code>cd iris-flower-classification-pytorch</code></pre>
        <li>Train the model:</li>
        <pre><code>python train_model.py</code></pre>
        <p>This will train the model using the Iris dataset and print the loss and accuracy at each epoch.</p>
        <li>Evaluate the model:</li>
        <p>Once training is completed, the script will evaluate the model's performance on the test set and print the classification report, including accuracy, precision, recall, and F1 score.</p>
    </ol>
    <h2>Training Output</h2>
    <p>After training the model for 300 epochs, the following results were obtained:</p>
    <h3>Training Accuracy: 0.9750</h3>
    <h3>Test Accuracy: 1.0000</h3>
    <h3>Classification Report:</h3>
    <img src="https://github.com/goldstring/iris_flower_classification_using_pytorch_demo/blob/main/loss_vs_epoch.png?raw=true"/>
    <img src="https://github.com/goldstring/iris_flower_classification_using_pytorch_demo/blob/main/confusion_matrix.png?raw=true"/>
    <h3>Test Loss: 0.0569</h3>
    <h2>Model Code Overview</h2>
    <h3><strong>train_model.py</strong></h3>
    <ul>
        <li>Loads the Iris dataset.</li>
        <li>Prepares the data for training and testing.</li>
        <li>Defines the neural network architecture (ANN).</li>
        <li>Trains the model using backpropagation and the Adam optimizer.</li>
        <li>Evaluates the model using the test data and prints the classification report.</li>
    </ul>
    <h3><strong>ANN Class</strong></h3>
    <p>The <strong>ANN</strong> class defines the architecture of the artificial neural network. It uses <code>nn.Sequential</code> to define the layers and ReLU activations in between.</p>
    <h3>Model Evaluation</h3>
    <p>The model is evaluated using:</p>
    <ul>
        <li><strong>Accuracy</strong>: Overall classification accuracy.</li>
        <li><strong>Confusion Matrix</strong>: Shows the performance of the classification model.</li>
        <li><strong>Precision, Recall, F1-Score</strong>: Detailed classification metrics.</li>
        <li><strong>Test Loss</strong>: The final loss after training.</li>
    </ul>
    <h2>Results</h2>
    <p>The model achieved an <strong>accuracy of 100%</strong> on the test set and a low test loss, indicating that it is performing well on the Iris classification task.</p>
    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
    <h2>Acknowledgements</h2>
    <p>The Iris dataset is available in the public domain and was sourced from <a href="https://archive.ics.uci.edu/ml/datasets/iris">UCI Machine Learning Repository</a>.</p>
