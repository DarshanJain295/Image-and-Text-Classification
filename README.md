Project Title: Advanced Neural Network Techniques for Image and Text Classification
Project Overview:
This project focuses on implementing and fine-tuning advanced machine learning models for classification tasks across both image and text data. It combines techniques in softmax regression, contrastive representation learning, parameter-efficient fine-tuning, and knowledge distillation, showcasing the application of these models on diverse datasets like CIFAR-10 and COLA.

Key Tasks and Contributions:
Image Classification with CIFAR-10:

Softmax Regression Implementation:

Developed a multi-class logistic regression model to classify images in the CIFAR-10 dataset, achieving over 80% accuracy on the validation set.
Executed extensive hyperparameter tuning, including batch size selection and early stopping criteria, to optimize the training process.
Visualized training progress by plotting loss and accuracy as functions of iterations.
Contrastive Representation Learning:

Designed and implemented a neural network for contrastive learning to map images to vector representations, enhancing classification accuracy by maximizing the similarity between similar images and minimizing the similarity between different images.
Applied t-SNE to visualize high-dimensional learned representations, analyzing clusters formed based on image labels.
Fine-tuned a multi-class logistic regression classifier using the learned representations, significantly improving classification accuracy.
Neural Network Fine-tuning:

Expanded the architecture by adding additional layers to enhance performance, using techniques like freezing pretrained weights and fine-tuning new layers.
Compared the performance of different models and submitted predictions to an online leaderboard for benchmarking.
Text Classification with COLA Dataset:

Parameter Efficient Fine-Tuning (LoRA):

Integrated Low Rank Adaptation (LoRA) into a GPT-2 model, enabling efficient fine-tuning with reduced computational resources.
Implemented auxiliary linear layers for LoRA, significantly reducing the number of trainable parameters while maintaining high classification accuracy.
Fine-tuned the model on the COLA dataset and optimized hyperparameters, achieving state-of-the-art results.
Knowledge Distillation:

Developed a Recurrent Neural Network (RNN) model to distill knowledge from the fine-tuned GPT-2 model, transferring its capabilities to a smaller, more efficient model.
Conducted experiments comparing the performance of the distilled model with a baseline model trained without distillation, demonstrating the effectiveness of knowledge distillation in preserving model accuracy with fewer resources.
Performance Evaluation:

Generated comprehensive training plots and metrics to analyze model performance across different tasks, providing insights into the trade-offs between model complexity and accuracy.
Technologies and Tools:
Languages: Python
Frameworks: PyTorch, TensorFlow
Libraries: Hugging Face Transformers, t-SNE, LogisticRegression, Contrastive Learning, GPT-2, LoRA
Datasets: CIFAR-10, COLA
Development: Model fine-tuning, hyperparameter optimization, knowledge distillation, and data visualization.