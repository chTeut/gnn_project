# Generative Neural Networks - Project Repository

Students: Jonas Gann, Christian Teutsch

This repository contains the code for the project of the course "Generative Neural Networks" at the University of Heidelberg. The objective of this project was to implement and train a Transformer model in order to gain a deeper understanding of the architecture. To keep the model complexity and resource consumption at a reasonable level, we decided not to train the model on natural language but on bash histories. We eventually wanted the model to be capable of being used as an autocompletion tool for the bash command line.

We implemented two versions of the model:

1. A simple Transformer model using mainly PyTorch: [Transformer](./Transformer)
2. A Transformer model using FastAI: [FastAI](./FastAI)

We were able to publish the model weights for the simple Transformer model. Use the [transformer.ipynb](./Transformer/transformer.ipynb) notebook to load the model and generate new bash commands.
