# Réseaux Adversariaux Génératifs (GANs)

Les Réseaux Adversariaux Génératifs (GANs) ont été introduits pour la première fois en 2014 par Ian Goodfellow et al. Depuis lors, ce sujet a ouvert un nouveau domaine de recherche.

En quelques années, la communauté de recherche a produit de nombreux articles sur ce sujet, dont certains ont des noms très intéressants. Vous avez CycleGAN, suivi de BiCycleGAN, suivi de ReCycleGAN, et ainsi de suite.

Avec l'invention des GANs, les modèles génératifs ont commencé à montrer des résultats prometteurs dans la génération d'images réalistes. Les GANs ont connu un succès considérable en Vision par Ordinateur. Récemment, ils ont également commencé à montrer des résultats prometteurs dans l'audio et le texte.

Parmi les formulations de GAN les plus populaires, on trouve :

- Transformer une image d'un domaine à un autre (CycleGAN)
- Générer une image à partir d'une description textuelle (texte-à-image)
- Générer des images de très haute résolution (ProgressiveGAN) et bien d'autres.

Dans ce projet, nous allons expliquer l'architecture de Gans Vanilla, ainsi que [l'implémenter avec Pytorch](cours-gans.ipynb). 

# Setup Instructions

1. Install Anaconda

If you do not have Anaconda installed, download and install it from [here](https://www.anaconda.com/products/distribution).

2. Create and Activate a Virtual Environment

Open a terminal (or Anaconda Prompt on Windows) and run the following commands to create and activate a new virtual environment:

```sh
conda create -n gan_env python=3.10
conda activate gan_env
```
3. Install the libraries needed for the project :

* **Cuda** If you have cuda:

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --upgrade pip
pip install -r requirements.txt 
```

* if you dont have cuda:

```sh
pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0
pip install --upgrade pip
pip install -r requirements.txt 
```