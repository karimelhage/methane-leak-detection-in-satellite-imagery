# Methane Leak Detection in Satellite Imagery Using ResNet 
## Authors
The project is done by [Leonardo Bassili](https://github.com/leobas0), [Antoine Cloute](https://github.com/AntAI-git), [Karim El Hage](https://github.com/karimelhage),[Yasmina Hobeika](https://github.com/yasminahobeika), [Annabelle Luo](https://github.com/annabelleluo), [Ali Najem](https://github.com/najemali), [Amine Zaamoun](https://github.com/Zaamine) for a  Hackathon in partnership with Mckinsey & Company Quantum: Black. The team has awarded **Second Place** for it's work by the jury.

## Objective
The objective of the project is to develop an application capable of detecting methane leaks in provided satellite imagery  to assist with MRV: monitoring, reporting, and verification

## Current Build
The current build of the application is only capable of image classification (methane leak or no methane leak). There is also a GradCAM functionality capable to give a form of interpretability to the results.

## Project Contents

1 - Models: Contains the training methdology of the ResNet model used for classification in the application. This includes train-test split methdology, image augmentation training feature and actual model training. The notebook within the folder contains the training of several different models. However, the model with the best AUC was selected also considering the most reasonable train test split that avoids data leakage.

2 - streamlit_web_app: Contains the Initial build of the application using StreamLit and having the best found model.

3 - Reports: Contains the final deck presented to the jury. 
