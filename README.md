<<<<<<< HEAD
# Is AI Changing US Employment?

Final project for CS 7830 Machine Learning (Fall 2026), Wright State University. Kamala Joshi and Diwakar Dahal.

## Project

ChatGPT was launched in November 2022. We investigate if there have been differences in the labor market trends of US occupations that utilize AI the most since its launch, and categorize occupations into types of jobs for the AI era.

* Regression: Is there a relationship between AI exposure in the occupation and the change in employment and wages from May 2022 to May 2025, controlling for wages, education, and skills? The placebo test is carried out on the period May 2016 to May 2019 before ChatGPT.
* Classification: The same question as grown versus shrank.
* Clustering: Cluster occupations based on exposure, wages, education, and skills.

Each row in our data is one occupation (2018 SOC code), not one person.

## Folder Structure

```
data/raw/               downloads 
data/processed/         merged table made by src/build_dataset.py
src/                    build_dataset.py
notebooks/              analysis notebooks, numbered in order
results/                figures and tables for the report
reports/                final report
```
=======
# machine_learning
>>>>>>> 438045c71691c5d381cb26500ed3d81829771472
