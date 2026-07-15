# Machine Learning Framework for Forward Kinematics Prediction of Concentric Tube Continuum Robots

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue)

![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)

![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

![License](https://img.shields.io/badge/License-MIT-green)

![Status](https://img.shields.io/badge/Status-Research-success)

</p>

<p align="center">

Machine Learning • Medical Robotics • Continuum Robotics • Regression • Ensemble Learning • Reproducible Research

</p>

---

> **A complete end-to-end machine learning framework for accurate and reproducible forward kinematics prediction of Concentric Tube Continuum Robots (CTCRs), integrating feature engineering, supervised regression, deep neural networks, ensemble learning, and systematic experimental evaluation.**

## Abstract

Forward kinematics is a fundamental problem in robotics, enabling the estimation of a robot's end-effector pose from its joint configuration. While this task is well understood for rigid manipulators, it remains significantly more challenging for **Concentric Tube Continuum Robots (CTCRs)** due to the complex nonlinear interactions between multiple pre-curved elastic tubes. Accurate forward kinematics prediction is essential for robot control, motion planning, and image-guided navigation, particularly in minimally invasive surgical applications where precision and computational efficiency are critical.

Traditional approaches rely on continuum mechanics models, such as Cosserat Rod Theory, to describe the elastic behavior of concentric tubes. Although these models provide physically consistent solutions, they often require iterative numerical optimization and substantial computational resources, making their direct integration into real-time robotic systems challenging.

This repository presents a **reproducible end-to-end machine learning framework** for data-driven forward kinematics prediction of CTCRs. Instead of replacing the underlying physical principles, the proposed framework investigates how supervised machine learning can efficiently approximate the nonlinear mapping between robot joint configurations and end-effector pose using data generated from validated physical simulations.

The experimental pipeline covers every stage of the machine learning workflow, including dataset validation, exploratory data analysis, geometry-inspired feature engineering, feature scaling, supervised regression, deep neural network modeling, ensemble learning, quantitative evaluation, and benchmarking against previously published work. Multiple regression algorithms are systematically compared, including **Linear Regression**, **Extra Trees Regressor**, **Multi-Layer Perceptron (MLP)**, and an **Enhanced Stacking Ensemble** based on Ridge meta-learning.

Special attention is given to feature representation and experimental reproducibility. The proposed methodology incorporates geometry-inspired feature engineering, Gaussian data augmentation, standardized preprocessing, controlled random initialization, early stopping, and reproducible evaluation procedures to improve predictive performance while ensuring fair model comparison.

The resulting framework demonstrates that carefully designed machine learning models, combined with domain-informed feature engineering and ensemble learning strategies, can achieve highly accurate forward kinematics prediction while providing a reproducible experimental foundation for future research in continuum robotics and intelligent robotic systems.

## Table of Contents

- [Abstract](#abstract)
- [Why This Project?](#why-this-project)
- [Research Motivation](#research-motivation)
- [Problem Statement](#problem-statement)
- [Research Objectives](#research-objectives)
- [Project Highlights](#project-highlights)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models Evaluated](#models-evaluated)
- [Proposed Ensemble Framework](#proposed-ensemble-framework)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Benchmark Comparison](#benchmark-comparison)
- [Discussion](#discussion)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Applications](#applications)
- [Future Work](#future-work)
- [Repository Usage](#repository-usage)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

# Why This Project?

Concentric Tube Continuum Robots (CTCRs) represent a new generation of highly flexible robotic manipulators capable of navigating confined and anatomically complex environments. Their unique mechanical structure enables minimally invasive surgical procedures that are difficult or impossible to achieve using conventional rigid robots.

However, this flexibility introduces significant challenges for robot modeling. Unlike traditional serial manipulators, the relationship between joint variables and end-effector pose is governed by complex nonlinear elastic interactions between multiple pre-curved concentric tubes. As a result, accurate forward kinematics estimation remains a computationally demanding problem.

Machine learning has recently emerged as a promising alternative for approximating complex nonlinear systems. Nevertheless, developing reliable data-driven models for continuum robotics requires more than selecting powerful algorithms. The quality of feature representation, experimental design, reproducibility, and model evaluation all play a critical role in achieving meaningful and trustworthy results.

This repository was developed to investigate whether a carefully designed machine learning pipeline can accurately learn CTCR forward kinematics while following rigorous experimental and reproducibility practices. Rather than focusing solely on predictive performance, the project emphasizes transparency, systematic evaluation, and reproducible research, providing a complete experimental framework that can serve as a foundation for future work in intelligent continuum robotics.

# Research Motivation

Forward kinematics is one of the core computational tasks in robotics, directly supporting robot control, trajectory planning, navigation, and interaction with the environment. For continuum robots, obtaining accurate forward kinematics predictions remains considerably more challenging than for conventional rigid manipulators because of their continuously deformable structure.

Traditional continuum mechanics models accurately capture these physical behaviors but often rely on iterative numerical solvers, limiting their computational efficiency in applications requiring rapid inference.

Recent advances in machine learning provide an opportunity to approximate highly nonlinear mappings directly from data. However, successfully applying machine learning to continuum robotics requires rigorous experimental methodology, appropriate feature representation, and reproducible evaluation rather than relying solely on increasingly complex models.

This project explores how data-driven regression techniques, combined with geometry-inspired feature engineering and ensemble learning, can provide an efficient approximation of CTCR forward kinematics while maintaining strong predictive performance.

# Problem Statement

Predicting the pose of a Concentric Tube Continuum Robot from its joint configuration is a highly nonlinear regression problem involving coupled elastic interactions and multiple degrees of freedom.

The primary challenge addressed in this project can be summarized as follows:

> **Can a reproducible machine learning framework accurately approximate the forward kinematics of Concentric Tube Continuum Robots while maintaining competitive prediction accuracy and computational efficiency?**

To answer this question, the project systematically investigates the influence of feature engineering, regression algorithms, neural networks, ensemble learning strategies, and evaluation methodologies on predictive performance.

# Research Objectives

The project was designed around five primary research objectives:

1. Develop a complete and reproducible machine learning pipeline for CTCR forward kinematics prediction.

2. Investigate the impact of geometry-inspired feature engineering on nonlinear regression performance.

3. Compare the predictive capabilities of classical machine learning algorithms, tree-based ensembles, and deep neural networks.

4. Design and evaluate an enhanced stacking ensemble architecture capable of improving robustness and prediction accuracy.

5. Benchmark the proposed framework against previously published CTCR forward kinematics research using standardized evaluation metrics.

# Project Highlights

This repository provides a complete and reproducible machine learning framework for forward kinematics prediction of **Concentric Tube Continuum Robots (CTCRs)**. Beyond developing predictive models, the project emphasizes rigorous experimental methodology, transparent evaluation, and reproducible research practices.

## Key Highlights

- **End-to-End Machine Learning Pipeline**  
  Covers the complete workflow from dataset validation and exploratory data analysis to feature engineering, model development, performance evaluation, and benchmarking.

- **Geometry-Inspired Feature Engineering**  
  Introduces engineered features derived from CTCR geometric and kinematic properties to improve nonlinear regression performance and feature interpretability.

- **Comprehensive Model Benchmarking**  
  Systematically compares multiple regression approaches, including Linear Regression, Extra Trees Regressor, Multi-Layer Perceptron (MLP), and an Enhanced Stacking Ensemble.

- **Enhanced Ensemble Learning Framework**  
  Combines multiple independently trained neural networks using Ridge Regression meta-learning to improve predictive robustness and generalization.

- **Multi-Output Pose Prediction**  
  Simultaneously predicts both the Cartesian position and orientation of the robot end-effector within a unified regression framework.

- **Rigorous Experimental Evaluation**  
  Models are assessed using consistent preprocessing, identical train-test partitions, standardized evaluation metrics, and quantitative error analysis.

- **Benchmark-Oriented Validation**  
  Experimental results are compared with previously published CTCR forward kinematics research to provide contextual performance assessment.

- **Fully Reproducible Research**  
  Every experiment can be reproduced through documented preprocessing steps, fixed random seeds, standardized data preparation, and a fully documented Jupyter Notebook.

- **Research-Oriented Repository**  
  Designed following modern scientific software engineering practices to facilitate reproducibility, transparency, and future research extensions.

  # Scientific Contributions

This project contributes to the growing field of data-driven continuum robotics through the design, implementation, and evaluation of a complete machine learning framework for forward kinematics prediction.

The main contributions of this work include:

### 1. End-to-End Machine Learning Framework

A complete experimental framework was developed, covering every stage of the machine learning lifecycle, from raw dataset validation to final model evaluation and benchmark comparison.

---

### 2. Geometry-Inspired Feature Engineering

Instead of relying exclusively on raw joint variables, the framework incorporates engineered features derived from the geometric characteristics of CTCR motion. These transformations simplify the nonlinear regression problem while improving model interpretability and predictive capability.

---

### 3. Comprehensive Regression Benchmark

Several regression paradigms are systematically evaluated under identical experimental conditions, enabling a fair comparison between linear models, tree-based ensembles, neural networks, and stacked architectures.

---

### 4. Enhanced Stacking Ensemble

A stacking ensemble composed of multiple independently trained neural networks and a Ridge Regression meta-learner was developed to improve prediction robustness while reducing estimation error.

---

### 5. Reproducible Experimental Pipeline

The complete workflow was designed to maximize reproducibility through deterministic preprocessing, controlled random initialization, documented feature engineering, standardized evaluation metrics, and version-controlled experimentation.

---

### 6. Benchmark-Based Performance Assessment

The proposed framework is evaluated against previously published CTCR forward kinematics results, providing a quantitative reference for assessing predictive performance.

# Repository Structure

```text
CTCR-Forward-Kinematics/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── CITATION.cff
│
├── notebook/
│   └── ctcr_forward_kinematics.ipynb
│
├── data/
│   └── README.md
│
├── assets/
│   ├── banner.png
│   ├── pipeline.png
│   ├── workspace.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── stacking_framework.png
│
└── results/
    ├── figures/
    └── tables/
```

The repository is organized to provide a clear, reproducible, and maintainable research workflow.

| Directory / File | Description |
|------------------|-------------|
| **README.md** | Project overview, methodology, experimental workflow, and documentation. |
| **LICENSE** | MIT License governing the use and distribution of this repository. |
| **requirements.txt** | Python dependencies required to reproduce the complete experimental pipeline. |
| **.gitignore** | Specifies files and directories excluded from version control. |
| **.gitattributes** | Defines repository file attributes and ensures consistent line endings across operating systems. |
| **CITATION.cff** | Citation metadata enabling GitHub's **"Cite this repository"** feature. |
| **notebook/** | Contains the complete Jupyter Notebook implementing the end-to-end machine learning workflow, including data validation, exploratory analysis, feature engineering, model development, ensemble learning, and performance evaluation. |
| **data/** | Dataset documentation. The original CTCR dataset should be obtained from its official source. This repository does not redistribute the dataset. |
| **assets/** | Images and visual resources used throughout the documentation, including workflow diagrams, workspace visualizations, model comparison figures, and project illustrations. |
| **results/** | Generated outputs from the experimental pipeline, including evaluation figures, comparison tables, and performance summaries. |

The repository follows a modular organization to facilitate reproducibility, readability, and future extensions. Although the current implementation is centered around a Jupyter Notebook, the project structure has been designed to support future migration toward a fully modular research codebase.

> **Repository Design Philosophy**
>
> This repository prioritizes **clarity**, **reproducibility**, and **scientific transparency**. Every stage of the machine learning workflow—from data exploration to model evaluation—is documented and organized to facilitate understanding, replication, and future research extensions.

# Dataset

The experiments are conducted using a dataset generated from validated forward kinematics simulations of Concentric Tube Continuum Robots.

Each observation represents a unique robot configuration defined by its joint parameters and the corresponding end-effector pose.

The dataset contains:

- Joint configuration variables
- Tube insertion parameters
- Tube rotation parameters
- Cartesian tip position
- End-effector orientation
- Derived geometric representations used during feature engineering

Before model development, the dataset undergoes several validation procedures, including:

- Integrity verification
- Missing value analysis
- Duplicate detection
- Statistical consistency analysis
- Workspace visualization
- Correlation analysis

These validation steps ensure the reliability of the experimental results while supporting full reproducibility of the machine learning pipeline.

# Methodology

The proposed framework follows a structured, end-to-end machine learning workflow specifically designed for forward kinematics prediction of Concentric Tube Continuum Robots (CTCRs). Rather than focusing solely on model development, the methodology emphasizes data quality, reproducibility, systematic experimentation, and rigorous comparative evaluation.

The complete workflow is organized into successive stages that progressively transform raw simulation data into accurate multi-output regression models capable of predicting both the robot tip position and orientation.

The experimental design ensures that every model is trained, validated, and evaluated under identical conditions, allowing a fair and reproducible comparison between different learning algorithms.

# Machine Learning Pipeline

The complete experimental workflow is illustrated below.

```text
                    CTCR Forward Kinematics Dataset
                                  │
                                  ▼
                     Dataset Integrity Verification
                                  │
                                  ▼
                  Missing Values & Duplicate Analysis
                                  │
                                  ▼
                   Exploratory Data Analysis (EDA)
                                  │
                                  ▼
                    Statistical & Workspace Analysis
                                  │
                                  ▼
                 Geometry-Inspired Feature Engineering
                                  │
                                  ▼
                     Feature Scaling & Normalization
                                  │
                                  ▼
                      Train / Test Data Partition
                                  │
                                  ▼
          ┌──────────────────────────────────────────────┐
          │                                              │
          ▼                                              ▼
 Linear Regression                              Extra Trees Regressor
          │                                              │
          └──────────────────────────────────────────────┘
                                  │
                                  ▼
                     Multi-Layer Perceptron (MLP)
                                  │
                                  ▼
                   Enhanced Stacking Ensemble Learning
                                  │
                                  ▼
             Multi-Output Position & Orientation Prediction
                                  │
                                  ▼
                    Performance Evaluation & Benchmarking
                                  │
                                  ▼
                   Comparative Analysis & Discussion
```

# Experimental Workflow

The complete notebook implements a reproducible experimental workflow composed of the following stages:

### 1. Dataset Validation

The raw dataset is first inspected to verify its integrity before any machine learning experiment is conducted. This stage includes dataset inspection, data type verification, duplicate detection, missing value analysis, and consistency checks.

---

### 2. Exploratory Data Analysis

A comprehensive exploratory analysis is performed to better understand the statistical properties of the dataset. Distribution plots, descriptive statistics, correlation matrices, and workspace visualizations are generated to identify potential relationships between variables.

---

### 3. Geometry-Inspired Feature Engineering

Domain knowledge from continuum robotics is incorporated through feature engineering. Additional variables are generated from the original joint parameters to provide more informative representations of robot configurations while preserving geometric meaning.

---

### 4. Data Preprocessing

Engineered features are standardized before model development to ensure comparable feature scales and stable optimization during neural network training.

---

### 5. Model Development

Multiple regression models are trained using identical experimental conditions. Baseline algorithms provide reference performance while increasingly sophisticated models evaluate the contribution of nonlinear learning and ensemble methods.

---

### 6. Ensemble Learning

The strongest individual neural networks are combined using a stacking strategy based on Ridge Regression meta-learning. This ensemble aims to improve prediction robustness while reducing variance.

---

### 7. Performance Evaluation

Model performance is assessed using multiple complementary regression metrics together with position and orientation prediction errors. All experiments use consistent evaluation procedures to ensure fair comparison.

---

### 8. Benchmark Comparison

Finally, the proposed framework is compared against previously published CTCR forward kinematics results to place the experimental findings within the context of existing literature.

# Geometry-Inspired Feature Engineering

Feature engineering represents one of the central components of the proposed framework.

Rather than relying exclusively on the original robot joint variables, the methodology introduces additional features derived from the geometric and kinematic properties of Concentric Tube Continuum Robots.

This approach aims to provide learning algorithms with representations that better capture the nonlinear relationships governing robot motion while preserving meaningful physical interpretation.

The engineered feature space includes transformations such as:

- Trigonometric encoding of rotational variables
- Normalized insertion variables
- Nonlinear interaction terms
- Geometry-inspired feature representations

These transformations simplify the learning problem, improve feature expressiveness, and enhance the predictive capability of nonlinear regression models.

The impact of feature engineering is systematically evaluated throughout the experimental study to quantify its contribution to overall model performance.

# Models Evaluated

To investigate the suitability of different regression paradigms for CTCR forward kinematics prediction, several machine learning models are evaluated under identical experimental conditions.

| Model | Category | Role |
|--------|----------|------|
| Linear Regression | Linear Model | Baseline reference |
| Extra Trees Regressor | Ensemble Learning | Tree-based nonlinear baseline |
| Multi-Layer Perceptron | Deep Learning | Neural network baseline |
| Enhanced Stacking Ensemble | Ensemble Learning | Proposed framework |

The comparison allows the influence of model complexity, nonlinear approximation capability, and ensemble learning strategies to be systematically analyzed.

# Proposed Ensemble Framework

The final predictive framework is based on an enhanced stacking ensemble architecture designed to combine the strengths of multiple independently trained neural networks.

Instead of relying on a single predictive model, the framework aggregates diverse MLP regressors through a Ridge Regression meta-learner capable of learning optimal model combinations.

Additional strategies integrated into the framework include:

- Gaussian data augmentation
- Early stopping
- Controlled random initialization
- Ridge regularization
- Ensemble diversity optimization

Together, these techniques improve prediction stability, reduce estimation variance, and enhance generalization performance while maintaining computational efficiency.

The resulting framework serves as the primary contribution of this project and represents the highest-performing model evaluated throughout the experimental study.

# Experimental Results

The proposed framework was evaluated through a comprehensive experimental study designed to assess the impact of feature engineering, regression algorithms, neural network architectures, and ensemble learning on forward kinematics prediction.

To ensure a fair comparison, every model was trained using the same preprocessing pipeline, identical train-test partition, standardized feature scaling strategy, and consistent evaluation metrics. This experimental protocol minimizes potential sources of bias and allows the observed performance differences to be attributed primarily to the learning algorithms themselves.

The evaluation focuses on the prediction of both the Cartesian position and orientation of the robot end-effector, providing a comprehensive assessment of multi-output regression performance.

## Performance Summary

| Model | Position Error (mm) | Orientation Error | Remarks |
|:------|--------------------:|------------------:|:--------|
| Linear Regression | 5.09 | 0.072 rad | Linear baseline |
| Extra Trees Regressor | 8.12 | 0.094 rad | Tree-based baseline |
| Multi-Layer Perceptron | 1.07 | 0.019 rad | Strong nonlinear learner |
| **Enhanced Stacking Ensemble** | **0.688** | **0.0166 rad (0.95°)** | **Best overall performance** |

The Enhanced Stacking Ensemble consistently achieved the lowest prediction errors across both position and orientation estimation, demonstrating the effectiveness of combining multiple neural network models through a regularized meta-learning strategy.

# Benchmark Comparison

To provide a meaningful point of reference, the proposed framework was compared with previously published work on CTCR forward kinematics prediction.

| Method | Position Error | Orientation Error |
|:-------|---------------:|------------------:|
| Grassmann et al. (2022) | 0.74 mm | 6.49° |
| **Proposed Framework** | **0.688 mm** | **0.95°** |

### Relative Improvement

- **7.03% lower mean position error**
- **85.34% lower mean orientation error**

These results demonstrate that the proposed framework achieves competitive predictive accuracy while maintaining a reproducible and systematic experimental methodology.

> **Important Note**
>
> Direct comparisons with previously published studies should be interpreted with appropriate caution, as differences in datasets, simulation environments, feature representations, and evaluation protocols may influence the reported performance. The benchmark is therefore intended to provide contextual reference rather than establish definitive superiority.

# Discussion

The experimental results provide several important observations regarding the application of machine learning to CTCR forward kinematics prediction.

First, the comparison between baseline regression models and neural networks highlights the highly nonlinear nature of the underlying kinematic mapping. Classical linear models provide limited predictive capability, while nonlinear learning approaches significantly improve estimation accuracy.

Second, the study demonstrates that feature representation plays a central role in predictive performance. Incorporating geometry-inspired engineered features enables the learning algorithms to capture complex relationships more effectively than relying solely on raw joint variables.

Finally, ensemble learning further improves robustness by reducing the dependence on any individual neural network. The Enhanced Stacking Ensemble consistently provides the strongest overall performance, illustrating the benefits of combining diverse predictive models through a regularized meta-learning strategy.

Beyond predictive accuracy, the project emphasizes reproducibility and experimental transparency, ensuring that every stage of the machine learning workflow can be independently reproduced and extended in future research.

# Key Findings

The experimental study leads to several important conclusions.

### Geometry-Inspired Feature Engineering Improves Learning

Transforming the original joint variables into more informative geometric representations significantly facilitates nonlinear regression and contributes to improved predictive performance.

---

### Neural Networks Capture CTCR Dynamics More Effectively

The Multi-Layer Perceptron substantially outperforms both Linear Regression and Extra Trees Regressor, confirming the suitability of deep learning for modeling complex continuum robot kinematics.

---

### Ensemble Learning Provides Additional Robustness

Combining multiple independently trained neural networks through stacking further reduces prediction errors while improving model stability and generalization.

---

### Experimental Reproducibility Matters

The use of standardized preprocessing, deterministic train-test partitions, fixed random seeds, and documented experimental procedures ensures that all reported results can be reliably reproduced.

# Limitations

Although the proposed framework demonstrates strong predictive performance, several limitations should be acknowledged.

- The models are trained and evaluated using simulation-generated data rather than measurements collected from a physical robotic platform.
- Generalization to unseen robot designs or different CTCR configurations has not been investigated.
- Hyperparameter optimization was intentionally limited to maintain computational efficiency.
- The current framework focuses exclusively on forward kinematics and does not address inverse kinematics or dynamic modeling.
- Experimental validation on real robotic hardware remains an important direction for future work.

Recognizing these limitations helps define realistic expectations while identifying promising opportunities for future research.

# Potential Applications

The proposed framework can serve as a foundation for several applications in robotics and intelligent systems, including:

- Robot motion planning
- Real-time forward kinematics approximation
- Robot navigation
- Surgical robotics
- Medical robotics research
- Robot simulation
- Intelligent robot control
- Data-driven robotic modeling
- Machine learning for continuum robotics
- Educational and research purposes

# Repository Usage

This repository is organized to provide a transparent and reproducible implementation of the complete experimental workflow.

The Jupyter Notebook contains every stage of the research pipeline, including:

- Dataset validation
- Exploratory data analysis
- Feature engineering
- Data preprocessing
- Model training
- Ensemble learning
- Performance evaluation
- Benchmark comparison
- Result visualization

The notebook is intended to be executed sequentially from top to bottom to reproduce all reported experiments.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/i3012/CTCR-Forward-Kinematics.git
cd <repository>
```

## Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Quick Start

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/ctcr_forward_kinematics.ipynb
```

Execute all notebook cells sequentially to reproduce the complete experimental workflow.

---

# Reproducibility

Reproducibility was a primary design objective throughout this project.

The experimental protocol includes:

- Fixed random seeds
- Deterministic train-test split
- Standardized preprocessing
- Documented feature engineering
- Consistent evaluation metrics
- Version-controlled source code
- Fully executable Jupyter Notebook

These practices ensure that every reported experiment can be independently reproduced under the same computational environment.

---

# Technologies

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python 3.11+ |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Data Visualization | Matplotlib, Seaborn |
| Development Environment | Jupyter Notebook |

---

# Future Work

Several promising research directions emerge from this work.

### Physics-Informed Machine Learning

Integrate physical constraints directly into the learning process through Physics-Informed Neural Networks (PINNs).

---

### Graph Neural Networks

Investigate graph-based representations to better capture structural relationships between concentric tubes.

---

### Transformer-Based Regression

Explore attention-based architectures for modeling complex nonlinear robotic systems.

---

### Bayesian Machine Learning

Estimate predictive uncertainty for safety-critical robotic applications.

---

### Sim-to-Real Transfer

Evaluate domain adaptation techniques to bridge the gap between simulation-generated datasets and real robotic systems.

---

### Real Robot Validation

Validate the proposed framework using experimental measurements collected from physical CTCR platforms.

---

### Online Learning

Investigate adaptive learning strategies capable of continuously updating the predictive model during robot operation.

---

# Potential Applications

Although developed as a research project, the proposed framework has potential applications in several domains.

- Robot motion planning
- Forward kinematics approximation
- Intelligent robot control
- Surgical robotics
- Medical robotics
- Robot simulation
- Machine learning for robotics
- Continuum robotics research
- Autonomous robotic systems
- AI-assisted robotic navigation

---

# Citation

If you find this repository useful for your research, please consider citing it.

```bibtex
@misc{bouchakour2026_ctcr,
  author       = {Imene Bouchakour},
  title        = {A Reproducible Machine Learning Framework for Forward Kinematics Prediction of Concentric Tube Continuum Robots},
  year         = {2026},
  note         = {GitHub Repository},
  url          = {https://github.com/<i3012>/<CTCR-Forward-Kinematics>}
}
```

---

# License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this work in accordance with the terms of the license.

See the `LICENSE` file for additional information.

---

# Acknowledgements

This work builds upon previous research in continuum robotics and machine learning for robotic modeling.

Special acknowledgment is given to the authors of the CTCR benchmark dataset and the scientific community whose research has contributed to the advancement of continuum robot kinematics.

The published benchmark used for comparison is fully credited within the accompanying notebook and supporting documentation.

---

# Contact

If you have questions, suggestions, or would like to discuss this work, feel free to open an issue or contact the repository author through GitHub.

Contributions, discussions, and constructive feedback are always welcome.

---

## Star the Repository

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps increase the visibility of the project and supports future research and development.