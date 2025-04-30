# Trustworthy Reviewer Scoring System for Detecting Fake Reviews

![System Diagram](./assests/overall.png)


## 📑 Overview

Fake reviews pose a significant challenge to online trust and transparency. Malicious actors can manipulate customer reviews to mislead consumers and promote products or services unfairly. Our research project aims to develop a robust and efficient system for detecting fake reviews using natural language processing (NLP) techniques and behavioral analysis.

The project addresses the limitations of traditional methods that rely solely on textual analysis by proposing a multi-pronged approach that combines:

- **Textual Analysis**: Advanced NLP techniques to detect linguistic patterns in fake reviews
- **Behavioral Feature Analysis**: Analysis of reviewer activity and patterns 
- **Sentiment-Rating Mismatch Detection**: Identifying inconsistencies between text sentiment and numeric ratings
- **Readability-Based Validation**: Detection of robotic or templated writing styles
- **Explainable AI**: Providing transparent reasoning for review classification

This combined approach is expected to improve the accuracy and robustness of fake review detection compared to traditional methods and contribute to building a more trustworthy online environment for consumers and businesses alike.

## 🏆 Team Registry

<div align="center">

| **Name with initials** | **Registration Number** | **Contact Phone Number** | **Email** | **Badge** |
|:----------------------:|:----------------------:|:------------------------:|:---------:|:---------:|
| Jayawardhana R.A.D.G.S | IT20137946 | 0712696233 | it20137946@my.sliit.lk | ![Badge](https://img.shields.io/badge/Leader-★★★-gold?style=for-the-badge&logo=github&logoColor=white) |
| Dissanayaka S.D | IT21827662 | 0774487666 | it21827662@my.sliit.lk | ![Badge](https://img.shields.io/badge/Member-✓-22c55e?style=for-the-badge&logo=github&logoColor=white) |
| Thathsarani H. A. N. N | IT21237522 | 0773747615 | it21237522@my.sliit.lk | ![Badge](https://img.shields.io/badge/Member-✓-22c55e?style=for-the-badge&logo=github&logoColor=white) |

</div>

## 🔍 Project Components

### 1. Textual Content Analysis Using NLP (Dissanayaka S.D)

This component aims to develop a reliable and efficient textual analysis system for detecting fake reviews by leveraging advanced Natural Language Processing (NLP) techniques. The system:

- Analyzes linguistic patterns, semantic meaning, and sentiment inconsistencies
- Extracts rich textual features, including TF-IDF scores, word embeddings, and readability metrics
- Uses transformer-based models for more accurate classification
- Detects sentiment-rating mismatches to identify potential fake reviews
- Employs readability metrics to identify robotic or templated writing styles

<div align="center">
  <img src="https://github.com/username/repo-name/raw/main/images/textual-analysis-flow.png" alt="Textual Analysis Flow" width="600">
</div>

### 2. Fake Reviewer Detection System (Jayawardhana R.A.D.G.S)

This component focuses on detecting fake reviewers by analyzing behavioral patterns and metadata. The system:

- Performs behavioral feature analysis to identify suspicious activity patterns
- Analyzes metadata to detect anomalies in user behavior
- Develops a trust score based on review history and account activity
- Implements anomaly detection algorithms to identify outliers
- Creates a trustworthy reviewer scoring system to rank reviewers by reliability

<div align="center">
  <img src="https://github.com/username/repo-name/raw/main/images/reviewer-detection-flow.png" alt="Reviewer Detection Flow" width="600">
</div>

### 3. Fusion-Based Hybrid Validation System (Thathsarani H.A.N.N)

This component combines textual and behavioral features into a unified system for more accurate fake review detection. The system:

- Integrates textual and behavioral features into a unified feature space
- Normalizes and standardizes features to ensure comparability
- Experiments with various machine learning algorithms for optimal performance
- Utilizes explainable AI techniques like SHAP to interpret model decisions
- Provides human-understandable explanations for the model's output

<div align="center">
  <img src="https://github.com/username/repo-name/raw/main/images/hybrid-system-flow.png" alt="Hybrid System Flow" width="600">
</div>

## 🛠️ Technology Stack

- **Programming Languages**: Python, R
- **NLP Libraries**: NLTK, spaCy, Transformers
- **Machine Learning Frameworks**: TensorFlow, PyTorch, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Explainable AI**: SHAP, LIME
- **Development Tools**: Jupyter Notebook, VS Code, Git

## 📊 Datasets

- **Amazon Fine Food Reviews Dataset**: Contains over 500,000 food reviews from Amazon
- **ClickHouse**: Custom datasets focusing on product reviews and customer sentiment analysis

## 📑 Research Comparison

### Textual Content Analysis

| Aspect | Previous Research | Our Approach |
|:-------|:-----------------|:-------------|
| Sentiment Analysis | Limited use in existing systems | Advanced sentiment analysis with context |
| Readability Metrics | Rarely implemented | Core feature for identifying templated reviews |
| Transformer Models | Limited adoption | State-of-the-art transformer architecture |
| Explainable AI | Not implemented | Transparent explanations for model decisions |

### Fake Reviewer Detection

| Feature | Existing Research | Our System |
|:--------|:-----------------|:-----------|
| Fake review detection | Present in some research | Enhanced with behavioral analysis |
| Rating scoring system | Not implemented | Core component for trustworthiness |
| Fake reviewer identification | Limited implementation | Comprehensive detection system |

## 🎯 Objectives

### Main Objective
- Develop a robust and accurate system for detecting fake online reviews

### Sub-Objectives
- Develop an efficient textual content analysis model using advanced NLP techniques
- Create a behavioral analysis system to identify suspicious reviewer patterns
- Build a hybrid validation system combining textual and behavioral features
- Implement explainable AI techniques for transparent decision-making

## 🌐 Target Audience and Applications

- **E-commerce Platforms**: Amazon, eBay, Alibaba
- **Review Websites**: Yelp, TripAdvisor, Google Reviews
- **Product Manufacturers**: Quality assurance and brand protection
- **Consumers**: Browser extensions for fake review detection

## 📝 License

[MIT License](LICENSE)

## 📚 References

1. Moawesh, H., & Xu, L. (2020). Fake Reviews Detection: A Survey. In 2020 2nd International Conference on Artificial Intelligence and Computer Science (AICS) (pp. 1-6). IEEE.
2. Ott, M., Choi, Y., Cardie, C., Hancock, J. T., & Turney, P. D. (2011). Finding deceptive opinion spam by any stretch of the imagination. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.
3. Xu, Y., Li, Y., Tian, Y., & Liu, Y. (2022). Fake Review Detection Model Based on Comment Content and Review Behavior. Sensors, 13(21), 4322.
4. Carcillo, S., & Gómez-Hernández, J. A. (2020). Fake Review Detection Using Transformer-Based Neural Networks. arXiv preprint arXiv:2003.00807.
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems (pp. 4765-4774).
