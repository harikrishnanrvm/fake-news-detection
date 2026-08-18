# Fake News Detection using Deep Learning

*Working draft of the final dissertation. Assembled part by part, per the University template
(`docs/BCA Major_Project Final Report template-1.docx`). Content ready to paste into Microsoft
Word — final page numbers, table of contents update, and formatting to be done in Word itself.*

---

## PART 1 — Front Matter

*(Guide and Coordinator names left as bracketed placeholders — fill these in before printing.)*

---

### COVER PAGE

<div align="center">

**FAKE NEWS DETECTION USING DEEP LEARNING**

A Dissertation
Submitted by

**HARIKRISHNAN R**
(AA.SCU3CSC2107088)

in partial fulfilment of the requirements for the award of the degree of

**BACHELOR OF COMPUTER APPLICATIONS**

July 2026

</div>

---

### BONAFIDE CERTIFICATE

This is to certify that this dissertation titled **"Fake News Detection using Deep Learning,"** submitted in partial fulfilment of the requirements for the award of the Degree of Bachelor of Computer Applications, by **Harikrishnan R** (AA.SCU3CSC2107088), is a bona fide record of the work carried out by him under my supervision during the academic term from April 2026 to August 2026, and that it has been submitted, to the best of my knowledge, in part or in full, for the award of any other degree or diploma.

&nbsp;

&nbsp;

**[Guide Name]** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **[Coordinator Name]**
Project Guide &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Coordinator

&nbsp;

**Reviewer**

&nbsp;

**Date:** _______________

---

### DECLARATION

I do hereby declare that this dissertation titled **"Fake News Detection using Deep Learning,"** submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Computer Applications, is a true record of work carried out by me and that all information contained herein, which does not arise directly from my own work, has been properly acknowledged and cited using acceptable standards. Further, I declare that the contents of this dissertation have not been submitted, in part or in full, for the award of any other degree or diploma.

&nbsp;

&nbsp;

**Date:** _______________ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **Harikrishnan R**
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(Signature of the student)

---

### ACKNOWLEDGEMENT

I would like to express my sincere gratitude to **[Guide Name]**, my Project Guide, for the guidance and feedback provided throughout the course of this project, from the initial dataset exploration to the final deployment of the application.

I am also thankful to **[Coordinator Name]**, Program Coordinator, for the support and coordination extended during the Major Project.

I would like to thank the faculty of the BCA program at Amrita for the knowledge and foundation that made this project possible.

I am grateful to my family for their constant support and encouragement throughout this project.

Finally, I would like to acknowledge the creators of the "Fake and Real News Dataset" (Clément Bisaillon, Kaggle), which formed the foundation of this project's dataset.

**Harikrishnan R**

---

### ABSTRACT

Fake news spreads rapidly through online news websites and social media, making it difficult for readers to identify reliable information. This project develops a system to automatically classify news articles as fake or real by comparing a traditional machine learning model with a deep learning model.

The system uses the Fake and Real News Dataset from Kaggle, consisting of the Fake.csv and True.csv files with approximately 45,000 news articles. During exploratory data analysis, it was found that the subject column and the Reuters dateline present in many genuine articles could reveal the class label without analysing the actual news content. These features were removed, and the text was cleaned and preprocessed before training the models.

Two models were developed and evaluated using the same training, validation, and test datasets. The first model used TF-IDF with Logistic Regression, while the second used an LSTM neural network. The Logistic Regression model achieved 98.24 percent accuracy, whereas the LSTM achieved 97.20 percent accuracy on the test dataset. McNemar's Test showed that the difference was statistically significant, although the practical improvement was small. An additional ablation study also showed that the label leakage was caused by multiple related features rather than a single keyword.

The trained LSTM model was deployed using a FastAPI backend with a Bootstrap-based web interface that allows users to paste a news article and receive a prediction with a confidence score. The results show that a more complex deep learning model does not always perform better than a simpler machine learning approach and highlight the importance of careful data preprocessing and model evaluation.

---

### TABLE OF CONTENTS *(draft)*

| Section | Page |
|---|---|
| Bonafide Certificate | i |
| Declaration | ii |
| Acknowledgement | iii |
| Abstract | iv |
| List of Figures | v |
| List of Tables | vi |
| List of Abbreviations | vii |
| **1. Objectives** | 1 |
| **2. Scope** | 2 |
| **3. Introduction** | 3 |
| **4. Literature Review** | – |
| **5. System Analysis / Problem Definition** | – |
| **6. Methodology** | – |
| **7. Algorithms and Tools Used** | – |
| **8. System Design** | – |
| **9. Implementation Details** | – |
| **10. Testing and Results** | – |
| **11. Discussion** | – |
| **12. Conclusion** | – |
| **13. Future Work** | – |
| **14. References** | – |
| **15. Appendix** | – |

*(Page numbers to be filled in once the full document is assembled in Word.)*

#### List of Figures *(draft — will be finalized once Parts 4–6 assign figure numbers)*

Based on figures already generated under `report/figures/`: class distribution, label-leakage signal comparison, article length distributions, preprocessing funnel, baseline confusion matrix, baseline ROC curve, baseline top features, Reuters-ablation confusion matrix/ROC curve/top features, LSTM confusion matrix, LSTM ROC curve, LSTM training history, model accuracy comparison, model training time comparison, prediction agreement (error overlap).

#### List of Tables *(draft)*

Based on tables already generated under `report/tables/`: data dictionary, duplicate statistics, label leakage summary, preprocessing pipeline stage statistics, baseline classification report, Reuters-ablation comparison and classification report, LSTM classification report, model comparison table, model strengths/weaknesses, prediction agreement breakdown.

#### List of Abbreviations

LSTM, TF-IDF, NLP, ML, DL, RNN, API, JSON, HTML, CSS, UI, REST, ROC, AUC, F1, CSV, BCA

---

## PART 2

### 1. Objectives

The overall objective of this project is to design, implement, and evaluate a Fake News Detection System that classifies news articles as Fake or Real using both a traditional Machine Learning approach and a Deep Learning approach, and to compare the effectiveness of the two approaches in a fair and systematic manner.

The specific objectives of this project are:

1. To collect and study the Fake and Real News Dataset from Kaggle and understand its structure, class distribution, and overall characteristics, through detailed exploratory data analysis.
2. To identify and address label leakage in the dataset, particularly the subject column and the Reuters wire-service dateline, which were found to predict the class label without learning meaningful patterns from the news article itself.
3. To clean and preprocess the news articles by removing HTML tags, URLs, punctuation, and other unwanted elements, and by preparing separate versions of the text suitable for each model.
4. To implement a baseline Machine Learning model using TF-IDF vectorization and Logistic Regression, and to evaluate it using standard classification metrics.
5. To implement a Deep Learning model using an LSTM neural network for the same classification task, using the same training, validation, and test datasets for a fair comparison.
6. To statistically compare the performance of the two models using metrics such as Accuracy, Precision, Recall, F1-Score, and McNemar's Test, and to analyse the nature of their errors.
7. To deploy the trained LSTM model as a working web application, using a FastAPI backend and a Bootstrap-based frontend, so that the system can be demonstrated with real news articles.

### 2. Scope

This project is limited to text-based fake news detection using supervised Machine Learning and Deep Learning techniques. The system works only on English-language news articles and classifies each article into one of two categories, Fake or Real, based on its title and body text.

The project covers the complete pipeline required to build such a system: dataset collection and exploration, data cleaning and preprocessing, feature extraction, model training, model evaluation, statistical comparison of models, and deployment as a web application. It does not cover image-based misinformation, video analysis, social network propagation analysis, or real-time fact-checking against external sources. The system does not verify individual facts or claims made in an article. It only classifies the article based on patterns learned from the training data.

The technologies used in this project are Python, TensorFlow and Keras for the LSTM model, scikit-learn for the Logistic Regression baseline, NLTK for text preprocessing, pandas and NumPy for data handling, matplotlib for visualisation, FastAPI for the backend API, and Bootstrap for the web interface. The dataset used is the "Fake and Real News Dataset" published on Kaggle by Clément Bisaillon, consisting of the Fake.csv and True.csv files with a combined total of approximately 45,000 news articles, mainly covering political news from around 2016-2017.

This project is intended mainly as an educational demonstration. It is meant for students, researchers, and anyone who wants to learn how Natural Language Processing and Deep Learning can be applied to a real text classification problem. It can also be useful for machine learning enthusiasts who want to see a working comparison between a traditional model and a deep learning model on the same task, along with a simple web interface to try it out.

There are also some limitations that define the scope of this work. The models were trained and evaluated only on the specific Kaggle dataset described above, and their performance is tied to the writing style and sources represented in that dataset. The Reuters ablation study carried out in this project showed that part of what the models learn comes from stylistic differences between the two sources of articles, and not only from the truthfulness of the content itself. The system does not guarantee accurate results on news articles from different domains, languages, or time periods. It is also not intended to replace professional fact-checking. For demonstration purposes, the deployed web application uses only the LSTM model, even though the Logistic Regression baseline performed slightly better. This was a deliberate choice for this phase of the project and is explained further in later chapters.

### 3. Introduction

The way people consume news has changed considerably over the past decade. Traditional newspapers and television broadcasts have been supplemented, and in many cases replaced, by online news websites, blogs, and social media platforms. This shift has made information more accessible and immediate, but it has also made it much easier for false or misleading information, commonly referred to as fake news, to be created and spread. A single misleading post can be shared thousands of times within hours, often reaching more people than the original, accurate report it may be responding to.

Fake news is not a completely new problem, but the speed at which it spreads today is much higher than before. It can take the form of fabricated stories presented as genuine journalism, exaggerated or out-of-context reporting, or content designed specifically to provoke an emotional reaction and encourage sharing. This can have serious consequences. Fake news has been linked to influencing public opinion during elections, causing panic during public health emergencies, and reducing people's trust in genuine news sources. Since more people now depend on digital platforms for information, fake news has a bigger impact on individual decisions and on society as a whole than before.

Identifying fake news reliably is an important and difficult problem. Manual verification, carried out by journalists or dedicated fact-checking organisations, is thorough but slow. It depends on trained human judgement, checking multiple sources, and time, and none of these can really keep up with the huge amount of content published online every day. By the time a misleading article has been checked and flagged, it may already have been read and shared by a large number of people. This gap between how quickly misinformation spreads and how slowly it can be verified is one of the main reasons automated detection systems are needed.

Machine Learning and Deep Learning offer a practical way to approach this problem at scale. Text classification is the task of assigning a category to a piece of text, and it is a well-studied area of Natural Language Processing. Fake news detection can be treated as a binary text classification problem, where the goal is to look at the text of an article and predict whether it is Fake or Real. Traditional Machine Learning approaches usually represent text using numerical features, such as word frequencies, and then apply a classification algorithm such as Logistic Regression on these features. Deep Learning approaches, on the other hand, can learn their own representation of text directly from data. Architectures such as the Long Short-Term Memory (LSTM) network are designed to read text as an ordered sequence of words, which allows them to make use of word order and context in a way that simpler models cannot.

This project builds a Fake News Detection System that puts both of these approaches into practice and compares them directly, rather than assuming that one is better than the other. The system is built around the "Fake and Real News Dataset" from Kaggle, which contains around 45,000 news articles labelled as Fake or Real. Before building any model, the dataset was studied carefully. This revealed an important issue: certain features of the dataset, such as the subject column and a Reuters wire-service dateline present in most genuine articles, could predict the label almost perfectly on their own, without the model needing to understand the actual content of the article. This issue is commonly referred to as label leakage, and it was addressed directly by removing these features and cleaning the article text before training any model.

Two models were then developed using the cleaned dataset. The first is a baseline model that combines TF-IDF, a method of representing text as weighted word frequencies, with Logistic Regression, a simple and interpretable classification algorithm. The second is an LSTM neural network, which processes the article text as a sequence of words using a trainable word embedding layer followed by a recurrent LSTM layer and a final classification layer. Both models were trained and evaluated on exactly the same split of the data, so that their results could be compared fairly.

The purpose of comparing these two approaches is to investigate whether the additional complexity of a deep learning model leads to better performance than a simpler machine learning model on this dataset. Building a solid baseline first, and only then developing and evaluating the LSTM against it, makes it possible to answer this question with actual evidence from this specific dataset rather than assuming it beforehand. As later chapters of this dissertation show, the results are not entirely one-sided.

Finally, to show that the system can work as a usable application and not just as a research exercise, the trained LSTM model was deployed through a FastAPI backend. A simple Bootstrap-based web page allows a user to paste the text of a news article and receive a prediction, along with a confidence score, in return.

The rest of this dissertation looks at these ideas in more detail. The next chapter reviews existing work related to fake news detection, followed by chapters on the system analysis, methodology, algorithms, design, and implementation of the project, and finally the testing carried out, the results obtained, and a discussion of what these results mean.

---

## Chapter 4 – Literature Review

### 4.1 Introduction

Before building any system, it helps to look at how other people have already tried to solve the same problem. This is what a literature review is for. It shows what has already been tried, what worked, what did not work so well, and where there is still room to improve. For a project like this one, looking at earlier work in fake news detection also helps in choosing the right techniques and in understanding why certain methods, such as TF-IDF or LSTM, are commonly used for this kind of problem.

This chapter looks at how fake news detection has developed over time. It starts with the traditional Machine Learning methods that were used first, moves on to Deep Learning methods such as LSTM, and then briefly touches on newer approaches such as Transformers and BERT. After that, the chapter points out some gaps that are still present in existing work, and explains how this project tries to address a few of them.

### 4.2 Traditional Machine Learning Approaches

Early work on fake news and text classification in general relied mostly on traditional Machine Learning methods. These methods first turn the text into numbers using a technique such as TF-IDF [9], and then use a classification algorithm such as Logistic Regression, Naive Bayes, or Support Vector Machine (SVM) to learn the difference between the two classes, Fake and Real. The exact working of these algorithms is covered later in the "Algorithms and Tools Used" chapter.

These methods became popular mainly because they were fast, simple to implement, and gave reasonably good results without needing large datasets or heavy computing power. Logistic Regression, Naive Bayes, and SVM were already common choices for general text classification tasks well before fake news detection became a specific area of study, so it was natural for early fake news research to build on these already well-established methods.

Shu et al. [1] reviewed a wide range of fake news detection techniques and pointed out that feature-based methods, using word frequency and linguistic style, were the starting point for most early fake news detection systems. Ahmed et al. [2] applied n-gram based TF-IDF features with several Machine Learning classifiers, including Logistic Regression and SVM, on a dataset of fake and genuine news articles, and found that these methods could reach high accuracy while remaining simple to implement. Pérez-Rosas et al. [3] also used linguistic and stylistic features with traditional classifiers and showed that this kind of writing-style analysis can separate fake and real articles reasonably well, even without a deep understanding of the actual facts in the text.

The main advantage of these methods is that they are fast to train, easy to interpret, and do not need large amounts of data or computing power. Their main limitation is that they treat text as a bag of words. This means that word order is ignored completely, and a sentence like "the report did not confirm the claim" is treated no differently from "the report did confirm the claim," since both contain the same words. This limitation is one of the main reasons researchers started looking at Deep Learning methods, which can read text as a sequence rather than as an unordered set of words.

### 4.3 Deep Learning Approaches

Deep Learning became popular for text classification because it can learn its own representation of language directly from data, instead of depending on a fixed formula such as TF-IDF. Recurrent Neural Networks (RNNs) were among the first Deep Learning models used for text, since they process a sentence one word at a time and carry forward some memory of the words seen so far. However, plain RNNs struggle with longer text, because the influence of earlier words fades quickly as the network processes more words. This is known as the vanishing gradient problem.

Long Short-Term Memory (LSTM) networks were developed to address this limitation. An LSTM uses a set of gates to control what information is kept, what is added, and what is passed on at each step, which allows it to remember relevant information over much longer sequences than a plain RNN. This is why LSTM became, and still remains, one of the most widely used models for text classification tasks, including fake news detection. Convolutional Neural Networks (CNNs), originally built for image processing, have also been applied to text by treating a sentence as a small grid of word vectors and scanning it for useful local patterns, such as short phrases.

Wang [4] introduced the LIAR dataset, a large collection of short political statements labelled for truthfulness, and tested several models on it, including CNN and LSTM-based architectures, showing that Deep Learning models could pick up useful patterns from political text beyond what simple word-frequency methods could capture. Hochreiter and Schmidhuber [5], in the original paper that introduced the LSTM architecture, showed how the gating mechanism solves the long-standing problem of training networks on long sequences, which laid the foundation for almost all later LSTM-based text classification work. Kim [6] showed that even a fairly simple CNN, applied to sentence-level text classification, could perform competitively, which encouraged more research into applying different neural network architectures to text problems, including fake news detection.

The main advantage of Deep Learning models over traditional Machine Learning methods is their ability to use word order and context, which allows them to notice patterns such as negation that a bag-of-words model cannot see at all. Their main limitations are that they usually need more data to train well, take much longer to train, and are harder to interpret, since there is no simple per-word weight to look at, unlike Logistic Regression.

### 4.4 Recent Developments

More recently, Attention mechanisms and Transformer-based models have become the standard approach for many Natural Language Processing tasks, including fake news detection. Vaswani et al. [7] introduced the Transformer architecture, which uses an attention mechanism to let a model look at all words in a sentence at once and decide which ones matter most for a given prediction, instead of reading the sentence strictly word by word like an LSTM does. Building on this idea, Devlin et al. [8] introduced BERT, a large pretrained language model that can be fine-tuned for many different tasks, including text classification, and has since been used in a number of fake news detection studies with strong results.

These models generally perform very well, but they also come with a much higher level of complexity, require significantly more computing power to train, and are harder to fully explain at an implementation level. Since this project focuses on comparing a traditional Machine Learning model with a straightforward Deep Learning model in a clear and explainable way, Transformer-based models such as BERT were kept out of scope, and are mentioned here only to show that they exist as a natural next step beyond what this project covers.

### 4.5 Research Gaps

Looking at the existing work as a whole, a few practical gaps stand out.

First, many studies place a strong emphasis on reaching the highest possible accuracy, often without spending much time explaining why one model performs better than another, or under what conditions. Second, while both traditional Machine Learning and Deep Learning methods have been studied individually, relatively few studies compare them directly on the exact same dataset, using the exact same train, validation, and test split. Without this kind of controlled comparison, it can be difficult to say whether a Deep Learning model is actually better, or whether the improvement is coming from some other difference in how the two models were evaluated.

Third, in many studies, dataset quality receives less attention than model performance. Several fake news datasets, including well-known ones, contain patterns that have nothing to do with the actual truthfulness of an article, such as formatting differences or source-specific writing style. Relatively few studies look closely enough to check whether their model is learning to detect fake news, or simply learning to detect which source an article came from. This is closely related to label leakage, a situation where a feature in the dataset can predict the label almost by accident, and this issue is not always investigated in detail in existing fake news detection literature.

Fourth, statistical comparison between models is often limited. A model reporting 98% accuracy and another reporting 97% accuracy are sometimes presented as if one is simply "better," without checking whether this difference is actually meaningful or could just be due to chance on that particular test set.

Finally, a number of studies stop at reporting evaluation metrics and do not go on to show a working, usable system. This can make it harder to judge how such a model would actually perform in a real, practical setting, such as being used through a web interface.

### 4.6 Chapter Summary

This chapter looked at how fake news detection techniques have developed, starting from traditional Machine Learning methods such as TF-IDF with Logistic Regression, Naive Bayes, and SVM, moving to Deep Learning methods such as RNN, LSTM, and CNN, and briefly touching on newer Transformer-based models such as BERT. Across this body of work, a few consistent gaps were identified: a strong focus on accuracy alone, limited fair comparison between traditional and Deep Learning models on identical data, less attention paid to dataset quality and label leakage, limited use of statistical testing when comparing models, and few examples of a fully working, deployed application.

This project attempts to address several of these gaps. It carries out a fair comparison between a traditional Machine Learning model and a Deep Learning model under the same conditions, along with a careful analysis of the dataset used. It also applies statistical evaluation to check whether any difference between the models is meaningful, rather than relying on accuracy numbers alone. Finally, it goes beyond model evaluation by deploying a working web application, so that the system can actually be used and demonstrated, not just measured on paper. The detailed methodology behind each of these steps is explained in the chapters that follow.

#### References

[1] Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. *ACM SIGKDD Explorations Newsletter*.

[2] Ahmed, H., Traore, I., & Saad, S. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. *International Conference on Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments (ISDDC)*.

[3] Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic Detection of Fake News. *Proceedings of the 27th International Conference on Computational Linguistics (COLING)*.

[4] Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)*.

[5] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.

[6] Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

[7] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

[8] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*.

[9] Jurafsky, D., & Martin, J. H. (2009). *Speech and Language Processing* (2nd ed.). Prentice Hall.

---

## Chapter 5 – System Analysis / Problem Definition

### 5.1 Introduction

Before describing how this project was built, it helps to look closely at the problem itself and at what already exists to solve it. This chapter defines the problem, looks briefly at existing approaches, and explains the requirements the proposed system needs to meet. Implementation details are left to later chapters.

### 5.2 Problem Definition

Online news and social media have made it very easy for anyone to publish and share information. This has clear benefits, since news can now reach people almost instantly. At the same time, it has made it just as easy to spread false or misleading information, usually called fake news, without any proper check on whether it is true.

Once a fake news article is published, it can be shared and read by a large number of people within a short time. This creates a real problem, because incorrect information, once it spreads, is hard to correct. People may form opinions or make decisions based on it before anyone points out that it is false.

Manual verification is the traditional way of dealing with this problem. Journalists and fact-checking organisations read an article, check its claims against reliable sources, and then confirm whether it is true or false. This process works well for accuracy, but it takes time, and it depends on the availability of trained people. Given how much content is published online every single day, manual verification alone cannot keep up with the volume.

This creates a clear need for an automated system that can look at a news article and give a quick indication of whether it is likely to be fake or real, based on patterns learned from a large number of examples. Such a system cannot replace careful human fact-checking, but it can act as a first, fast filter, and it is exactly this need that defines the problem this project addresses.

### 5.3 Existing System

A few different approaches already exist for dealing with fake news.

Manual verification, done by journalists and fact-checking organisations, remains the most trusted approach, since it is based on direct human research. However, it works on one claim at a time and cannot be scaled up to the volume of content published daily.

Traditional Machine Learning methods, such as TF-IDF with Logistic Regression, Naive Bayes, or Support Vector Machines, represent text using word-based features and classify it as fake or real. These methods are fast and simple but do not consider word order, as discussed in the Literature Review chapter.

Deep Learning methods, such as LSTM networks, read text as a sequence and can make use of word order and context, unlike traditional methods, though they generally need more data and computing power to train. Both approaches are compared in detail in later chapters of this dissertation.

### 5.4 Limitations of Existing Systems

Table 5.1 summarises the practical limitations of the existing approaches described above.

**Table 5.1: Limitations of Existing Approaches**

| Limitation | Description |
|---|---|
| Manual verification | Slow, and depends on the availability of trained people. |
| Scalability | Difficult to check the large volume of news published every day. |
| Traditional machine learning | Does not consider word order or context. |
| Deep learning | Needs more data and computing power to train. |
| Dataset quality | Can affect what a model actually learns, discussed further in later chapters. |
| Deployment | Many studies stop after reporting evaluation metrics, without building a usable system. |

### 5.5 Proposed System

This project proposes a Fake News Detection System that, at a high level:

- accepts a news article as input,
- predicts whether the article is Fake or Real,
- compares a traditional Machine Learning model (Logistic Regression) with a Deep Learning model (LSTM) on the same task, and
- provides a simple web interface where a user can submit an article and view the prediction.

The details of how the dataset was prepared, how each model was built, and how the system was deployed are explained in the chapters that follow.

### 5.6 Functional Requirements

1. **Accept and validate input.** The system must allow a user to submit the text of a news article and check that it is not empty or an unreasonable length before processing it.
2. **Process the article.** The system must clean and prepare the input text so that it can be used by the trained model.
3. **Classify the article.** The system must predict whether the input is Fake or Real using the trained model.
4. **Display the prediction.** The system must show the result to the user, along with a confidence score.
5. **Provide web and API access.** The system must be usable both through a web page and through a structured API response.
6. **Handle invalid input.** The system must show a clear, non-technical message if the input is invalid or if an error occurs.

### 5.7 Non-functional Requirements

1. **Usability.** The system should be simple enough for a non-technical user to operate.
2. **Reliability.** The system should consistently return a valid prediction without crashing.
3. **Maintainability.** The code should be organised clearly enough to be understood and modified later.
4. **Performance.** The system should return a prediction quickly enough to feel responsive.
5. **Scalability.** The design should not prevent the system from handling more users or data in the future.
6. **Reproducibility.** The same input and the same trained model should always give the same prediction.

### 5.8 Chapter Summary

This chapter defined the problem this project addresses: the difficulty of manually verifying the growing volume of online news, and the resulting need for an automated detection system. It looked briefly at existing approaches and their limitations, described the proposed system at a high level, and listed its functional and non-functional requirements. The next chapter explains the methodology followed to build this system.

---

## Chapter 6 – Methodology

### 6.1 Introduction

This chapter describes the methodology followed to build the Fake News Detection System, from selecting the dataset to deploying the final application. It explains what was done at each stage and in what order. The algorithms used are explained in detail in the Algorithms and Tools chapter, and the actual results obtained are presented in the Testing and Results chapter.

### 6.2 Overall Workflow

The project followed a clear, step-by-step workflow:

1. Dataset selection
2. Exploratory Data Analysis (EDA)
3. Label leakage investigation
4. Data preprocessing
5. Train, validation, and test split
6. TF-IDF and Logistic Regression model development
7. LSTM model development
8. Model evaluation
9. Reuters ablation study
10. Statistical significance testing (McNemar's Test)
11. Deployment using FastAPI and a Bootstrap frontend

*[Figure 6.1: Overall project workflow — see `report/diagrams/workflow_diagram.png`]*

Each of these stages is explained in more detail in the sections that follow.

### 6.3 Dataset

The dataset used in this project is the "Fake and Real News Dataset," published on Kaggle by Clément Bisaillon. It consists of two files, Fake.csv and True.csv, containing about 23,500 fake articles and 21,400 real articles respectively, for a combined total of around 45,000 news articles. Each article includes a title, the article text, a subject category, and a publication date.

This dataset was chosen because it matches the dataset already named in the project proposal, is large enough to train an LSTM model without needing a GPU, and has a roughly balanced number of fake and real articles, which avoids the need for extra techniques to handle class imbalance. It is also a widely used dataset in fake news detection research, which made it easier to relate this project's approach to the existing work discussed in the Literature Review chapter.

### 6.4 Exploratory Data Analysis

Before any preprocessing or model building, the dataset was studied through Exploratory Data Analysis (EDA). This included checking the class distribution to confirm the dataset was reasonably balanced between Fake and Real articles, checking for missing values in the title and text fields, identifying duplicate articles, and looking at the distribution of article lengths for both classes.

This analysis also looked at the subject and date columns, and at the way each article began, to understand how the dataset was structured. This revealed some unusual patterns that seemed to separate the two classes too easily, which led to a closer investigation into possible label leakage, described in the next section.

### 6.5 Label Leakage Investigation

Label leakage happens when a feature in the dataset allows a model to predict the label without actually learning anything meaningful about the content. If left unaddressed, a model can appear highly accurate while really just relying on a shortcut rather than genuine understanding.

During EDA, two such shortcuts were found in this dataset. The subject column showed an almost complete separation between the two classes, meaning it could predict whether an article was Fake or Real almost by itself. Similarly, most Real articles began with a Reuters wire-service dateline, such as "WASHINGTON (Reuters) -", which Fake articles almost never contained.

Since neither of these patterns has anything to do with whether an article's content is actually true or false, both were treated as label leakage. The subject and date columns were removed before modelling, and the leading Reuters dateline was stripped from the article text during preprocessing. This ensured that the models would need to rely on the actual content of the articles rather than these incidental patterns.

### 6.6 Data Preprocessing

Once the leakage-related columns and text were addressed, the remaining article text was cleaned and prepared for modelling. This included removing HTML tags and URLs, converting all text to lowercase, and removing punctuation.

Preprocessing then differed slightly between the two models. For the Logistic Regression model, stop words were removed and the remaining words were reduced to their base form through lemmatization, since this model does not depend on word order. For the LSTM model, stop words and word order were kept intact, since the model relies on reading the article as a sequence, and the resulting text was tokenized and padded to a fixed length so it could be used as input to the network.

The exact preprocessing functions and settings used for each model are described in the Algorithms and Tools chapter.

### 6.7 Model Development

#### 6.7.1 Logistic Regression

The cleaned article text for this model was converted into numerical features using TF-IDF, and a Logistic Regression classifier was then trained on these features using the training portion of the dataset. The trained vectorizer and model were saved for later evaluation and reuse.

#### 6.7.2 LSTM

For the LSTM model, the cleaned and tokenized article text was converted into padded numerical sequences and passed through an embedding layer, followed by an LSTM layer and a final classification layer. The model was trained on the same training data used for the Logistic Regression model, using the corresponding validation set to monitor training progress. The trained model and its tokenizer were saved for later evaluation and reuse.

Both models are described in full detail, including their architecture and configuration, in the Algorithms and Tools chapter.

### 6.8 Model Evaluation

Both models were evaluated using the same held-out test dataset, which was not used during training or validation for either model. This ensured that the two models could be compared fairly.

The metrics used to evaluate both models were Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix. Accuracy shows the overall proportion of correct predictions, Precision and Recall show how well the model performs on each class specifically, F1-Score combines Precision and Recall into a single value, and the Confusion Matrix shows the exact number of correct and incorrect predictions for each class. The actual values obtained for these metrics are presented in the Testing and Results chapter.

### 6.9 Reuters Ablation Study

Since the Reuters dateline was identified as a possible source of label leakage, an additional ablation study was carried out to check whether this specific pattern was still influencing the Logistic Regression model, even after the leading dateline had already been removed.

An ablation study works by removing one specific element from the system and observing what changes. In this case, all remaining occurrences of the word "Reuters" were removed from the article text, and the Logistic Regression model was retrained and evaluated on this modified version of the dataset, using the same train, validation, and test split as before. The outcome of this study is discussed in the Testing and Results chapter.

### 6.10 Statistical Significance Testing

Comparing the Logistic Regression and LSTM models only by their accuracy or F1-Score does not show whether any difference between them is meaningful or simply due to chance on that particular test set. To check this, McNemar's Test was used, which compares two models evaluated on the same test set by looking specifically at the cases where they disagree. The purpose and outcome of this test are explained further in the Testing and Results chapter.

### 6.11 Deployment

Once the LSTM model was trained and evaluated, it was deployed as a working web application. A FastAPI backend loads the trained model and tokenizer once when the application starts, and exposes an endpoint that accepts a news article and returns a prediction along with a confidence score. A simple Bootstrap-based web page allows a user to paste an article, submit it, and view the result. This application is described in more detail in the System Design and Implementation chapters.

### 6.12 Chapter Summary

This chapter described the methodology followed in this project, from selecting and studying the dataset to deploying the final web application. It covered the dataset used, the exploratory analysis carried out, the label leakage investigation, the preprocessing steps applied, the development of the Logistic Regression and LSTM models, the metrics used for evaluation, the Reuters ablation study, and the statistical test used to compare the two models. The next chapter describes the algorithms and tools used in more detail.

---

## Chapter 7 – Algorithms and Technologies Used

### 7.1 Introduction

This chapter explains the algorithms, evaluation metrics, and technologies used in this project. It focuses on what each of these is and why it is generally suitable for a task like fake news detection, rather than on how this project specifically used them, which is covered in the Methodology and Implementation chapters.

### 7.2 TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a technique used to convert text into numerical features that a Machine Learning model can work with. It gives each word in a document a score based on how often the word appears in that document and how rare it is across the rest of the dataset:

**TF-IDF(t, d) = TF(t, d) × IDF(t)**

A word that appears often in one article but rarely in others gets a higher score, while common words that appear in almost every article get a lower score.

TF-IDF is widely used for text classification because it is simple, fast to compute, and does not need a large amount of training data to work well. It was selected for this project because it gives a clear, interpretable numerical representation of each article, without needing complex preprocessing or heavy computing power. This makes it a suitable feature representation for the Logistic Regression baseline model used in this project.

### 7.3 Logistic Regression

Logistic Regression is a Machine Learning algorithm used for binary classification tasks, where the goal is to assign an input to one of two classes, such as Fake or Real. It works by calculating a probability, a value between 0 and 1, using the sigmoid function:

**P(y=1) = 1 / (1 + e^(-z))**

where z is a weighted combination of the input features. This probability is then converted into a final class label using a threshold, typically 0.5.

Logistic Regression is efficient to train, easy to interpret, since each feature has a weight showing how strongly it is associated with one class, and performs well on text classification tasks despite its simplicity.

It was selected as the baseline model for this project because it offers a good balance between simplicity, speed, and interpretability, making it a useful and fair point of comparison against the more complex LSTM model.

### 7.4 Long Short-Term Memory (LSTM)

Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequences of data, such as sentences, by reading one word at a time and carrying forward information from earlier words. However, plain RNNs struggle to retain information over long sequences, since the influence of earlier words tends to fade out as more words are processed. This is known as the vanishing gradient problem.

Long Short-Term Memory (LSTM) networks were developed to address this limitation. An LSTM has an internal memory, along with three gates that control how this memory is used: a forget gate, which decides what information to discard, an input gate, which decides what new information to add, and an output gate, which decides what information to pass on at each step. Together, these gates allow an LSTM to retain useful information over much longer sequences than a plain RNN, while still being able to forget information that is no longer relevant.

LSTM was selected as the Deep Learning model for this project because it reads an article as an ordered sequence of words rather than as an unordered collection, allowing it to make use of word order and context in a way TF-IDF cannot. This made it a suitable choice for testing whether this additional capability leads to better performance on this dataset.

### 7.5 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions. |
| Precision | Measures how many predicted Real cases are actually Real. |
| Recall | Measures how many actual Real cases are correctly identified. |
| F1-Score | Balances Precision and Recall into a single value. |
| Confusion Matrix | Shows correct and incorrect predictions for each class. |

### 7.6 McNemar's Test

McNemar's Test is a statistical test used to compare two classifiers evaluated on the same test dataset. Instead of comparing overall accuracy directly, it focuses only on the cases where the two models disagree, one predicts correctly and the other does not, and checks whether one model is wrong noticeably more often than the other in these cases:

**χ² = (|b − c| − 1)² / (b + c)**

where b and c are the number of disagreement cases in each direction.

This test is useful when comparing two models like Logistic Regression and LSTM, because a small difference in accuracy could simply be due to chance. McNemar's Test gives a p-value, which indicates whether the observed difference is statistically significant or could reasonably have happened by chance. A low p-value suggests the difference is unlikely to be due to chance alone, while a high p-value suggests there is not enough evidence to say one model is genuinely better than the other.

### 7.7 Technologies Used

**Table 7.1: Technologies Used in the Project**

| Technology | Purpose |
|---|---|
| Python | Main programming language used to build the entire project. |
| Pandas | Used for loading, cleaning, and manipulating the dataset. |
| NumPy | Used for numerical operations required during preprocessing and model training. |
| Scikit-learn | Used to build the TF-IDF vectorizer and the Logistic Regression model, and to compute evaluation metrics. |
| TensorFlow / Keras | Used to build, train, and save the LSTM model. |
| FastAPI | Used to build the backend API that serves predictions from the trained model. |
| Bootstrap | Used to build the web interface that allows users to submit articles and view predictions. |
| Uvicorn | Used as the server that runs the FastAPI application. |
| Jupyter Notebook | Used to carry out exploratory data analysis and to develop and test the models step by step. |

### 7.8 Chapter Summary

This chapter explained the algorithms, evaluation metrics, and technologies used in this project. It covered TF-IDF and Logistic Regression as the basis for the baseline model, LSTM as the Deep Learning model, the metrics used to evaluate both models, McNemar's Test used to compare them statistically, and the technologies used to build and deploy the system. The next chapter describes the design of the system in more detail.

---

## Chapter 8 – System Architecture and Design

### 8.1 Introduction

This chapter describes the overall architecture and design of the Fake News Detection System. It focuses on how the different components of the system fit together and interact with each other, rather than on how each component was implemented, which is covered in the Implementation chapter.

### 8.2 Overall System Architecture

The architecture presented in this chapter reflects the deployed system, the working application described in Chapter 5, rather than the full set of experiments carried out during the study. Both the Logistic Regression and LSTM models were developed and evaluated as part of this project, but only the trained LSTM model is included in the deployed application, since the objective of this phase is to demonstrate a Deep Learning-based fake news detection system in a working form.

The system follows a simple client-server architecture, made up of five main parts: a Bootstrap-based web interface, a FastAPI backend, a text preprocessing module, a trained LSTM model, and a prediction response that is sent back to the user.

*[Figure 8.1: Overall System Architecture — see `report/diagrams/architecture_diagram.png`]*

This architecture separates the presentation layer, application logic, and prediction components from each other, which makes the system easier to maintain and extend.

A user interacts with the system entirely through the web interface, which runs in the browser. When the user submits a news article, the request travels to the FastAPI backend, which passes the article through the preprocessing module before handing it to the trained LSTM model. The model produces a prediction, which the backend formats into a response and sends back to the web interface for display.

### 8.3 Data Flow

Figure 8.2 shows the flow of data through the system as a Level 0 Data Flow Diagram.

*[Figure 8.2: Level 0 Data Flow Diagram — see `report/diagrams/dfd_level0.png`]*

The data flow follows five steps:

1. User enters or pastes a news article.
2. The web interface sends the article to the FastAPI backend.
3. The backend preprocesses the article text.
4. The trained LSTM model predicts whether the article is Fake or Real.
5. The prediction and confidence score are returned to the user.

The system does not permanently store submitted articles or prediction results.

### 8.4 Use Case Diagram

Figure 8.3 shows the use case diagram for the system.

*[Figure 8.3: Use Case Diagram — see `report/diagrams/use_case_diagram.png`]*

There is only one actor in this system: the User. The User can perform four main actions: entering or pasting the text of a news article, submitting the article for classification, viewing the predicted label, and viewing the confidence score associated with that prediction. These use cases follow one after another in a simple, linear sequence, since the system does not include any other actor, such as an administrator, or any additional functionality beyond prediction.

### 8.5 Component Design

Table 8.1 summarises the main components of the system and their individual responsibilities.

**Table 8.1: System Components and Responsibilities**

| Component | Responsibility |
|---|---|
| Bootstrap UI | Provides the web page where a user enters an article, submits it, and views the prediction and confidence score. |
| FastAPI Backend | Receives requests from the UI, coordinates preprocessing and prediction, and returns the response. |
| Preprocessing Module | Cleans and prepares the article text so it can be used by the trained model. |
| LSTM Model | Takes the processed text and predicts whether the article is Fake or Real. |
| Response Handler | Formats the prediction and confidence score into a structured response sent back to the UI. |

### 8.6 User Interface Design

The user interface is a single web page, kept intentionally simple so that it is easy to use without any training.

*[Figure 8.4: User Interface Design — see `report/diagrams/ui_wireframe.png`]*

The home page displays the title of the system and a large text input area where a user can paste the content of a news article. Below the text area is a Predict button, which the user clicks to submit the article. Once a prediction is ready, the page displays two additional pieces of information: the predicted label, either Fake or Real, and the confidence score, shown as a percentage. The interface consists of a single page without additional menus or navigation, since the application is designed to perform one primary task: classifying news articles as Fake or Real.

### 8.7 Chapter Summary

This chapter described the architecture and design of the Fake News Detection System. It explained how the Bootstrap UI, FastAPI backend, preprocessing module, and LSTM model interact with each other, presented the data flow and use case diagrams, summarised the responsibility of each component, and described the layout of the user interface. The next chapter explains how this design was actually implemented.

---

## Chapter 9 – Implementation Details

### 9.1 Introduction

This chapter describes how the Fake News Detection System was actually implemented, covering the development environment, dataset preparation, text preprocessing, model implementation, the backend, the user interface, and the overall project structure. It focuses on how each part was built, not on why particular methods or algorithms were chosen, which was already covered in earlier chapters.

### 9.2 Development Environment

**Table 9.1: Development Environment**

| Component | Version / Technology |
|---|---|
| Python | 3.10 |
| Pandas | 2.3.3 |
| NumPy | 2.2.6 |
| Scikit-learn | 1.7.2 |
| TensorFlow / Keras | 2.21.0 / 3.12.3 |
| FastAPI | 0.139.2 |
| Uvicorn | 0.51.0 |
| Bootstrap | 5.3.3 (via CDN) |
| Jupyter Notebook | 1.1.1 |
| Visual Studio Code | Current stable release |

### 9.3 Dataset Preparation

The implementation began by loading Fake.csv and True.csv separately using Pandas. Each row in Fake.csv was assigned the label Fake, and each row in True.csv was assigned the label Real. The two datasets were then combined into a single DataFrame and shuffled.

The subject and date columns were removed from the combined dataset, along with a small number of corrupted rows identified during data cleaning, and duplicate articles were removed. The remaining data was then split into training, validation, and test sets. This split was performed using a fixed random seed, so that the same training, validation, and test sets could be reproduced consistently throughout the project.

### 9.4 Text Preprocessing

The cleaned article text was processed through a series of steps, including removing HTML tags and URLs, converting text to lowercase, and removing punctuation. Two separate versions of the cleaned text were then produced, since the two models required slightly different input.

For the Logistic Regression model, the text was further processed by removing stop words and reducing each word to its base form through lemmatization.

For the LSTM model, stop words and word order were kept as they were, since the model needed the full sequence of words. This text was later converted into tokenized, padded sequences during model implementation.

### 9.5 Model Implementation

#### 9.5.1 Logistic Regression

The Logistic Regression model was implemented using scikit-learn. The cleaned text for this model was converted into numerical features using a TF-IDF vectorizer, fitted only on the training data. A Logistic Regression classifier was then trained on these features. After training, both the fitted vectorizer and the trained classifier were saved to disk using joblib, so that they could be reloaded later without retraining.

#### 9.5.2 LSTM

The LSTM model was implemented using TensorFlow and Keras. A Keras Tokenizer was fitted on the training text and used to convert each article into a sequence of integers. These sequences were padded to a fixed length so that they could be processed in batches.

The model itself was built as a sequential stack of layers: an embedding layer, which converts each word into a dense vector, followed by an LSTM layer, and a final dense output layer that produces a single probability. After training, the trained model was saved in the Keras format, and the fitted tokenizer was saved separately using joblib, so that both could be reloaded for prediction without retraining.

### 9.6 Backend Implementation

The backend was implemented using FastAPI. The application loads the trained LSTM model and tokenizer during application startup, so that they do not need to be reloaded for every request.

The main functionality is exposed through a POST /predict endpoint, which accepts a news article as JSON input. Incoming requests are validated using Pydantic models, which check that the submitted text is not empty and falls within an acceptable length range before any prediction is attempted. Once validated, the article is passed through the preprocessing and prediction pipeline, and the resulting label, confidence score, and probability values are returned to the client as a JSON response, so that the frontend can read and display the result directly. Two additional endpoints, GET / and GET /health, are also provided to give a basic welcome message and confirm that the server is running.

*[Figure 9.1: Backend Request Flow — see `report/diagrams/backend_request_flow.png`]*

### 9.7 User Interface Implementation

The user interface was built as a single HTML page styled with Bootstrap. It contains a text area where a user can paste a news article, along with a Predict button and a Clear button.

When the Predict button is clicked, the page sends the article to the backend using a JavaScript fetch request and disables the button while waiting for a response, showing a short "Predicting..." message. Once a response is received, the page displays the predicted label, the confidence score, and the name of the model used, without requiring the page to reload.

*[Figure 9.2: Application Home Page — use the actual screenshot at `report/screenshots/home_page.png`]*

### 9.8 Project Structure

```
fake-news-detection/
├── api/            FastAPI backend (app, routes, model loading, prediction)
├── config/         central project settings
├── dataset/        raw and processed data
├── preprocessing/  text cleaning and preprocessing pipeline
├── training/       dataset splitting and model training scripts
├── evaluation/     metrics, plots, and statistical tests
├── models/         saved Logistic Regression and LSTM artifacts
├── notebooks/      Jupyter notebooks for each project phase
├── frontend/       Bootstrap web interface
├── testing/        automated API tests
├── demo/           sample articles for demonstration
├── docs/           project documentation and reports
├── report/         figures, tables, and screenshots
├── utils/          shared helper modules, such as logging
└── requirements.txt
```

### 9.9 Chapter Summary

This chapter described how the Fake News Detection System was implemented, covering dataset preparation, text preprocessing, the implementation of the Logistic Regression and LSTM models, the FastAPI backend, the Bootstrap user interface, and the overall project structure. The next chapter presents the testing carried out on the system and the results obtained.

---

## Chapter 10 – Testing and Results

### 10.1 Introduction

This chapter presents the testing carried out on the Logistic Regression and LSTM models, and the results obtained. It covers the test environment used, the results for each model individually, a direct comparison between the two, the outcome of the Reuters ablation study, and the statistical test used to check whether the difference between the two models is meaningful. The discussion at the end of this chapter interprets what these results actually mean.

### 10.2 Test Environment

Both models were evaluated on the same held-out test set, kept aside using a fixed random seed and not used during training or validation for either model. The test set contains 5,796 articles, made up of 2,617 Fake and 3,179 Real articles, matching the overall class balance of the full dataset.

The trained Logistic Regression and LSTM models saved during implementation were loaded for evaluation. No additional training was performed during testing. All testing was carried out using Python 3.10, with scikit-learn used to evaluate the Logistic Regression model and TensorFlow/Keras used to evaluate the LSTM model, as listed in Chapter 9.

#### 10.2.1 Hardware Configuration

Testing was carried out on a laptop with an AMD Ryzen 7 5700U processor, 16 GB of RAM, running Windows 11 Home. No GPU acceleration was used; both models were evaluated using CPU-only computation.

### 10.3 Evaluation Metrics

The same evaluation metrics introduced in Chapter 7, Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix, were used to evaluate both models on the test set described above.

### 10.4 Logistic Regression Results

**Table 10.1: Logistic Regression Results**

| Metric | Value |
|---|---|
| Accuracy | 98.24% |
| Precision | 97.88% |
| Recall | 98.93% |
| F1-Score | 98.40% |

*[Figure 10.1: Logistic Regression Confusion Matrix]*

The confusion matrix shows 2,549 true negatives, 68 false positives, 34 false negatives, and 3,145 true positives. This means the model correctly classified the large majority of both Fake and Real articles in the test set, with relatively few mistakes in either direction.

### 10.5 LSTM Results

**Table 10.2: LSTM Results**

| Metric | Value |
|---|---|
| Accuracy | 97.20% |
| Precision | 97.57% |
| Recall | 97.33% |
| F1-Score | 97.45% |

*[Figure 10.2: LSTM Confusion Matrix]*

The confusion matrix shows 2,540 true negatives, 77 false positives, 85 false negatives, and 3,094 true positives. The LSTM model also performed well overall, though with slightly more false positives and false negatives than the Logistic Regression model.

### 10.6 Model Comparison

**Table 10.3: Comparison of Logistic Regression and LSTM**

| Aspect | Logistic Regression | LSTM |
|---|---|---|
| Accuracy | 98.24% | 97.20% |
| Precision | 97.88% | 97.57% |
| Recall | 98.93% | 97.33% |
| F1-Score | 98.40% | 97.45% |
| Training time | 0.25 seconds | About 599 seconds (~10 minutes) |
| Inference time per article | 0.22 ms | 1.43 ms |
| Trainable parameters | 20,001 | 2,044,353 |
| Model size (saved files) | 0.90 MB | 28.69 MB |

On this dataset, Logistic Regression performed slightly better than the LSTM on every metric measured. The difference in training time is much larger, since the Logistic Regression model trains in a fraction of a second, while the LSTM takes several minutes. The LSTM also has many more parameters and a larger saved file size, since most of its parameters come from the embedding layer. In terms of inference time, both models are fast enough to be used in a live web application, though the Logistic Regression model is still faster per article.

Overall, the Logistic Regression model gave slightly better accuracy at a much lower computational cost, while the LSTM required significantly more time and resources without a corresponding improvement in performance on this particular dataset.

### 10.7 Reuters Ablation Study

As explained in Chapter 6, the Reuters wire-service dateline was identified as a possible source of label leakage. To check whether this pattern was still affecting the Logistic Regression model after the leading dateline had already been removed, every remaining occurrence of the word "Reuters" was removed from the article text, and the model was retrained and evaluated on this modified dataset, using the same train, validation, and test split as before.

**Table 10.4: Reuters Ablation Results**

| Metric | Original Baseline | Reuters Removed |
|---|---|---|
| Accuracy | 98.24% | 98.26% |
| Precision | 97.88% | 97.97% |
| Recall | 98.93% | 98.87% |
| F1-Score | 98.40% | 98.42% |
| Training time | 0.25 seconds | 0.23 seconds |

Removing the remaining "Reuters" mentions did not meaningfully change the model's performance. All metrics changed by less than 0.2 percentage points, in either direction. This suggests that the model's high accuracy is not dependent on this one specific word, even though it had shown up as the strongest individual feature in the model's feature importance analysis.

### 10.8 McNemar's Test

McNemar's Test, introduced in Chapter 7, was used to check whether the difference between the Logistic Regression and LSTM models is statistically meaningful, and separately, whether the difference caused by removing "Reuters" mentions is meaningful.

**Table 10.5: McNemar's Test Results**

| Comparison | Discordant Cases | p-value |
|---|---|---|
| Logistic Regression vs. LSTM | 176 (118 in favour of Logistic Regression, 58 in favour of LSTM) | 0.000007 |
| Original Baseline vs. Reuters-Removed | 9 (4 in favour of the original, 5 in favour of Reuters-removed) | 1.0000 |

For the Logistic Regression vs. LSTM comparison, the p-value of 0.000007 is well below the common threshold of 0.05, which means the difference between the two models is statistically significant and unlikely to be due to chance. However, since this difference amounts to only about 1 percentage point in accuracy, and affects only a small proportion of the test set (176 out of 5,796 articles), it is a real but modest difference, not a dramatic one.

For the Reuters ablation comparison, the p-value of 1.0000 shows no statistically significant difference at all, supporting the earlier observation that removing the word "Reuters" did not meaningfully change the model's performance.

### 10.9 Discussion

The Logistic Regression model performed slightly better than the LSTM on this dataset, on every metric measured. A likely reason is that the Fake and Real classes in this dataset are separated largely by writing style rather than by more subtle, order-dependent language patterns. Reuters articles, which make up the Real class, generally follow a fairly consistent wire-service writing style, and this style-based signal is something a word-frequency-based model like Logistic Regression can pick up efficiently.

The LSTM's ability to use word order and context did not translate into a measurable advantage here, and its training was noticeably less stable across epochs than the Logistic Regression model's single, deterministic training run. With a moderate-sized dataset and no use of pretrained word embeddings, the LSTM likely did not have enough data to fully make use of its additional capacity.

Removing the label leakage caused by the subject column and the Reuters dateline was an important step, since without it, either model could have reached a very high accuracy without actually learning anything meaningful about the articles. The Reuters ablation study further showed that this leakage was not tied to one specific word, but was spread across several related patterns in how Reuters articles are written.

These results do not suggest that Deep Learning models are generally worse than traditional Machine Learning models. They indicate that, for this specific dataset and this specific LSTM configuration, the added complexity of the LSTM did not lead to better results, which is a useful and honest finding rather than a disappointing one.

### 10.10 Chapter Summary

This chapter presented the results of testing the Logistic Regression and LSTM models on the same test dataset. Logistic Regression reached 98.24% accuracy, and the LSTM reached 97.20% accuracy. McNemar's Test showed that this difference is statistically significant, although modest in practical terms. The Reuters ablation study showed that removing the word "Reuters" did not meaningfully affect performance, confirming that the earlier label leakage was addressed effectively rather than being reduced to a single removable word. The next chapter concludes the dissertation by summarising what this project achieved.

---

## Chapter 11 – Conclusion and Future Work

### 11.1 Introduction

This final chapter brings the dissertation to a close. It summarises the work carried out across the project, reflects on how well the stated objectives were met, draws together the key findings from earlier chapters, and honestly discusses the limitations of the current work. It ends with suggestions for future improvements and a final concluding statement.

### 11.2 Summary of the Project

This project set out to build a Fake News Detection System and to compare a traditional machine learning approach with a deep learning approach on the same task. Work began with collecting and exploring the Kaggle Fake and Real News Dataset, which led to the discovery of label leakage in the subject column and the Reuters wire-service dateline. These issues were addressed before any model was built, and the remaining article text was cleaned and prepared separately for each model.

A TF-IDF and Logistic Regression baseline was implemented first, followed by an LSTM model, both trained and evaluated on the same dataset split. The two models were then compared using standard evaluation metrics and McNemar's Test, and a separate ablation study was carried out to check whether the Reuters-related leakage had a lasting effect on the baseline's performance. Finally, the trained LSTM model was deployed as a working web application, using a FastAPI backend and a Bootstrap frontend, so that the system could be demonstrated in practice rather than only evaluated on paper.

### 11.3 Objectives Achieved

The objectives set out in Chapter 1 were addressed across the different stages of this project. A working Fake News Detection System was designed, built, and evaluated, meeting the project's central goal. The dataset was studied carefully before any modelling began, which made it possible to identify and address label leakage related to the subject column and the Reuters dateline, rather than allowing it to silently inflate accuracy.

Both a traditional machine learning model and a deep learning model were implemented and evaluated under the same conditions, using an identical dataset split, which allowed the two approaches to be compared fairly rather than assumed. This comparison was supported by statistical testing, using McNemar's Test, and by additional analysis through the Reuters ablation study, going beyond a simple accuracy comparison. Finally, the trained LSTM model was deployed as a functional web application, meeting the objective of producing a usable system rather than stopping at model evaluation alone.

### 11.4 Key Findings

A few key findings emerged from this project. Logistic Regression achieved the best overall performance on the test dataset, reaching 98.24% accuracy compared to the LSTM's 97.20%, and yet the LSTM still performed competitively, reaching a similar level of accuracy despite the added complexity of a deep learning approach. Removing the remaining "Reuters" mentions from the dataset had little effect on the Logistic Regression model's performance, showing that the earlier preprocessing steps had already addressed the more serious label leakage concerns, and McNemar's Test confirmed that the performance difference between Logistic Regression and LSTM was statistically significant, even though the practical difference in accuracy was small.

Taken together, these findings show that careful preprocessing and a deliberate investigation into label leakage were essential to trust the evaluation results, rather than accepting a high accuracy score at face value.

### 11.5 Limitations

This project has some limitations that should be kept in mind when interpreting its results.

The evaluation was carried out using a single public dataset, the Kaggle Fake and Real News Dataset, so the results reflect this dataset's specific characteristics and may not generalise directly to other sources of news. Only English-language news articles were considered, and the system has not been tested on other languages. Only one deep learning architecture, the LSTM, was implemented and evaluated; other architectures, including Transformer-based models such as BERT, were not explored as part of this project. The deployed system also only accepts manually entered or pasted news text, rather than, for example, a news article URL, and it does not monitor or process news in real time.

### 11.6 Future Work

Based on the limitations above, a few directions could be explored in future work. The system could be evaluated on additional, independently sourced datasets, to check whether its performance generalises beyond this project's specific dataset, and extended to support multilingual fake news detection rather than English text only. Transformer-based models, such as BERT or RoBERTa, could also be investigated to see whether they offer a meaningful improvement over the LSTM used in this project.

On the practical side, the application could be extended to accept a news article URL, with the system automatically extracting and processing the article content instead of requiring manually pasted text, and to support real-time news monitoring, so that articles could be checked as they are published rather than one at a time. Deploying the application on a cloud platform would make it accessible beyond a local machine, and the user interface could be improved with additional feedback or guidance for the user. Finally, incorporating Explainable AI techniques would allow a user to see which parts of an article influenced a given prediction, similar to the interpretability already available for the Logistic Regression model.

### 11.7 Final Conclusion

This project successfully developed and evaluated a deep learning-based Fake News Detection System, built around an LSTM model, and compared it against a traditional machine learning baseline using TF-IDF and Logistic Regression. Rather than assuming that a deep learning approach would automatically perform better, this project tested that assumption directly, and found that, on this particular dataset, the simpler Logistic Regression model performed slightly better while requiring far less time and computational effort to train.

Although the LSTM did not outperform the Logistic Regression model, this outcome does not diminish the value of the work carried out. Building and fairly evaluating both models required careful dataset analysis, a deliberate investigation into label leakage, and the use of appropriate statistical testing, all of which contributed to a more trustworthy set of results than accuracy figures alone could provide. The project also went beyond model evaluation by deploying a working web application, demonstrating that the system can be used in practice and not only measured in a notebook.

Overall, this project reflects a complete and honest attempt to apply machine learning and deep learning techniques to a real-world problem, from understanding the data through to building a usable system. It stands as a solid foundation on which the future improvements discussed above, and further work in this area, can be built.

---

## Appendix

### Appendix B: Sample API Request and Response

This appendix shows real request and response examples from the deployed `/predict` endpoint, captured while testing the application (see `testing/testing_report.md` for the full test run). These are actual outputs, not illustrative examples.

**Example 1 — A genuine Reuters article (`demo/real_article.txt`)**

Request:

```json
POST /predict
Content-Type: application/json

{
  "text": "Senior U.S. Republican senator: 'Let Mr. Mueller do his job'\n\nWASHINGTON (Reuters) - The special counsel investigation of links between Russia and President Trump's 2016 election campaign should continue without interference in 2018, despite calls from some Trump administration allies and Republican lawmakers to shut it down, a prominent Republican senator said on Sunday. ... [full article text, 2,789 characters]"
}
```

Response (`200 OK`):

```json
{
  "prediction": "Real",
  "confidence": 99.73,
  "probability_real": 0.9973,
  "probability_fake": 0.0027,
  "model": "LSTM"
}
```

**Example 2 — A fake/opinion-style article (`demo/fake_article.txt`)**

Request:

```json
POST /predict
Content-Type: application/json

{
  "text": "Drunk Bragging Trump Staffer Started Russian Collusion Investigation\n\nHouse Intelligence Committee Chairman Devin Nunes is going to have a bad day. He's been under the assumption, like many of us, that the Christopher Steele-dossier was what prompted the Russia investigation... [full article text, 1,898 characters]"
}
```

Response (`200 OK`):

```json
{
  "prediction": "Fake",
  "confidence": 99.40,
  "probability_real": 0.0060,
  "probability_fake": 0.9940,
  "model": "LSTM"
}
```

**Example 3 — Validation error (empty input)**

Request:

```json
POST /predict
Content-Type: application/json

{
  "text": ""
}
```

Response (`422 Unprocessable Entity`):

```json
{
  "error": "validation_error",
  "detail": "Article text cannot be empty."
}
```
