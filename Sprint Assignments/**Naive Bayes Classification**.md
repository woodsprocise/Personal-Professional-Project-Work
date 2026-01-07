# **Naive Bayes Classification**
## Overview
**Naive Bayes Classification** is a machine learning classification algorithm that stems from the prominent probability theorem, **Bayes Theorem.** Regarding what makes Bayes Classification a *naive assumption*, that will be covered later on in this article. In short, *Bayes Theorem* revolves around **conditional probabilities**, as stated in the formula, $P(B|A)$, "probability of event B, given event A has occurred". Virtually, *Bayes Theorem* is centered around the idea that additional events impact the initial probability.

**Bayes Theorem: 
$$P(B|A)=\frac{P(A|B)*P(B)}{P(A)}$$**

**Breaking down Bayes Theorem**

* $P(B|A)$ *Posterior Probability* - Our updated belief in $B$, given more data
    * **$Posterior Probability=\frac{(Conditional Probability)(Prior Probability)}{Evidence}$**
* $P(A|B)$ *Likelihood* - The probability of observing our data if $B$ is true
    * *Conditional probability*; liklihood of evidence given hypothesis is true
* $P(A)$ *Marginal* - Probability of our dataset across all possible $B$ values
    * Prior probability that the *evidence* is true
* $P(B)$ *Prior* - Our belief in $B$ before gathering more data
    * *Prior probability* of hypothesis

![image](https://hackmd.io/_uploads/Sk9TJNr1xe.png)(Patwariraghottam)

**Connecting Bayes Theorem and Classification**

The above information regarding *Bayes Theorem* provides insights into how it is connected to *Naive Bayes Classification*. Specifically, formulas above also demonstrate how *Naive Bayes Classification* uses concepts from *Bayes Theorem* such as *conditional* and *prior* probabilities to calculate *posterior* probabilities. Additionally, the image above provides a simple visual representation of how Naive Bayes model classifies information.

**Naive Bayes Classification**

Given *datasets*, **Naive Bayes Classification** assumes the variables that predict the target to be *conditionally independent features*. The only pushback to this is the variance in the *real world application* of the algorithm, which also provides insight into the meaning behind **Naive Bayes Classification**.

Virtually, in *practice*, the **independence assumption** is effective. On the flip side, when dealing with *real-world data*, features depend on each other in finding the target value which *Naive Bayes classifier fails to account for, resulting in **incorrect assumptions**, hence why the algorithm is called **'Naive'** (gamal).

Conceptually, **Naive Bayes Classification** takes in $x$ values, which are the *known values*, that inform $y$ values that we are trying to predict. On the other hand, the algorithm operates mathematically by *classifying* data from a dataset using the **Chain Rule**.

## Mathematical Process of Classification

* **Starting Point: Bayes Theorem**
    * Given the **Naive Assumption** that *all features are independent*, given a category, Bayes Theorem states:
        * $P(y|X)=\frac{P(X|y)*P(y)}{P(X)}$
            * $X=(x_1,x_2,...,x_n)$ ----> Feature Values
            * $y=$ Class Variable 
* **Decompose features using Chain Rule**
    * The *chain rule of conditionality* simplifies the classification process by decomposing *conditional probabilities*, known as *likelihood* $(P(X|y))$
        * $P(X|y)=P(x_1,x_2,...,x_n|y)$
    * Note that the conditional probabilities are independent of each other under **Naives conditional independence assumption**, hence the likelihood probability can be rewritten as:
        * $P(X|y)=P(x_1|y)*P(x_2|y)...P(x_n|y)$
* **Rewrite Bayes Theorem using New Information**
    * Given the independent feature values that were found using the chain rule, *Bayes Theorem* can be rewritten as:
        * $P(y|X)=\frac{P(x_1,x_2,...,x_n|y)}{P(x_1,x_2,...,x_n)}$
    * The denominator is constant regardless of what *class value*$(y)$ is chosen, making the above equation proportional to:
        * $P(y|X)=P(x_1|y)*P(x_2|y)...P(x_n|y)*P(y)$
* **Solve for Posterior Probability**
    * From here, solve for the **Posterior Probability** of each *feature value*, then **choose the category$(y)$ that has the highest probability** in order to maximize the quantity $P(y|X)=P(x_1|y)*P(x_2|y)...P(x_n|y)*P(y)$
    * Since the denominator stays constant (gamal), the *posterior probability* can be proportional to:
        * $P(y|x_1,x_2,...,x_n)\propto P(y)\prod_{i=1}^{n}P(x_i|y)$ 
* **Naive Bayes Classification Computationally**
    * Alike to *classifying* data mathematically, **Naive Bayes** shares the same goal *computationally* of **choosing the hypothesis with the highest probability of occurring**
    * This is known as finding the *maximum a posteriori*, by using an operation called **Argmax** that finds the feature value that solves for the maximum value from a category in relation to the highest probability for the target value (gamal)
        * $y=argmax_yP(y)\prod_{i=1}^{n}P(x_i|y)$

## The Application of the Algorithm

With a foundational understanding of the origin of **Naive Bayes Classification** and how it works conceptually as well as mathematically, lets take a deeper dive into *types of classifiers*, *kinds of data* it works with and its *application.*

### **Types of Classifiers**

*Naive Bayes Classifiers* work under the assumptions that predictors in a model are **conditionally independent** as well as that **all features contribute equally to the outcome** (What are Naive Bayes Classifiers). Without those two aforementioned conditions holding true, the algorithm will prove ineffective. In addition, for **Naive Bayes Classification** to prove effective for any dataset, it is crucial that the correct classifier is used. **Choosing the right bayes classifier is dependent upon the values within a given dataset and the kind of distribution that is best fit for each model.** 

The various types of **Naive Bayes Classifiers** are listed below (skit-learn developers)(What are Naive Bayes Classifiers)(patwariraghottam):

**Gaussian** (GaussianNB)
*Description:* Effectively used in datasets that have normal distribution patterns with continuous features. 
$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$$
*Variable meanings:* 
* $P(x_i|y)$ - predicted class
* $\mu_y$ - mean for feature class $y$
* $x_i$ - value of $i^{th}$ feature
* $\sigma^2_y$ - variance of feature value for class $y$

**Multinomial** (MultinomialNB)
*Description:* Especially useful when classifying texts or documents as well as with spam filtering and sentiment analysis. Deals with multinomially distributed data and often uses number counts or vector word counts to represent data.

$$\hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}$$
*Variable meanings:* 
$\hat{\theta}_{y}$ - vectors for each class $y$
$n$ - number of features 
$N_{yi}$ - number of times that features $i$ occur in the training data
$N_y$ - total count for all features in class $y$
$\alpha$ - Considers features not present in training data and prevents zero probabilities; $\alpha>0$

**Bernoulli** (BernoulliNB)
*Description:* Assumes features to have binary values, such as '0 and 1' or 'true or false'. These boolean variables are effectively distributed when key words are missing in text classification, as there are only two possible results. Useful in cases with discrete probability and only one parameter. 
$$P(x_i \mid y) = P(x_i = 1 \mid y) x_i + (1 - P(x_i = 1 \mid y)) (1 - x_i)$$
*Variable meanings:* 
* $P(x_i|y)$ - predicted class
* $i$ - feature value
* $y$ - class value
* $P(x_i=1)$ - probability that trial is a success
* $1-P(x_i=1)$ - probability that trial fails

### **How Naive Bayes Classification Algorithm Works**

Earlier, this article alludes to the idea that *Naive Bayes Classification* is an application that is based on *Bayes Theorem*. Given that, the question still remains how the algorithm uses *Bayes Theorem* to classify data, both mathematically and its real world application. *Naive Bayes Classification* is best known as an algorithm that is highly efficient in classifying large datasets, yet is extremely versatile. 

**Prominent Types of Classification Applications**

* Spam filtering
    * classifies emails as spam or not spam based on key words or phrases
* Text classification
    * takes words and phrases from a text and assigns the features to categories to organize the data
* Sentiment analysis
    * assesses overall attitudes by classifying words and phrases from reviews to determine whether general feelings are positive or negative
* Recommender sytems
    * makes predictions based on user preferences through analyzing data
* Multi-class prediction
    * predicts the correct class from multiple categories through multinomial classification

In these applications, *Bayes Theorem* is used to find the probability by predicting feature values of an unknown class (Awan). Although there are other kinds of data that *Bayes Classification* can work with, these applications are its most common uses. Lets take a dive into how these applications work in practice.

With knowledge of the *types of classifiers* and the *data Naive Bayes classifies*, lets transition into how *real world application* of **Naive Bayes Classification** works using practice scenarios.

## Bayes Classification in Practice

### Scenario #1:  Posterior Probability
![Screenshot 2025-04-24 at 10.34.41 PM](https://hackmd.io/_uploads/HJsCEddJlx.png)

**Question:** Is a *source* that uses the phrase **"Posterior Probability"** *useful*?

$P(useful|posterior probability)=\frac{P(posterior probability|useful)*P(useful)}{P(posterior probability)}$
$$P(useful|posterior probability)=P(posterior probability|useful)*P(useful)$$
$$=P(posterior|useful)*P(probability|useful)*p(useful)$$
$$=\frac{3}{36}*\frac{7}{36}*\frac{8}{15}$$
$$=.00864$$
**$.00864$ probability of being a *useful source***

$P(unuseful|posterior probability)=\frac{P(posterior probability|unuseful)*P(unuseful)}{P(posterior probability)}$
$$=P(posterior|unuseful)*P(probability|unuseful)*p(unuseful)$$
$$=\frac{1}{31}*\frac{6}{31}*\frac{7}{15}=.00291$$
**$.00291$ probability of being an *unuseful source***

**Conclusion:** A *source* with the phrase *"Posterior Probability"* is more likely to be a **useful source** because it returns the *highest probability*. 

### Scenario #2: Prior and Conditional Probabilities
![Screenshot 2025-04-24 at 10.55.25 PM](https://hackmd.io/_uploads/r1cGqudyxl.png)

**Question:** What is the prior probability of a positive or negative review?

*Prior Probability* $(P(A))$ = $\frac{Count Of Reveiws}{Total Reviews}$

$$P(A)=\frac{Positive}{Total Reviews}=\frac{55}{70}$$
$$P(A)=\frac{Negative}{Total Reviews}=\frac{15}{70}$$

**Question:** What is the likelihood that reviews are positive or negative given food or price were mentioned?

$Likelihood=P(B|A)$

$$P(Food|Positive)=\frac{35}{144}$$
$$P(Food|Negative)=\frac{5}{42}$$
$$P(Price|Positive)=\frac{14}{144}$$
$$P(Price|Negative)=\frac{7}{42}$$

## Bayes Classification Computationally
### Scenario #2 Continued

    # Importing Required Libraries
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report data = {
        'text': [
            'Very friendly service'
            'The dessert was delicious'
            'The food was too expensive'
            'Our waiter was attentive'
            'The music was too loud'
            'Food was worth being expensive'
            'Parking was hard to find'
            'Awesome food and friendly manager'
        ],
        'label': ['positive', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive']
    }
    df2 = pd.DataFrame(data)
    # Initialize CountVectorizer
    vectorizer = Countvectorizer()
    
    # Transform the text data into a bag-of-words model
    X = vectorizer.fit_transform(df2['text'])
    
    # Convert labels to numerical format
    # y = df2['label'].map({'positive': 1, 'negative': 0})
    y = df2['label']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test size=0.2, random_state=42)
    
    # Initialize the Multinomial Naive Bayes Classifier
    model = MultinomialNB()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    print(f"Accuracy: {accuracy_score(y_test, y_pred): .2f}")
    print("\nClassification Report:")
(Patwariraghottam)

**Predict the probability** of the text, '*Very friendly service*' being a *positive review* from scenario #2:

$P(positive|FriendlyService)=\frac{P(FriendlyService|positive)*P(positive)}{P(FriendlyService)}$
$$P(positive|FriendlyService)=P(FriendlyService|positive)*P(positive)$$
$$=P(friendly|positive)*P(service|positive)*p(positive)$$
$$=\frac{26}{144}*\frac{19}{144}*\frac{55}{70}$$
$$=.01872$$
**$.01872$ probability of being a *positive review***

$P(negative|FriendlyService)=\frac{P(FriendlyService|negative)*P(negative)}{P(FriendlyService)}$
$$P(negative|FriendlyService)=P(FriendlyService|negative)*P(negative)$$
$$=P(friendly|negative)*P(service|negative)*p(negative)$$
$$=\frac{2}{42}*\frac{6}{42}*\frac{15}{70}$$
$$=.00146$$
**$.00146$ probability of being a *negative review***

**Result:** A review with the text **'Very friendly service'** is far more likely to be a **positive review** as the *posterior probability* of being a positive review is $.01872$, much higher than $.00146$, the probability of being a negative review.

## Advantages and Limitations
### Advantages
* High speed and simple classification process
* **Extremely versatile algorithm** 
    * works with different kinds of data such as continuous and discrete as well as with large data sets
    * wide range of scalability regarding features
    * numerous classification models
* Low computational costs, while maintaining accuracy and efficiency

### Limitations
* **Conditional independence assumption** is virtually impossible to hold true with real-world data
* Cannot make predictions on a value in the test dataset that is not observed in the training dataset, creating the **zero frequency problem**, in which the probability of a given feature is zero

## Conclusion
All in all, **Naive Bayes Classification** is a fascinating machine learning algorithm. It is highly effective and useful in classifying data. Although its independence assumption is an obvious limitation when working with real-world data, the classifiers unique level of versitality among the kinds of data it can work with should not be overlooked. Overall, despite having an evident weakness, the strengths that *Naive Bayes classification* has to offer make it a widely used algorithm that successfully and effectively classifies data in real-world scenarios. 

## Sources
* Awan, A. A., & Navlani, A. (2023, March 3). Naive Bayes Classification Tutorial using Scikit-Learn. DataCamp. https://www.datacamp.com/tutorial/naive-bayes-scikit-learn 
* Gamal, B. (2020, December 17). Naïve Bayes algorithm. Medium. https://medium.com/analytics-vidhya/na%C3%AFve-bayes-algorithm-5bf31e9032a2 
* Patwariraghottam. (2024, August 22). Mastering Naive Bayes: A Comprehensive Python Guide to Probabilistic Classification. Medium. https://medium.com/@patwariraghottam/mastering-naive-bayes-a-comprehensive-python-guide-to-probabilistic-classification-b7fe67c6763f 
* scikit-learn developers. (2025). 1.9. naive Bayes. scikit learn. https://scikit-learn.org/stable/modules/naive_bayes.html 
* What are naïve Bayes classifiers?. ibm.com. (2025, April). https://www.ibm.com/think/topics/naive-bayes#:~:text=Na%C3%AFve%20Bayes%20is%20part%20of,important%20to%20differentiate%20between%20classes. 