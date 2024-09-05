
import os
import utils
import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 
import random



def classify_document(document, vocabulary, use_logarithm, first_multiplicand, denom_of_second_multiplicand_pos, denom_of_second_multiplicand_neg, frequency_of_words_positive, frequency_of_words_negative, alpha):

    vocabulary = list(vocabulary)

    # Calculate the probability the document is a positive review:
    prob_pos = calc_prob_pos_doc(document, vocabulary, use_logarithm, first_multiplicand, denom_of_second_multiplicand_pos, frequency_of_words_positive, alpha)

    # Calculate the probability the document is a negative review:
    prob_neg = calc_prob_neg_doc(document, vocabulary, use_logarithm, first_multiplicand, denom_of_second_multiplicand_neg, frequency_of_words_negative, alpha)

    # If the probability the document is a positive review is greater, classify the document as positive
    if prob_pos > prob_neg:
        return "positive"
    # If the probabilities are equal, randomly choose one classification.
    elif prob_pos == prob_neg:
        if random.randint(0, 1) == 0:
            return "positive"
        else:
            return "negative"
    # Otherwise, classify the document as a negative review.
    else: 
        return "negative"


def calc_prob_pos_doc(document, vocabulary, use_logarithm, first_multiplicand, denom_of_second_multiplicand_pos, frequency_of_words_positive, alpha):

    # Calculate the probability the document is a positive review:
    # Pr(pos) * product(Pr(unique word | pos))
    # Pr(pos) = Num of positive Docs / total Num of Docs
    prob_pos = first_multiplicand

    # Pr(unique word | pos) = Num of occurrences of unique word in positive Docs / total number of words that appear in positive Docs 
    product = (float)(1)
    summation = (float)(0)
    total_words_positive_docs = denom_of_second_multiplicand_pos
    unique_words_in_doc = set(document)
    for word in unique_words_in_doc:
        
        # Look up the number of times the word occurs in the Counter that stores all the respective frequencies.
        word_occurrences = frequency_of_words_positive[word]
        word_probability = (float)(word_occurrences) / (float)(total_words_positive_docs)

        if use_logarithm:
            # Use a very small constant to ensure no errors with log0.
            if word_probability == 0:
                word_probability = (float)(alpha) / (float)(total_words_positive_docs)
            summation += math.log(word_probability)
        else:
            product *= word_probability
    
    
    if use_logarithm:
        return math.log(prob_pos) + summation
    else:
        return prob_pos * product

def calc_prob_neg_doc(document, vocabulary, use_logarithm, first_multiplicand, denom_of_second_multiplicand_neg, frequency_of_words_negative, alpha):

    # Calculate the probability the document is a negative review:
    # Pr(neg) * product(Pr(unique word | neg))
    # Pr(neg) = Num of negative Docs / total Num of Docs
    prob_neg = first_multiplicand

    # Pr(unique word | neg) = Num of occurrences of unique word in negative Docs / total number of words that appear in negative Docs
    product = (float)(1)
    summation = (float)(0)
    total_words_negative_docs = denom_of_second_multiplicand_neg
    unique_words_in_doc = set(document)
    for word in unique_words_in_doc:
        # Find the number of occurrences of unique word in negative Docs
        word_occurrences = frequency_of_words_negative[word]

        word_probability = (float)(word_occurrences) / (float)(total_words_negative_docs)

        if use_logarithm:
            # Use a very small constant to ensure no errors with log0.
            if word_probability == 0:
                word_probability = (float)(alpha) / (float)(total_words_negative_docs)
            summation += math.log(word_probability)
        else:
            product *= word_probability
    
    
    if use_logarithm:
        return math.log(prob_neg) + summation
    else:
        return prob_neg * product



def setup(is_training, positive_docs, negative_docs, use_logarithm, use_laplace, alpha):
    # Preprocessing -- uncomment when submitting
    # Iterate through each file in the positive folder and pre-process using preprocess_text() in utils.py
    for filename in os.listdir(positive_docs):
        if filename.endswith('.txt'):
            document_path = os.path.join(positive_docs, filename)
            with open(document_path, 'r', encoding='utf8') as file:
                document_content = file.read()
                document_content = utils.preprocess_text(document_content)
                new_file_content = ""
                for word in document_content:
                    new_file_content += word + " "
                with open(document_path, 'w') as file:
                    file.writelines(new_file_content)


    # Iterate through each file in the negative folder and pre-process using preprocess_text() in utils.py
    for filename in os.listdir(negative_docs):
        if filename.endswith('.txt'):
            document_path = os.path.join(negative_docs, filename)
            with open(document_path, 'r', encoding='utf8') as file:
                document_content = file.read()
                document_content = utils.preprocess_text(document_content)
                new_file_content = ""
                for word in document_content:
                    new_file_content += word + " "
                with open(document_path, 'w') as file:
                    file.writelines(new_file_content)

    
    # Load documents to use as training set using load_training_set(%pos from training set, %neg from training set) from utils.py
    training_pos_docs, training_neg_docs, vocab_training = utils.load_training_set(0.2, 0.2)  # Set reference to data structures representing positive + negative documents

    # Test the training set on the training set
    if is_training:
        to_be_classified_pos_docs = training_pos_docs  # Set reference to data structures representing positive documents
        to_be_classified_neg_docs = training_neg_docs  # Set reference to data structures representing negative documents
    # Otherwise, test the testing set on the training set
    else: 
        # Load documents to use as testing set using load_test_set(%pos from testing set, %neg from testing set) from utils.py
        to_be_classified_pos_docs, to_be_classified_neg_docs = utils.load_test_set(0.2, 0.2)  # Set reference to data structures representing positive & negative documents

    # List of classifications is made of tuples described by: 
    # (original class, identified class)
    classifications = []


    # Set number of positive documents, negative documents, and total documents
    num_pos_docs = len(training_pos_docs)
    num_neg_docs = len(training_neg_docs)
    total_num_docs = num_pos_docs + num_neg_docs


    # First multiplicand to calculate probabilities should be the same, done here to avoid recomputation.
    first_multiplicand_positive = (float)(num_pos_docs) / (float)(total_num_docs)
    first_multiplicand_negative = (float)(num_neg_docs) / (float)(total_num_docs)

    # Generate array of all words from all positive documents. Do the same for negative documents.
    flat_training_pos_docs = [item for sublist in training_pos_docs for item in sublist]
    flat_training_neg_docs = [item for sublist in training_neg_docs for item in sublist]

    # Precompute these values since it will be same across pos and neg docs respectively.
    frequency_of_words_positive = Counter(flat_training_pos_docs)
    frequency_of_words_negative = Counter(flat_training_neg_docs)
    denom_of_second_multiplicand_pos = sum(frequency_of_words_positive.values())
    denom_of_second_multiplicand_neg = sum(frequency_of_words_negative.values())

    
    if use_laplace:
        vocab_training = set()
        for key in frequency_of_words_positive.keys():
            vocab_training.add(key)
        for key in frequency_of_words_negative.keys():
            vocab_training.add(key)
        smoothing = alpha * len(vocab_training)
        denom_of_second_multiplicand_pos = sum(frequency_of_words_positive.values()) + smoothing
        denom_of_second_multiplicand_neg = sum(frequency_of_words_negative.values()) + smoothing
        frequency_of_words_positive = Counter({key: value + alpha for key, value in frequency_of_words_positive.items()})
        frequency_of_words_negative = Counter({key: value + alpha for key, value in frequency_of_words_negative.items()}) 
        
        
    for pos_doc in to_be_classified_pos_docs:
        classification = classify_document(pos_doc, vocab_training, use_logarithm, first_multiplicand_positive, denom_of_second_multiplicand_pos, denom_of_second_multiplicand_neg, frequency_of_words_positive, frequency_of_words_negative, alpha)
        classifications.append(("positive", classification))
    
    for neg_doc in to_be_classified_neg_docs:
        classification = classify_document(neg_doc, vocab_training, use_logarithm, first_multiplicand_negative, denom_of_second_multiplicand_pos, denom_of_second_multiplicand_neg, frequency_of_words_positive, frequency_of_words_negative, alpha)
        classifications.append(("negative", classification))
    
    return classifications

def accuracy(classifications):
    correct = 0
    total = 0
    for item in classifications:
        if item[0] == item[1]:
            correct += 1
        total += 1
    
    if total == 0:
        return 0
    return (float)(correct) / (float)(total)

def get_confusion_matrix(classifications):
    tp_count = 0
    fn_count = 0
    fp_count = 0
    tn_count = 0
    for item in classifications:
        true_class = item[0]
        predicted_class = item[1]
        if true_class == predicted_class:
            if true_class == "positive":
                tp_count += 1
            else:
                tn_count += 1
        else:
            if true_class == "positive":
                fn_count += 1
            else:
                fp_count += 1
        
    return (tp_count, fn_count, fp_count, tn_count)


def precision(classifications):
    predicted_positive = 0
    true_positive = 0
    for item in classifications:
        if item[1] == "positive":
            predicted_positive += 1
            if item[0] == "positive":
                true_positive += 1
    
    if predicted_positive == 0:
        return 0
    return (float)(true_positive) / (float)(predicted_positive)

def recall(classifications):
    correct_prediction = 0
    positive_instances = 0
    for item in classifications:
        if item[0] == "positive":
            positive_instances += 1
            if item[1] == "positive":
                correct_prediction += 1
    
    if positive_instances == 0:
        return 0
    return (float)(correct_prediction) / (float)(positive_instances)

def show_graph(training_points, testing_points):

    x = np.array([coord[0] for coord in training_points])
    y = np.array([coord[1] for coord in training_points])

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    plt.xlabel('Alpha (log scale (e))')
    plt.ylabel('Accuracy')
    #ax0.set_ylim(0.72, 1.0)
    ax0.errorbar(x, y, yerr=None, fmt='-o')
    ax0.set_title('Accuracy of Training Set based on alpha in LaPlace Smoothing')
    
    m = np.array([coord[0] for coord in testing_points])
    n = np.array([coord[1] for coord in testing_points])

    plt.xlabel('Alpha (log scale (e))')
    plt.ylabel('Accuracy')
    #ax1.set_ylim(0.72, 1.0)
    ax1.errorbar(m, n, yerr=None, fmt='-o')
    ax1.set_title('Accuracy of Testing Set based on alpha in LaPlace Smoothing')


    plt.show()



def confusion_matrix(tup):
    tp, fn, fp, tn = tup
    tp_string = "TP: " + str(tp)
    fn_string = "FN: " + str(fn)
    fp_string = "FP: " + str(fp)
    tn_string = "TN: " + str(tn)
    labels = [" ", "  +  ", "  -  ", "+", tp_string, fn_string, "-", fp_string, tn_string]


    table = [[None, None, None],
             [None, None, None],
             [None, None, None]]


    for i in range(3):
        for j in range(3):
            table[i][j] = labels[i*3 + j]

    # Print the table
    for row in table:
        print("|", end="")
        for item in row:
            print(f" {item} |", end="")
        print()

if __name__ == "__main__":

    positive_training_docs = 'train/pos'
    negative_training_docs = 'train/neg'

    positive_testing_docs = 'test/pos'
    negative_testing_docs = 'test/neg'

    # Choose whether you want to use logs to classify instances by performing the log-transformation trick.
    use_logarithm = True

    # Choose whether you want to use LaPlace Smoothing.
    use_laplace = True

    graph_results = True

    if use_laplace and graph_results:
        alpha = 0.0001
        training_points = []
        while alpha <= 1000:
            training_classifications = setup(True, positive_training_docs, negative_training_docs, use_logarithm, use_laplace, alpha)
            accuracy_training = accuracy(training_classifications)
            training_points.append((math.log(alpha), accuracy_training))
            alpha *= 10

        testing_points = []
        alpha = 0.0001
        while alpha <= 1000:
            testing_classifications = setup(False, positive_testing_docs, negative_testing_docs, use_logarithm, use_laplace, alpha)
            accuracy_testing = accuracy(testing_classifications)
            testing_points.append((math.log(alpha), accuracy_testing))
            alpha *= 10
        
        show_graph(training_points, testing_points)
    else:
        alpha = 1
        training_classifications = setup(True, positive_training_docs, negative_training_docs, use_logarithm, use_laplace, alpha)
        print("Accuracy of training set: ", accuracy(training_classifications))
        tup = get_confusion_matrix(training_classifications)
        print("Confusion matrix of training set: ", tup)
        confusion_matrix(tup)
        print("Precision of training set ", precision(training_classifications))
        print("Recall of training set: ", recall(training_classifications))

        testing_classifications = setup(False, positive_testing_docs, negative_testing_docs, use_logarithm, use_laplace, alpha)
        print("Accuracy of testing set: ", accuracy(testing_classifications))
        tup = get_confusion_matrix(testing_classifications)
        print("Confusion matrix of testing set: ", tup)
        confusion_matrix(tup)        
        print("Precision of testing set ", precision(testing_classifications))
        print("Recall of testing set: ", recall(testing_classifications))




