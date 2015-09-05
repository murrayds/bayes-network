__author__ = "Dakota Murray"

# This python script implements a naive bayes network which is able to classify
# a set of facebook posts as either "Action", "Information", or "Community" with a
# reasonable degree of accuracy.
#
# The file train.csv contains the messages to be used as training data
# The file test.csv contains the messages to be used during the testing phase
#
# I will state that this is not 100% polished, it is simply that I am out of time to work
# on it before the posted due date, and need to submit what I have. Despite my struggles I never quite
# got the log-likelihood calculations working in a way that I saw clean and elegant. I also realize that
# some values are slightly greater than one, a circumstance that I don't quite understand but am attributing to
# mild errors in my calculations.
#
# Despite these issues, the program will predict the test data with about a 66% accuracy, which is better than
# chance. It will also offer a basic visualization of the data however I could not figure out how to draw that
# nice looking decision boundary which was displayed on the provided example plot on the assignment.
from collections import defaultdict
from decimal import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv


#
# Runs the program. Trains the data, performs
#
def main():
    print("Training...")
    (messages, categories) = parse("train.csv")
    model = train(messages, categories)
    num_correct = 0

    print("Testing......")
    (messages, categories) = parse("test.csv")
    predictions = []
    predicted_labels = []
    for i in range(len(messages) - 1):
        message = messages[i].replace('\\n', ' ')
        category = categories[i]

        prediction = predict(message, model)
        classification = max(prediction, key=prediction.get)
        predictions.append(prediction)
        # print("Classified as: %s, Full Prediction: %s" % (classification, prediction))
        predicted_labels.append(classification)
        if classification == category:
            num_correct += 1

    percentage_correct = num_correct / float(len(messages))
    print("Predicted %d labels correct out of a total of %d messages" % (num_correct, len(messages)))
    print("Percentage Predictions Correct = %f%%" % (percentage_correct * 100))
    visualize(predictions, predicted_labels)


#
# Trains the priors and term likelihoods of the naive ayes text classifier. Takes as input
# two lists of size N, where the messages list contains the list of all messages to train with,
# and the categories list contains a list where position i of categories is the category which
# messages[i] belong to
#
# Returns a model which contains the data necessary for the predict function. The model is a
# a list of values containing the set of trained priors, term likelihoods, and the counts of
# the words in each category.
#
def train(messages, categories):
    # calculate total_count
    words = defaultdict(lambda: defaultdict(lambda: 0))
    category_counts = defaultdict(lambda: 0)
    for i in range(len(messages) - 1):
        message = messages[i]
        category = categories[i]
        for term in message.split():
            category_counts[category] += 1
            words[term][category] += 1

    total_count = 0
    for category in category_counts.keys():
        total_count += category_counts[category]
    num_categories = len(category_counts)
    num_unique_words = len(words)

    priors = defaultdict(lambda: 0)
    for category in categories:
        priors[category] = prior(category_counts[category], total_count, num_categories)

    term_likelihoods = defaultdict(lambda: defaultdict(lambda: 0))
    for word in words:
        for cat in categories:
            term_likelihoods[word][cat] = likelihood(words[word][cat], category_counts[cat], num_unique_words)

    model = [priors, term_likelihoods, category_counts]
    return model


# Parses the provided .csv file and returns two dictionaries, one containing the count of a word as it appears
# in each category, and another containing the count of categories as they appear in the .csv file
#
# Simply takes as input the filename of the .csv file to parse. Each row of the .csv file should be in the
# following format:
#
#     ["<category>", "<message>"]
#
# Returns two lists, one which contains the list of messages, and one which contains the list of categories
# where each index, i, of the categories list is the category which messages[i] belongs to
#
def parse(filename):
    messages = []
    categories = []

    with open(filename, 'r', encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # skip headers on the first line
        next(reader)
        for row in reader:
            messages.append(row[1].replace('\\n', ' '))
            categories.append(row[0])

    return messages, categories


#
# Calculates the prior probability of an item appearing in a category before
# knowing any details about that item. For example: if given three categories:
# [A, B, C], this function would calculate P(A), P(B), or P(C), depending on
# the passed parameters.
#
# - item_count is the number of items in the category
# - total_count is the total number of items in all categories
# - category_count is the count of possible categories
#
def prior(item_count, total_count, category_count):
    return (item_count + 1) / (total_count + category_count)


#
# Calculates and returns the probability of observing the words in a message given that the
# message is in a particular category.
#
# P("Thank You Everyone" | C) = P("Thank" | C) * P("You" | C) * P("Everyone" | C)
#
# - word_count_in_cat is the how many times the word being queried appears in the category being queried
# - total_words_in_category - is the total number of words appearing in the queried category
# - num_unique_words - is the total number of unique terms in the dictionary of words
#
def likelihood(count_word_in_category, total_words_in_category, num_unique_words):
    return (count_word_in_category + 1) / (total_words_in_category + num_unique_words)


#
# Calculates and returns the probability of message and category, written as follows:
#
#   P(message, category)
#
#   P(message, A) = P(‘thank’|A) P(‘you’|A)P(‘everyone’|A)P(A)
#
# This unction will typically only be called as part of the total probability calculation
#
# - message is the message to calculate P(message, category) of
# - category is the category involved in the calculation of P(message, category)
# - words is the dictionary containing the counts of words in each category
# - categories is the dictionary containing the number of messages and categories for each message category
#
def probability_message_and_category(message, category, category_counts, priors, term_likelihoods):
    multiplier = Decimal(1)
    for term in message.split():
        if term_likelihoods[term][category] == 0:
            multiplier = multiplier * Decimal(likelihood(0, category_counts[category], len(term_likelihoods.keys())))
        else:
            multiplier = multiplier * Decimal((term_likelihoods[term][category]))

    return multiplier * Decimal(priors[category])


#
# Calculates the total probability of a message, written as follows:
#
#   P(message)
#
#   P(message) = P(message, A) + P(message, C) + P(message, I)
#
# - message is the message involved in the total probability calculation
# - words is the dictionary containing the counts of words in each category
# - categories is the dictionary containing the number of messages and categories for each message category
#
def total_probability(message, category_counts, priors, term_likelihoods):
    total_probability_sum = 0
    for category in category_counts.keys():
        total_probability_sum += probability_message_and_category(message, category, category_counts, priors,
                                                                  term_likelihoods)

    return total_probability_sum


#
# Calculates the posterior probability, which is the probability that a given message appears in a category
#
#   P(Action | message) = P(message | Action) P(Action) / P(message)
#
#
def posterior(message, category, category_counts, priors, term_likelihoods):
    numerator = Decimal(1)
    for term in message.split():
        # handle the case where the term is not in the dictionary of term likelihoods
        if term_likelihoods[term][category] == 0:
            numerator = numerator * Decimal(likelihood(0, category_counts[category], len(term_likelihoods.keys())))
        else:
            numerator = numerator * Decimal(term_likelihoods[term][category])

    numerator = numerator * Decimal((priors[category]))

    return numerator / total_probability(message, category_counts, priors, term_likelihoods)


#
# returns a prediction in the form:
#
#  d = {
#      ‘Action’: 0.03,
#      ‘Community’: 0.91,
#      ‘Information’: 0.06
#   }
#
# Takes an input a set of messages to perform predictions on, as well as the model
# produced from the training phase which contains information about priors and term
# likelihoods.
#
def predict(message, model):
    # unpack the model
    priors = model[0]
    term_likelihoods = model[1]
    category_counts = model[2]

    prediction = {}
    for category in category_counts.keys():
        probability = posterior(message, category, category_counts, priors, term_likelihoods)
        prediction[category] = probability

    return prediction


#
# Creates a basic scatter plot to help visalize the data from the algorithm
#
def visualize(prediction_list, predicted_labels):
    x = []
    y = []
    colors = []
    for i in range(len(prediction_list) - 1):
        x.append(prediction_list[i]["Action"])
        y.append(prediction_list[i]["Community"])
        if predicted_labels[i] == "Action":
            colors.append("r")
        elif predicted_labels[i] == "Community":
            colors.append("g")
        else:
            colors.append("b")

    plt.scatter(x, y, c=colors)
    plt.title("Naive Bayes Network")
    plt.xlabel("Actions")
    plt.ylabel("Community")
    actions = mpatches.Patch(color='red', label='Actions')
    community = mpatches.Patch(color='green', label='Community')
    information = mpatches.Patch(color='blue', label='Infomraiton')
    plt.legend(handles=[actions, community, information])
    plt.show()


#
if __name__ == "__main__":
    main()
