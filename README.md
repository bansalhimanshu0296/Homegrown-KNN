# Homegrown-KNN

This project was done as a part of CSCI-B-551 Elements of Artificial Intelligence Coursework under Prof. Dr. David Crandall.

## Command to run the program ##

python3 main.py knn

## Overview of Solution ##

In this solution firstly, I fitted the data to model which was easiest task in knn, just set class varable _X and _y with all the training data and their labels respectively. In the predict function of knn we just had to find the distance of a particular point from all the test data and then we take the closest neigbours depending on the number of nearest neighbours we want to use to predict. Based on weight each neighbour is given a specific voting power to vote for the class they belong to. If weights are uniform then all neighbours have voting power of 1 else the voting power of each neigbour is inversely proportional to distance of it from test data. In last the class with highest votes is predicted class for the test data. In similar way, I implemented the predict function for knn in the k_nearest_neighbour class. First i took an empty list to store the class for each test data as test data was not a single point but a list of points. then i iterated through each point from the test data list. In each iteration i.e. for each point, i caculated the distance of it from each point store the result and its category as a tuple in a list. When calculated the distance from each point a stored in a list, we sorted that list depending upon value of distance. Then i slice the list containing only number of nearest neighbors which we passed in the constructor. Then made an empty dictionary for storing categories and their votes. Then Iterate through list containging neighbour if the weights were uniform then increase the count of category by 1 otherwise meaning if the weights were distance increase the count of category by 1/(distance) for the point. At last found the category with greatest number of count or votes and append that to list for a given test point. After doing all this for each point I am returning list which contain categories for each test data. The above mentioned distance can be calculated by two methods manhattan or eucledian depending what is passed into class. Both methods are implemented in util.py. The eucledian distance has been calculated by already numpy implemented method where as manhatton distance is calculated by using formula for manhattan distance going through each coordinate of points and finding sum of their absolute difference. The k_nearest_neighbour class is called from main.py with different combinations of number of neighbors, type of weights and type of distances, with train and test data for each combination an accuracy is calculated and stored in html file.

## Accuracy ##

Accuracy for my knn model is same as sklearn model.

## Challanges ##

Main challange in this part was to understand the concept of distance weights, first i used a inverse of a number which was increasing exponentially. But then i took the inverse of distasnce directly.
