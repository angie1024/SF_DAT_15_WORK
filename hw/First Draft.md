**July 22 (Mandatory) First Draft Due**

**Components that should be covered in my paper**
Problem statement and hypothesis
* Where can I find the best cheap tacos in the bay area?
Description of your data set and how it was obtained
* I spent a good amount of time trying to understand how to attain data from Yelp.
* There are requirements for using yelp-api and one of them is to include their logo so the audience will know the data came from Yelp.
* Yelp has an API for developers to gather data from their site.
* Yelp has a repository on github called yelp-api. I have forked it over to my repo and pulled the sample python code provided.
* I am running the codes provided but haven't fully grasped what the codes are doing or mean.
* The next steps to take:
		*how to run the python codes provided by yelp sample and add comments of what each code is doing.
		*figure how to filter for the factors I have mentioned in my Question and Data Set.
		*clean the data to have it ready for analysis. The nltk we learned Monday will be very helpful in this step.
		*So far, I am still in the data obtaining phase of this project.
Description of any pre-processing steps you took
What you learned from exploring the data, including visualizations
How you chose which features to use in your analysis
Details of your modeling process, including how you selected your models and validated them
Your challenges and successes
Possible extensions or business applications of your project
Conclusions and key learnings



**Below is what I submitted for my Question and Data Set**
I’m trying to find out where I can find the best cheap tacos in the bay area.

I plan to scrape data from yelp.

Yelp is a pretty useful tool with it’s filter functions but maybe we can be even more specific:
Only consider restaurants with four and up star rating.
Users leave comments regarding how delicious food is at the restaurant which I will have to clean and exclude every food except tacos. Keyword filtering.
Have to clean out the users who are not trustworthy.
Maybe single out users who frequent taco restaurants often.
View only restaurants with one ‘$’ symbol.
Set a minimum number to reviews requirement for restaurants.
Only consider authentic tacos. (No fusion)
Restaurants only. (No food trucks)

I chose this topic because I frequent different areas in the bay and I tend to crave for tacos when someone asks what I would like to eat for lunch/dinner. This topic will be useful to everybody.


**Comments from Sinan**
This is interesting!

Perhaps you could even:
1. make a map of the best tacos in python
2. Identify words that people use when naming great tacos
3. try to predict if a taco place will be good or bad based on comments/words used/location/etc