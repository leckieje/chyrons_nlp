# chyrons_nlp

<p align="center">
  <img width="800" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/achilles_injures_heel.png">
</p>


The television news chyron is an important, if often overlooked, element of our media diets. Named after the wise and learned centaur Chiron from Homer's Iliad (the 'i' changed to a 'y' to avoid a patent dispute) who taught a young Achilles how to shoot a bow and play the lyre, the banners that grace the lower third of our television screens are often the first piece of information we receive. If we find ourselves in a gym or noisy bar, it may be the only information received. In the same way Chiron helped develop a young mind and frame the world Achilles was about to enter, news chyrons literally and figuratively frame the public debate, subtly influencing our perceptions and beliefs. The goal of this project is to apply natural language processing and predictive modeling to a dataset of chyrons labeled with the originating network to analyze any differences and similarities in how the chyrons are written.
 
 
## The Data 

<img align="right" width="250" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/thirdEye.png">

<img align="right" width="175" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/archive_logo.jpeg">

For data, I turned to the [Internet Archive](https://archive.org/about/) and its [Third Eye](https://archive.org/services/third-eye.php) project built by TV architect Tracey Jaquith. Launched in late 2017, the Third Eye captures chyrons for four major cable news networks, including BBC, CNN, Fox News and MSNBC. Building a script to utilize the site's simple API (included in src/), I downloaded more the 2.6 million chyrons from all four networks between September 7, 2017 and March 16, 2021. The dataset had been slightly filtered by the Internet Archive for use in its Twitter feeds and limited to 60 second gaps between entries. Features included the time the chyron appeared, its duration, and of course the network and text of the chyron.

For the purposes of this project, I chose to focus on extremes and thus limited the data I would use for analysis and modeling to chyrons from Fox News and MSNBC. Likewise, I chose to focus on a contentious period in our domestic politics between March 25, 2020, the day George Floyd was murdered in Minneapolis, and November 3, 2020 the day of the presidential election between Joe Biden and Donald Trump. After filtering for the two networks and within the date window, I was left with slightly less than 200,000 chyrons.

It is important to note that the Third Eye's optical character recognition software is not perfect. As you can see in the image from their website below, there are often misspellings, such as 'mic' being read as 'nic'. Similarly, given the brevity of chyrons, abbreviations are often used. Here 'WH' is a stand in for 'White House', but others were found as well, including 'pres' for 'president' and 'rep' for 'representative'.

---

## Word Use Frequency 

<img align="right" width="475" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/msnbc_worduse_bar.png">

<img align="right" width="475" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/fox_worduse_bar.png">

Some of the most illuminating insights during EDA came from charting word use frequencies at each network. In the charts to the right, you can see a few of the most significant differences in word frequencies. For example, on MSNBC you are much more likely to see the words `'virus', 'cases', 'covid', 'nyt'` and `'harris'` than on Fox News. Alternatively, you are much more likely to see the words `'president trump', 'police', 'violence', 'media'` and `'left'` on Fox News compared with MSNBC. Later, I will show the results of a Naive Bayes model used to make predictions about which network ran which chyron, but already we are starting to see some separation in chyron vocabulary from each network

I also chose to look at shared words among the networks. While shared words may not give insight into how the Naive Bayes model is making decisions, it does shed some light on the issues and people that most dominate news coverage. To try and glean these insights, I charted the networks against each other with Fox News on the y-axis and MSNBC on the x-axis. The points represent a single word. Of the 2,282 words in my vocabulary, the vast majority clustered into the bottom left hand corner of the chart. Many of these words appeared more than 5,000 times on both networks, but visually still clustered near words much closer to the origin. This was due to the overwhelming popularity of a single word at both networks: `trump`. That `trump` was the most popular word at each network wasn't terribly surprising. He is the incumbent president in the middle of a re-election campaign. But the degree to which the term dominated the word counts at both networks did surprise. For comparison, `biden`, which was the second most popular word, appeared 39,575 fewer times on MSNBC and 13,268 fewer times on Fox News.

I had also initially been surprised to see how much more the term `trump` appeared on MSNBC than on Fox News. After further investigation, I found that, in general, Fox News chyrons tended to be shorter than MSNBC's. In addition, Fox News also used the terms `president` alone and `president trump`together when referring to the eponymous man while MSNBC almost exclusively referred to him only as `trump`.

<p align="center">
  <img width="600" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/shared_words.png">
</p>

---

## The Model 

<img align="right" width="475" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/final_roc_NB_bigrams.png">

As mentioned, I ran a Naive Bayes model to predict which network ran each chyron. In the text processing pipeline, I ultimately settled on the use of single words and bigrams. I also set the `fit_prior` hyperparameter in SKLearn's `MultinomialNB()` to `False`. This meant the prior used to calculate probabilities was set to 50/50 given two targets. Although my targets were fairly well balanced, 56% for MSNBC to 44% for Fox News, setting the uniform prior greatly improved the model's accuracy and helped solve earlier problems with a model that only predicted one class.

In the end, with these parameters, the model performed with roughly 77% accuracy.

Since the goal of this project was analysis and not only prediction, I dug a little deeper into the model's performance and wrote a script that would return the full text of the chyron along with the predicted probabilities generated by the model. The model was absolutely certain the chyron `new lousiville law would change body camera policy after breonna taylor killed in own home` came from MSNBC. Likewise, it was certain Fox News ran the chyron `fired minneapolis officer involved in floyd's death arrested`.

What is interesting about this classification is that it aligns with popular opinions about these networks. In the MSNBC chyron, Breonna Taylor is an innocent victim killed in her own home. As a result of this tragedy, a new law is created that holds law enforcement to higher standards of accountability. On the other hand, the Fox News chyron places the police officer as the sentence's subject, his victim as its object. Further, sympathy is built for the police officer as he has been both `fired` and `arrested`. The alleged crime is also downplayed as a `death` (if one can downplay a death) opposed to other words such as 'murder' which would accurately describe the crime for which the officer was arrested.

<p align="center">
  <img width="800" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/msnbc_chyron.png">
</p>

<p align="center">
  <img width="800" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/fox_news_chyron.png">
</p>

---

## The Future

As a first pass ad given its time constraints, this project largely served its purpose but more could certainly be done. Among the ambitions for further investigation would be better/more robust language processing system to deal with many of the irregularities generated by the Third Eye process that includes misspellings and special characters like breaks. The process also occasionally picks up text from the ticker that appears below the chyron that could potentially be eliminated.

Second, would be to use more of the data gathered from the Internet Archive's API. This projected used less than 8% of the total data that was accessed. Part of the restriction and the decision to narrow the time period window was due to local hardware limitations. But were those limitations not present, the robustness the model, and thus its usefulness, could potentially be increased.

Finally, I would be interested in spending more time analyzing the full text of chyrons based on the model's predicted probabilities. The single pair of chyrons analyzed above provided useful insights in how each network frames the stories crucial to our civil discourse. Following the process further with additional chyrons combined with knowledge of how Naive Bayes works, could provide a more evidence-based approach the critical textual analysis of chyrons that was the impetus for this project.


## The Tech

<p align="center">
  <img width="400" src="https://github.com/leckieje/chyrons_nlp/blob/main/imgs/tech_logos.png">
</p>


#### Contact:

leckieje@gmail.com
