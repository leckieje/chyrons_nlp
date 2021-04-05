# chyrons_nlp

[Achilles Injures Heel]()

The televison news chyron is an important, if often overlooked, element of our media diets. Named after the wise and learned centaur Chiron from Homer's *Iliad* (the 'i' changed to a 'y' to avoid a patent dispute) who taught a young Achilles how to shoot a bow and play the lyre, the banners that grace the lower third of our television screens are often the first peice of information we recive. If we find ourselves in a gym or noisey bar, it may be the only information recieved. In the same way Chrion helped develop a young mind and frame the world Achilles was about to enter, news chyrons literally and figuratively frame the public debate, subtly influencing our perceptions and beliefs. The goal of thiss project is to apply natural language processing and predictive modeling to a datsest of chyrons labeled with the originating network to analyize any differrnces or simularities in how the chyrons are written. 
 
 
## The Data 

[Internet Archive Logo]()

[Third Eye Logo]()

For data, I turned to the [Internet Archive](https://archive.org/about/) and it's [Third Eye](https://archive.org/services/third-eye.php) project built by TV architect Tracey Jaquith. Launched in late 2017, the Third Eye captures chyrons for four major cable news networks, including BBC, CNN, Fox News and MSNBC. Building a script to utilize the site's simple API (included in `src/`), I downloaded more thhe 2.6 million chyrons from all four networks between September 7, 2017 and March 16, 2021. The dataset had been slightly filtered by the Internet Archive for use in its twitterr feeds and limtied to 60 second gaps between entries. Features included the time the chyron appeared, its duration, and of course the network and text of the chyron. 

For the purposes of this project, I chose to focus on extremes and thus limited the data I would use for anylisis and modeling to chyrons from Fox News and MSNBC. Likewise, I chose to focus on a contencious period in our domesstic politics between March 25, 2020, the day George Floyd was murdered in Minneapolis, and November 3, 2020 the day of the presidential election between Joe Biden and Donald Trump. After filtering for the two networks and within the date window, I was left with slightly less than 200,000 chyrons. 

It is important to note that the Third Eye's optical character recognition software is not perfect. As you can see in the image from their website below, there are often mispelings, such as 'mic' being read as 'nic'. Similarly, given the brevity of chyrons, abrreviations are often used. Here 'WH' is a stand in for 'White House', but others were found as well, including 'pres' for 'president' and 'rep' for 'representative'.

[Third Eye process slide]()

## Word Use Frequency 

Some of the most illuminating insights during EDA came from charting word use frequencies at each network. In the charts here, you can see a few of the most significant differences in word frequecies. For example, on MSNBC you are much more likely to see the words `'virus', 'cases', 'covid', 'nyt'` and `'harris'` appear on the networks chyrons than on Fox News. Alternatively, you are much more likely to see the words `'president trump', 'police', 'violence', 'media'` and `'left'` on Fox News compared with MSNBC. 

Alrerady we arre staring to ssee separation 


## The Model 


## The Future

  * Better/more robust filtering 

  * More data 

## The Tech

