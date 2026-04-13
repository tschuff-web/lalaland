# _La La Land_ - Using My Apple Music Data to Predict My Listening Eras

Thomas Schuff  
Professor MacIsaac  
CPSC 222 - Data Science  
5/6/26

---

## Project Description

The main idea of this project is to divide my listening history into different
_eras_, or distinct periods where my taste, mood, or habits were similar, before
changing or shifting.

These "eras" could be defined by several factors including:

- average song BPM?
- song duration
- release year / nostalgia
- genre
- genre diversity
- artist diversity
- artist concentration
- skip rate
- listening duration
- time-of-day listening
- mood categories?

Then, I will train a classifeier to predict which era a given time period (day,
week, month, etc.) belongs to, based on my listening activity during that
period.

The class label (what I will predict) will be the era label (Indie Era, Study
Era, etc.)

The two tables I will join are:

1. My Apple Music Listening History
2. Song Metadata

And these tables will be joined by the song/album ID

For my exploratory data analysis, I could include several visuals including:

- Rolling BPM trendline
- Genre proportions over time
- Artist dominance/preference cycles
- Heatmap of listening by hour of day
- Skip rate by hour of day
- Cluster visualization of daily listening profiles
- Era-by-Era comparison plots

For my hypothesis tests, these are some ideas:

- Average BPM differs between eras
- Genre proportions differ between era
- Skip rate changes across eras

For my classification models, I could use a _kNN_ model for my numeric
attributes like BPM, skip rate, listening duration, etc. but a _Decision Tree_
model would be better for my categorical data like genre, artists, etc.
