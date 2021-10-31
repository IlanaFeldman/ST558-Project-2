Project 2 - Predictive Models for Popularity of Online News
================
Jasmine Wang & Ilana Feldman
10/31/2021

-   [Introduction](#introduction)
-   [Data](#data)
-   [Exploratory Data Analysis](#exploratory-data-analysis)
    -   [Numerical Summaries](#numerical-summaries)
    -   [Visualizations](#visualizations)
        -   [Correlation Plot](#correlation-plot)
        -   [Boxplot](#boxplot)
        -   [Barplot](#barplot)
        -   [Line Plot](#line-plot)
        -   [Scatterplots](#scatterplots)
        -   [QQ Plots](#qq-plots)
-   [Modeling](#modeling)
    -   [Linear Regression](#linear-regression)
    -   [Random Forest](#random-forest)
    -   [Boosted Tree](#boosted-tree)
-   [Model Comparisons](#model-comparisons)
-   [Final Model](#final-model)

# Introduction

Due to the expansion of online businesses, people can almost do anything
online. With the increasing amount of Internet usages, there has been a
growing interest in online news as well since they allow for an easier
and faster spread of information around the world. Hence, predicting the
popularity of online news has become an interest for research purposes.
Popularity of online news is frequently measured by the number of
interactions in the social networks such as number of likes, comments
and shares. Predicting such measures is important to news authors,
advertisers and publishing organizations. Therefore, the study collected
news articles published between January 7th, 2013 and January 7th, 2015
from different channels on Mashable which is one of the largest online
news sites. 

The study collected a total of 39,000 news articles from six data
channel categories such as lifestyle, entertainment, business,
social media, technology and world. In addition, the features contained
in the articles were also measured to help predict the popularity of the
news contents. Such features include digital media content (number of
images or videos), earlier popularity of news referenced in the article;
average number of shares of keywords, natural language features (title
popularity or Latent Dirichlet Allocation topics) and many others. The
study included 58 predictive attributes, 2 non-predictive attributes and
1 goal field which is the number of shares of the articles. The
collected data was donated by the study to the [UCI Machine Learning
repository](https://archive.ics.uci.edu/ml/datasets/online+news+popularity)
where we downloaded the data.

Table 1 shows the list of variables we used in the analysis and their
descriptions. The study shows after the best predictive model was
selected using a test set, these variables that we are interested in are
among the top ranked features according to their importance in the final
predictive model using the entire data set. Thus, we are going to
investigate their importance in predicting the number of shares using
the predictive models we propose. 

The purpose of the analyses is to compare different predictive models
and choose the best model in predicting the popularity of online news
regarding their features in different channel categories. The methods
implemented in prediction of shares are linear regression models with
different inputs, a random forest model and a boosted tree model. The
optimal model is chosen based on the smallest root MSE value fitting the
test set. More details are in *Modeling* section.

Table 1. Attributes used in the analyses for prediction of online news
popularity

| Index | Attribute                    | Attribute Information                             | Type    |
|-------|------------------------------|---------------------------------------------------|---------|
| 1     | `shares` (target)            | Number of shares                                  | number  |
| 2     | `kw_avg_avg`                 | Average keyword (average shares)                  | number  |
| 3     | `LDA_02`                     | Closeness to LDA topic 2                          | ratio   |
| 4.1   | `weekday_is_monday`          | Was the article published on a Monday?            | boolean |
| 4.2   | `weekday_is_tuesday`         | Was the article published on a Tuesday?           | boolean |
| 4.3   | `weekday_is_wednesday`       | Was the article published on a Wednesday?         | boolean |
| 4.4   | `weekday_is_thursday`        | Was the article published on a Thursday?          | boolean |
| 4.5   | `weekday_is_friday`          | Was the article published on a Friday?            | boolean |
| 4.6   | `weekday_is_saturday`        | Was the article published on a Saturday?          | boolean |
| 4.7   | `weekday_is_sunday`          | Was the article published on a Sunday?            | boolean |
| 5     | `self_reference_avg_sharess` | Average shares of referenced articles in Mashable | number  |
| 6     | `average_token_length`       | Average length of the words in the content        | number  |
| 7     | `n_tokens_content`           | Number of words in the content                    | number  |
| 8     | `n_tokens_title`             | Number of words in the title                      | number  |
| 9     | `global_subjectivity`        | Text subjectivity                                 | ratio   |
| 10    | `num_imgs`                   | Number of images                                  | number  |

``` r
library(tidyverse)
library(knitr)
library(caret)
library(corrplot)
library(ggplot2)
library(gbm)

allnews <- read_csv("../_Data/OnlineNewsPopularity.csv", col_names = TRUE)

########KNIT with parameters!!!!!!!!!channels is in quotes!!!!Need to use it with quotes!!!!!!!!!!!!!!!!!!!!!!!!
channels <- paste0("data_channel_is_", params$channel)
subnews <- allnews[allnews[, channels] == 1, ]

news <- subnews %>% select(
  -data_channel_is_lifestyle, -data_channel_is_entertainment, -data_channel_is_bus, -data_channel_is_socmed, 
  -data_channel_is_tech, -data_channel_is_world, -url, -timedelta)
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

diffday <- news %>% mutate(log.shares = log(shares),
                           class_shares = if_else(shares < 1400, 0, 1),
                           dayweek = if_else(weekday_is_monday == 1, 1,
                                    if_else(weekday_is_tuesday == 1, 2,
                                    if_else(weekday_is_wednesday == 1, 3,
                                    if_else(weekday_is_thursday == 1, 4,
                                    if_else(weekday_is_friday == 1, 5,
                                    if_else(weekday_is_saturday == 1, 6, 7))))))
                           )

sel_data <- diffday %>% select(class_shares, shares, log.shares, dayweek, 
                               kw_avg_avg, 
                               LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
                               weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday,
                               weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday,
                               self_reference_avg_sharess, 
                               average_token_length, 
                               n_tokens_content, n_tokens_title, global_subjectivity, 
                               num_imgs)

set.seed(388588)
sharesIndex <- createDataPartition(sel_data$shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]

train1 <- train %>% select(-class_shares, -shares, 
                           -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                           -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
train1
```

    ## # A tibble: 4,941 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg_sh~ average_token_len~ n_tokens_content n_tokens_title global_subjectiv~ num_imgs
    ##         <dbl>   <dbl>      <dbl>  <dbl>                  <dbl>              <dbl>            <dbl>          <dbl>             <dbl>    <dbl>
    ##  1       6.39       1         0  0.0400                   496                4.68              219             12             0.522        1
    ##  2       7.09       1         0  0.495                      0                4.40              531              9             0.430        1
    ##  3       7.65       1         0  0.215                   6300                4.52              194             14             0.396        0
    ##  4       7.09       1         0  0.0200                  8261.               4.45              161             12             0.572        0
    ##  5       7.09       1         0  0.0225                  1300                5.06              177             12             0.574        1
    ##  6       6.45       1         0  0.0200                  2100                4.47              356              5             0.436       12
    ##  7       7.17       2      1114. 0.324                    951                4.61              281             11             0.434        1
    ##  8       8.76       3       849. 0.0400                  2692.               4.85              241              6             0.518        1
    ##  9       7.00       3       935. 0.258                      0                4.80              349             11             0.439        1
    ## 10       7.31       3       827. 0.0333                  1834.               4.44              464             13             0.426        8
    ## # ... with 4,931 more rows

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# Data

When a subset of data is selected for the entertainment channel articles
which contain 7057 articles, the subset of data is then split into a
training set (70% of the subset data) and a test set (30% of the subset
data) based on the target variable, the number of shares. There are 4941
articles in the training set and 2116 observations in the test set
regarding the entertainment channel. The `createDataPartition` function
from the `caret` package is used to split the data into training and
test sets. We set a seed so that the analyses we implemented are
reproducible.

The data donated by the study contains a “day of the week” categorical
variable but in a boolean format (dummy variable) asking if the article
was published on a day of a week for all seven days which is also shown
in Table 1. Thus, we created a new variable called `dayweek` with seven
levels to combine these dummy variables for the linear regression
models. When `dayweek` = 1, the article was published on a Monday, when
`dayweek` = 2, the article was published on a Tuesday, …, and when
`dayweek` = 7, the article was published on a Sunday.

However, these `dayweek` related variables for each day of the week in
boolean format are needed when we run the ensemble models. In addition,
we classified the articles based on their number of shares into two
categories, a “popular” group when their number of shares is more than
1,400 and an “unpopular” group when their number of shares is less than
1,400. Note that, when we dichotomize a continuous variable into
different groups, we lose information about that variable. We hope to
see some patterns in different categories of shares although what we
discover may not reflect on what the data really presents because we did
not use the “complete version” of the information within the data. This
is purely for data exploratory analysis purpose in the next section.

# Exploratory Data Analysis

The entertainment channel has 4941 articles collected. Now let us take a
look at the relationships between our response and the predictors with
some numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. We classified the number of shares greater than 1400 in a day
as “popular” and the number of shares less than 1400 in a day as
“unpopular”. We can see the number of articles from the entertainment
channel classified into “popular” group or “unpopular” group on
different days of the week from January 7th, 2013 to January 7th, 2015
when the articles were published and retrieved by the study. Note, this
table may not reflect on the information contained in the data due to
dichotomizing the data.

Table 3 shows the average shares of the articles on different days of
the week. We can compare and determine which day of the week has the
most average number of shares for the entertainment channel. Here, we
can see a potential problem for our analysis later. Median shares are
all very different from the average shares on any day of the week.
Recall that median is a robust measure for center. It is robust to
outliers in the data. On the contrary, mean is also a measure of center
but it is not robust to outliers. Mean measure can be influenced by
potential outliers.

In addition, Table 3 also shows the standard deviation of shares is huge
for any day of the week. They are potentially larger than the average
shares. This tells us the variance of shares for any day is huge. We
know a common variance stabilizing transformation to deal with
increasing variance of the response variable, that is, the
log-transformation, which could help us on this matter. Therefore, Table
3 again shows after the log-transformation of shares, the mean values
are similar to their corresponding median values, and their standard
deviations are much smaller than before relatively speaking.

Table 4 shows the numerical summaries of *average keywords* from
entertainment channel in mashable.com on different days of the week.
This table indicates the number of times *average keywords* shown in the
articles regarding the average number of shares, and the table is
showing the average number of those *average keywords* calculated for
each day of the week so that we can compare to see which day of the
week, the *average keywords* showed up the most or the worst according
to the average of shares in the entertainment channel.

Table 5 shows the numerical summaries of average shares of referenced
articles in mashable.com on different days of the week. We calculated
the average number of shares of those articles that contained the
earlier popularity of news referenced for each day of the week so that
we can compare which day has the most or the worst average number of
shares when there were earlier popularity of news referenced in the
entertainmentarticles.

Table 6 checks the numerical summaries of the `global_subjectivity`
variable between popular and unpopular articles, to see if there’s any
difference or a higher variation in subjectivity in popular articles.
Text subjectivity is a value between 0 and 1, so there isn’t any need
for transformation.

Table 7 checks the numerical summaries of the image count per article on
different days of the week, to see if there is a noticeable difference
in image count on weekends versus weekdays across all channels, or only
certain ones. Much like in table 2, the mean is smaller than the
standard deviation for most of the days of the week, and the solution
isn’t as straightforward, since many of the articles don’t have any
images at all. I’ll additionally include a log transformation of
`images + 1` to account for this.

``` r
# contingency table
edadata <- train
edadata$class.shares <- cut(edadata$class_shares, 2, c("Unpopular","Popular"))
edadata$day.week <- cut(edadata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
table(edadata$class.shares, edadata$day.week) %>% kable(caption = "Table 2. Popularity on Day of the Week")
```

|           | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|:----------|-------:|--------:|----------:|---------:|-------:|---------:|-------:|
| Unpopular |    614 |     566 |       576 |      536 |    392 |       97 |    134 |
| Popular   |    355 |     345 |       325 |      323 |    269 |      172 |    237 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   2889.597 |  7029.259 |          1100 |        7.2639 |       0.9370 |           7.0031 |
| Tuesday   |   2877.996 |  7099.666 |          1100 |        7.2498 |       0.9504 |           7.0031 |
| Wednesday |   2693.796 |  6957.247 |          1100 |        7.2173 |       0.9289 |           7.0031 |
| Thursday  |   2654.277 |  5501.523 |          1100 |        7.2530 |       0.9217 |           7.0031 |
| Friday    |   2763.859 |  6564.225 |          1200 |        7.3003 |       0.8834 |           7.0901 |
| Saturday  |   3196.398 |  4899.815 |          1600 |        7.5907 |       0.8478 |           7.3778 |
| Sunday    |   3928.189 |  6540.015 |          1700 |        7.7251 |       0.9212 |           7.4384 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    3138.246 |  1000.8555 |       2986.230 |    964.4435 |
| Tuesday   |    3196.072 |  1047.9397 |       2948.443 |   1092.8238 |
| Wednesday |    3095.430 |   919.6049 |       2924.109 |    977.7699 |
| Thursday  |    3131.609 |   978.9344 |       2946.254 |    983.3106 |
| Friday    |    3149.342 |   888.7727 |       3025.972 |   1010.1086 |
| Saturday  |    3250.077 |  1072.2700 |       3021.647 |   1069.6917 |
| Sunday    |    3212.291 |  1050.1105 |       3028.401 |    906.0124 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      5005.183 |    10446.460 |             2000 |      3800.000 |
| Tuesday   |      5252.047 |     9066.755 |             2100 |      4292.298 |
| Wednesday |      4972.436 |     9763.866 |             2000 |      3776.500 |
| Thursday  |      4859.891 |    10757.457 |             2000 |      3773.750 |
| Friday    |      4483.275 |     8536.953 |             1980 |      3733.333 |
| Saturday  |      5377.535 |    19079.956 |             1850 |      3015.667 |
| Sunday    |      5213.800 |    11704.089 |             2200 |      3868.667 |

Table 5. Summary of Average shares of referenced articles in Mashable on
Day of the Week

``` r
edadata %>% group_by(class.shares) %>% summarize(
  Avg.subjectivity = mean(global_subjectivity), Sd.subjectivity = sd(global_subjectivity), 
  Median.subjectivity = median(global_subjectivity)) %>% 
  kable(digits = 4, caption = "Table 6. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4498 |          0.1112 |              0.4619 |
| Popular      |           0.4549 |          0.1164 |              0.4673 |

Table 6. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 7. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     5.9690 |   10.7461 |             1 |         1.2408 |        1.0790 |            0.6931 |
| Tuesday   |     6.5543 |   11.5622 |             1 |         1.2622 |        1.1429 |            0.6931 |
| Wednesday |     5.3041 |   10.0689 |             1 |         1.1316 |        1.0681 |            0.6931 |
| Thursday  |     6.0221 |   10.4489 |             1 |         1.2352 |        1.0990 |            0.6931 |
| Friday    |     6.0908 |   12.3168 |             1 |         1.1603 |        1.1109 |            0.6931 |
| Saturday  |     6.9591 |   12.6545 |             1 |         1.3676 |        1.1102 |            0.6931 |
| Sunday    |     8.1536 |   13.8876 |             1 |         1.4512 |        1.1846 |            0.6931 |

Table 7. Comparing Image Counts by the Day of the Week

## Visualizations

Graphical presentation is a great tool used to visualize the
relationships between the predictors and the number of shares (or log
number of shares). Below we will see some plots that tell us stories
between those variables.

### Correlation Plot

Figure 1 shows the linear relationship between the variables, both the
response and the predictors, which will be used in the regression models
as well as the ensemble models for predicting the number of shares.
Notice that there may be potential collinearity among the predictor
variables. The correlation ranges between -1 and 1, with the value
equals 0 means that there is no linear relationship between the two
variables. The closer the correlation measures towards 1, the stronger
the positive linear correlation/relationship there is between the two
variables. Vice verse, the close the correlation measures towards -1,
the stronger the negative linear correlation/relationship there is
between the two variables.

The correlation measures the “linear” relationships between the
variables. If the relationships between the variables are not linear,
then correlation measures cannot capture them, for instance, a quadratic
relationship. Scatterplots between the variables may be easier to spot
those non-linear relationships between the variables which we will show
in the following section.

``` r
correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt", 
         title="Figure 1. Correlations Between the Variables",
         mar = c(0, 0, 2, 0))
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n")
```

![](../images/entertainment/unnamed-chunk-3-1.png)<!-- -->

### Boxplot

Figure 2 shows the number of shares across different days of the week.
Here, due to the huge number of large-valued outliers, I capped the
number of shares to 10,000 so that we can see the medians and the
interquartile ranges clearly for different days of the week.

This is a boxplot with the days of the week on the x-axis and the number
of shares on the y-axis. We can inspect the trend of shares to see if
the shares are higher on a Monday, a Friday or a Sunday for the
entertainment articles.

Figure 2 coincides with the findings in Table 3 that the variance of
shares is huge across days of the week, and the mean values of shares
across different days are driven by larged-valued outliers. Therefore,
those mean values of shares are not close to the median values of shares
for each day of the week.

``` r
ggplot(data = edadata, aes(x = day.week, y = shares)) + 
  geom_boxplot(fill = "white", outlier.shape = NA) + 
  coord_cartesian(ylim=c(0, 10000)) + 
  geom_jitter(aes(color = day.week), size = 1) + 
  guides(color = guide_legend(override.aes = list(size = 8))) + 
  labs(x = "Day of the Week", y = "Number of Shares", 
       title = "Figure 2. Number of shares across different days of the week") + 
  scale_color_discrete(name = "Day of the Week") +
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 14))
```

![](../images/entertainment/unnamed-chunk-4-1.png)<!-- -->

### Barplot

Figure 3 shows the popularity of the news articles in relations to their
closeness to a top LDA topic for the entertainment channel on any day of
the week. The Latent Dirichlet Allocation (LDA) is an algorithm applied
to the Mashable texts of the articles in order to identify the five top
relevant topics and then measure the closeness of the current articles
to each topic, and there are five topics categories. Thus, each article
published on Mashable was measured for each of the topic categories.
Together, those LDA measures in ratios are added to 1 for each article.
Thus, these LDA topics variables are highly correlated with one another.

We calculated the mean ratios of these LDA topics variables for the
specific day of the week. These mean ratios are further classified into
a “popular” group and an “unpopular” group according to their number of
shares (&gt; 1400 or &lt; 1400) which is shown in Figure 3 barplot.
Note, the `position = "stack"` not `position = "fill"` in the `geom_bar`
function.

Some mean ratios of a LDA topic do not seem to vary over the days of a
week while other mean ratios of the LDA topics vary across different
days of the week. Recall, when we dichotomize a continuous variable into
different groups, we lose information about that variable. Here, I just
want to show you whether or not the mean ratios of a LDA topic differ
across time for different categories of shares.

``` r
b.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

ggplot(data = b.plot2, aes(x = day.week, y = avg.LDA, fill = LDA.Topic)) + 
  geom_bar(stat = "identity", position = "stack") + 
  labs(x = "Day of the Week", y = "Closeness to Top LDA Topic", 
       title = "Figure 3. Popularity of Top LDA Topic on Day of the Week") + 
  scale_fill_discrete(name = "LDA Topic") + 
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 13), 
        axis.title.y = element_text(size = 13), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13)) + 
  facet_wrap(~ class.shares)
```

![](../images/entertainment/unnamed-chunk-5-1.png)<!-- -->

### Line Plot

Figure 4 is a line plot that shows the same measurements as in Figure 3
that we can see the patterns of the mean ratios of a LDA topic vary or
not vary across time in different popularity groups more clearly. Again,
some mean ratios of LDA topics do not seem to vary across time when the
corresponding lines are flattened while other mean ratios of LDA topics
vary across time when their lines are fluctuating. The patterns observed
in the “popular” group may not reflect on the same trend in the
“unpopular” group for articles in the entertainment channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

ggplot(data = l.plot2, aes(x = day.week, y = avg.LDA, group = LDA.Topic)) + 
  geom_line(aes(color = LDA.Topic), lwd = 2) + 
  labs(x = "Day of the Week", y = "Closeness to LDA Topic", 
       title = "Figure 4. Popularity of LDA Topic on Day of the Week") + 
  scale_color_discrete(name = "LDA Topic") +
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13)) +
  facet_wrap(~ class.shares)
```

![](../images/entertainment/unnamed-chunk-6-1.png)<!-- -->

### Scatterplots

Figure 5 shows the relationship between the average keyword and
log-transformed shares for articles in the entertainment channel across
different days of the week. In the news popularity study, it showed
average keyword was ranked top one predictor in variable importance in
the optimal predictive model (random forest) they selected that produced
the highest accuracy in prediction of popularity online articles.
Therefore, we are interested to see how average keyword is related with
log shares. The different colored linear regression lines indicate
different days of the week.

If the points display an upward trend, it indicates a positive
relationship between the average keyword and log-shares. With an
increasing log number of shares, the number of average keywords also
increases, meaning people tend to share the article more when they see
more of those average keywords in the article. On the contrary, if the
points are in a downward trend, it indicates a negative relationship
between the average keyword and log-shares. With an decreasing log
number of shares, the number of average keywords decreases as well.
People tend to share the articles less when they see less of these
average keywords in the articles from the entertainment channel.

Figure 6 is similar, except it compares the log-transformed number of
shares to the log-transformed images in the article. As noted
previously, both of these variables do not behave properly in a linear
model due to the existence of extreme outliers in the data.

``` r
ggplot(data = edadata, aes(x = kw_avg_avg, y = log.shares, color = day.week)) + 
  geom_point(size = 2) + #aes(shape = class.shares)
  scale_color_discrete(name = "Day of the Week") + 
  coord_cartesian(xlim=c(0, 10000)) +
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "Average Keywords", y = "log(number of shares)", 
       title = "Figure 5. Average Keywords vs Log Number of Shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/entertainment/unnamed-chunk-7-1.png)<!-- -->

``` r
ggplot(data = edadata, aes(x = log(num_imgs + 1), y = log.shares, color = day.week)) + 
  geom_point(size = 2) +
  scale_color_discrete(name = "Day of the Week") + 
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "log(number of images)", y = "log(number of shares)", 
       title = "Figure 5. Log Number of Images vs Log Number of Shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/entertainment/unnamed-chunk-8-1.png)<!-- -->

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the entertainment channel in figures
7a, 7b, 7c, and 7d. We’re aiming for something close to a straight line,
which would indicate that the data is approximately normal in its
distribution and does not need further standardization.

``` r
ggplot(edadata) + geom_qq(aes(sample = shares)) + geom_qq_line(aes(sample = shares)) + 
  labs(x = "Theoretical Quantiles", y = "Share Numbers", 
       title = "Figure 7a. QQ Plot for Non-Transformed Shares") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/entertainment/unnamed-chunk-9-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = log(shares))) + geom_qq_line(aes(sample = log(shares))) +
    labs(x = "Theoretical Quantiles", y = "Log(Share Numbers)", 
       title = "Figure 7b. QQ Plot for Log-Transformed Shares") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/entertainment/unnamed-chunk-10-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = num_imgs)) + geom_qq_line(aes(sample = num_imgs)) + 
  labs(x = "Theoretical Quantiles", y = "Image Numbers", 
       title = "Figure 7c. QQ Plot for Non-Transformed Image Numbers") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/entertainment/unnamed-chunk-11-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = log(num_imgs + 1))) + geom_qq_line(aes(sample = log(num_imgs + 1))) +
    labs(x = "Theoretical Quantiles", y = "Log(Image Numbers)", 
       title = "Figure 7d. QQ Plot for Log-Transformed Image Numbers") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/entertainment/unnamed-chunk-12-1.png)<!-- -->

Whether it’s appropriate to perform a logarithmic transformation on the
number of images is somewhat less clear than for the number of shares.

# Modeling

## Linear Regression

The linear regression process takes a matrix of all of the predictor
variables we’ve chosen and compares their values to each of the
corresponding values of the response variable, `log.shares`. This allows
us to calculate the most accurate linear combination of the predictor
variables to make up the response variable. We can choose a variety of
sets of predictors and compare their Root Mean Square Errors, R-Squared
values, and Mean Absolute Errors to see which one is the strongest
model. Below, we’ve fit multiple linear models that include all of our
variables and various combinations of interaction terms and/or quadratic
terms.

Originally, we used 20 top ranked predictors selected by the optimal
predictive model used in the paper and fit them in three types of models
below.

1.  All predictors  
2.  All predictors and their quadratic terms  
3.  All predictors and all their interaction terms

We slowly filtered out predictors, their corresponding second order
terms and the interaction terms that were insignificant at 0.05 level.
In addition, we also examined the predictors using correlation plots. We
slowly got rid of some predictors that are highly correlated with each
other such as the LDA topics variables which are also shown in Figures 3
and 4, average number of shares of keywords, maximum number of shares of
average keywords, minimum number of shares of average keywords and many
of others. We carefully monitored this process and compared the models
with the adjusted R-squared and RMSE values. Due to multi-collinearity
among the predictors, reducing the number of predictors that are
correlated with one another did not make the model fit worse, and the
RMSE value from the model was surprisingly decreased.

We repeated this process to trim down the number of predictors and
eventually selected the ones that seem to be important in predicting the
number of shares, and they are not highly correlated with each other.
The parameters in the linear regression model 1 were chosen through this
process, and they are listed in Table 8 below. The response variable is
`log(shares)`.

Table 8. The predictors in linear regression model 1  
Index \| Parameter in Model 1 —— \| —————————– 1 \| `kw_avg_avg`  
2 \| `LDA_02`  
3 \| `dayweek`  
4 \| `self_reference_avg_sharess`  
5 \| `average_token_length`  
6 \| `n_tokens_content`  
7 \| `n_tokens_title`  
8 \| `global_subjectivity`  
9 \| `num_imgs`  
10 \| `I(n_tokens_content^2)` 11 \| `kw_avg_avg:num_imgs` 12 \|
`average_token_length:global_subjectivity` 13 \|
`dayweek:self_reference_avg_sharess`

``` r
train1$dayweek <- as.factor(train1$dayweek)
test1$dayweek <- as.factor(test1$dayweek)
preProcValues <- preProcess(train1, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train1)
testTransformed <- predict(preProcValues, test1)

cv_fit3 <- train(log.shares ~ . + I(n_tokens_content^2) + kw_avg_avg:num_imgs + 
                   average_token_length:global_subjectivity + dayweek:self_reference_avg_sharess, 
                 data=trainTransformed,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
summary(cv_fit3)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.9592 -0.5991 -0.2379  0.3679  4.5103 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.062687   0.032092  -1.953 0.050837 .  
    ## dayweek2                                   -0.031163   0.044299  -0.703 0.481790    
    ## dayweek3                                   -0.040931   0.044439  -0.921 0.357061    
    ## dayweek4                                   -0.012187   0.045000  -0.271 0.786536    
    ## dayweek5                                    0.040930   0.048463   0.845 0.398395    
    ## dayweek6                                    0.340453   0.066301   5.135 2.93e-07 ***
    ## dayweek7                                    0.475131   0.058704   8.094 7.24e-16 ***
    ## kw_avg_avg                                  0.178338   0.014200  12.559  < 2e-16 ***
    ## LDA_02                                     -0.033894   0.013940  -2.431 0.015076 *  
    ## self_reference_avg_sharess                  0.121476   0.031641   3.839 0.000125 ***
    ## average_token_length                        0.017702   0.043895   0.403 0.686763    
    ## n_tokens_content                           -0.028525   0.020922  -1.363 0.172821    
    ## n_tokens_title                             -0.002402   0.013774  -0.174 0.861588    
    ## global_subjectivity                         0.060294   0.018758   3.214 0.001316 ** 
    ## num_imgs                                    0.039491   0.015823   2.496 0.012601 *  
    ## `I(n_tokens_content^2)`                     0.006502   0.005444   1.194 0.232427    
    ## `kw_avg_avg:num_imgs`                       0.048956   0.016256   3.012 0.002612 ** 
    ## `average_token_length:global_subjectivity`  0.014620   0.012327   1.186 0.235703    
    ## `dayweek2:self_reference_avg_sharess`       0.001241   0.048953   0.025 0.979774    
    ## `dayweek3:self_reference_avg_sharess`       0.014674   0.047087   0.312 0.755333    
    ## `dayweek4:self_reference_avg_sharess`       0.027502   0.045289   0.607 0.543705    
    ## `dayweek5:self_reference_avg_sharess`      -0.044841   0.056327  -0.796 0.426019    
    ## `dayweek6:self_reference_avg_sharess`      -0.136792   0.045555  -3.003 0.002689 ** 
    ## `dayweek7:self_reference_avg_sharess`      -0.113790   0.055358  -2.056 0.039880 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9593 on 4917 degrees of freedom
    ## Multiple R-squared:  0.08411,    Adjusted R-squared:  0.07982 
    ## F-statistic: 19.63 on 23 and 4917 DF,  p-value: < 2.2e-16

``` r
cv_fit4 <- train(log.shares ~ . - num_imgs + I(log(num_imgs + 1)) + I(n_tokens_content^2) +
                 I(self_reference_avg_sharess^2) + kw_avg_avg:I(log(num_imgs + 1)) +
                 average_token_length:global_subjectivity, 
                 data=trainTransformed,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
summary(cv_fit4)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.8928 -0.5931 -0.2364  0.3658  4.5183 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.041642   0.032952  -1.264  0.20639    
    ## dayweek2                                   -0.034820   0.044290  -0.786  0.43180    
    ## dayweek3                                   -0.042110   0.044433  -0.948  0.34332    
    ## dayweek4                                   -0.012686   0.044996  -0.282  0.77801    
    ## dayweek5                                    0.044541   0.048418   0.920  0.35766    
    ## dayweek6                                    0.348078   0.066361   5.245 1.63e-07 ***
    ## dayweek7                                    0.470597   0.058676   8.020 1.31e-15 ***
    ## kw_avg_avg                                  0.190841   0.015665  12.183  < 2e-16 ***
    ## LDA_02                                     -0.032066   0.013934  -2.301  0.02142 *  
    ## self_reference_avg_sharess                  0.156062   0.021170   7.372 1.96e-13 ***
    ## average_token_length                        0.018460   0.043838   0.421  0.67370    
    ## n_tokens_content                           -0.031180   0.021532  -1.448  0.14766    
    ## n_tokens_title                             -0.003118   0.013763  -0.227  0.82080    
    ## global_subjectivity                         0.056625   0.018688   3.030  0.00246 ** 
    ## `I(log(num_imgs + 1))`                      0.060732   0.025212   2.409  0.01604 *  
    ## `I(n_tokens_content^2)`                     0.008400   0.005486   1.531  0.12574    
    ## `I(self_reference_avg_sharess^2)`          -0.007462   0.001780  -4.192 2.82e-05 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.066075   0.023252   2.842  0.00451 ** 
    ## `average_token_length:global_subjectivity`  0.015593   0.012342   1.263  0.20650    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9591 on 4922 degrees of freedom
    ## Multiple R-squared:  0.08349,    Adjusted R-squared:  0.08014 
    ## F-statistic: 24.91 on 18 and 4922 DF,  p-value: < 2.2e-16

``` r
#result_tab <- data.frame(t(cv_fit3$results), t(cv_fit4$results))
#colnames(result_tab) <- c("Model 1","Model 2")
#rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

#kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")

#pred1 <- predict(cv_fit1, newdata = testTransformed)
#pred2 <- predict(cv_fit2, newdata = testTransformed)
pred3 <- predict(cv_fit3, newdata = testTransformed)
pred4 <- predict(cv_fit4, newdata = testTransformed)
#cv_rmse1 <- postResample(pred1, obs = testTransformed$log.shares)
#cv_rmse2 <- postResample(pred2, obs = testTransformed$log.shares)
cv_rmse3 <- postResample(pred3, obs = testTransformed$log.shares)
cv_rmse4 <- postResample(pred4, obs = testTransformed$log.shares)
result2 <- rbind(cv_rmse3, cv_rmse4)
row.names(result2) <- c("Model 1","Model 2")
kable(result2, digits = 4, caption = "Table ###. Cross Validation - Model Predictions on Test Set")
```

|         |   RMSE | Rsquared |    MAE |
|:--------|-------:|---------:|-------:|
| Model 1 | 0.9760 |   0.0581 | 0.7014 |
| Model 2 | 0.9711 |   0.0637 | 0.6988 |

Table \#\#\#. Cross Validation - Model Predictions on Test Set

## Random Forest

The bootstrap approach to fitting a tree model involves resampling our
data and fitting a tree to each sample, and then averaging the resulting
predictions of each of those models. The random forest approach adds an
extra step for each of these samples, where only a random subset of the
predictor variables is chosen each time, in order to reduce the
correlation between each of the trees. We don’t have to worry about
creating dummy variables for categorical variables, because our data
already comes in an entirely numeric form.

``` r
train2 <- train %>% select(-class_shares, -shares, -dayweek, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
test2 <- test %>% select(-class_shares, -shares, -dayweek, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
train2
```

    ## # A tibble: 4,941 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       6.39         0  0.0400                 1                  0                    0                0                0                0
    ##  2       7.09         0  0.495                  1                  0                    0                0                0                0
    ##  3       7.65         0  0.215                  1                  0                    0                0                0                0
    ##  4       7.09         0  0.0200                 1                  0                    0                0                0                0
    ##  5       7.09         0  0.0225                 1                  0                    0                0                0                0
    ##  6       6.45         0  0.0200                 1                  0                    0                0                0                0
    ##  7       7.17      1114. 0.324                  0                  1                    0                0                0                0
    ##  8       8.76       849. 0.0400                 0                  0                    1                0                0                0
    ##  9       7.00       935. 0.258                  0                  0                    1                0                0                0
    ## 10       7.31       827. 0.0333                 0                  0                    1                0                0                0
    ## # ... with 4,931 more rows, and 7 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, average_token_length <dbl>,
    ## #   n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>

``` r
preProcValues <- preProcess(train2, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train2)
testTransformed <- predict(preProcValues, test2)

random_forest <- train(log.shares ~ ., data = trainTransformed,
    method = "rf",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = data.frame(mtry = 1:5))

random_forest_predict <- predict(random_forest, newdata = testTransformed)
rf_rmse <- postResample(random_forest_predict, obs = testTransformed$log.shares)
random_forest
```

    ## Random Forest 
    ## 
    ## 4941 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 3952, 3953, 3952, 3954, 3953 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared    MAE      
    ##   1     0.9694197  0.08374922  0.7154593
    ##   2     0.9568442  0.08486424  0.7064913
    ##   3     0.9582016  0.08324761  0.7093603
    ##   4     0.9594011  0.08297425  0.7115407
    ##   5     0.9603584  0.08271308  0.7131091
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 2.

## Boosted Tree

Random forest models use bagging technique (bootstrap aggregation) to
build independent decision trees with different subsets of predictors
and combine them in parallel. On the contrary, gradient boosted trees
use a method called boosting. Boosting method trains each weak learner
slowly and then combines them sequentially, with weak learners being the
decision trees with only one split. Thus, each new tree can correct the
errors made by the previous tree. Because boosting is to slowly train
the trees so that they avoid overfitting the data. Since the trees grow
slowly and in a sequential manner, each tree we create is based off a
previous tree and we update the predictions as we go. For instance, we
fit the model, we get our predictions, and now we create a new model
based off the previous model. Then, we update our predictions based on
the new model. Then, we build a newer model based off the previous one,
and we update the predictions from the new model. The process is
repeated until the criteria set for the tuning parameters are met.

There are tuning parameters in gradient boosting machine learning
technique to help us prevent from growing the trees too quickly and thus
keep us from overfitting the model.

-   `shrinkage`: A shrinkage parameter controls the growth rate of the
    trees, slows fitting process  
-   `n.trees`: The amount of times we want to the process to repeat in
    training the trees.  
-   `interaction.depth`: The amount of splits we want to fit a tree.  
-   `n.minobsinnode`: The minimum number of observations in a node at
    least.

Here, we use the `caret` package and the `gbm` package to run the
boosted tree model with the training set and predict on the test set.
The values of the tuning parameters are set as below:

-   `shrinkage` = 0.1  
-   `n.trees` = 25, 50, 75, 100, 125  
-   `interaction.depth` = 1, 2, 3, 4, 5  
-   `n.minobsinnode` = 10

We then use 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
model chosen by cross validation across all combinations of tuning
parameters values produces the lowest root mean squared error (RMSE).

``` r
#expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10)
boosted_tree <- train(log.shares ~ . , data = trainTransformed,
      method = "gbm", 
      trControl = trainControl(method = "cv", number = 10), #method="repeatedcv", repeats=5
      tuneGrid = expand.grid(n.trees = c(25, 50, 75, 100, 125), interaction.depth = 1:5, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 4941 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4446, 4447, 4449, 4446, 4447, 4446, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9643302  0.07888021  0.7134713
    ##   1                   50      0.9586503  0.08359554  0.7080847
    ##   1                   75      0.9562280  0.08637902  0.7052508
    ##   1                  100      0.9556091  0.08675206  0.7052493
    ##   1                  125      0.9556349  0.08689053  0.7052031
    ##   2                   25      0.9609199  0.08066718  0.7096189
    ##   2                   50      0.9570665  0.08428552  0.7054863
    ##   2                   75      0.9568658  0.08459773  0.7055063
    ##   2                  100      0.9573341  0.08408729  0.7068111
    ##   2                  125      0.9574520  0.08375835  0.7071326
    ##   3                   25      0.9593915  0.08157992  0.7078445
    ##   3                   50      0.9579721  0.08248928  0.7059061
    ##   3                   75      0.9583643  0.08235434  0.7059683
    ##   3                  100      0.9593947  0.08098389  0.7066653
    ##   3                  125      0.9601421  0.08078315  0.7069069
    ##   4                   25      0.9599583  0.07973907  0.7075342
    ##   4                   50      0.9598277  0.07919665  0.7049714
    ##   4                   75      0.9589847  0.08222032  0.7041328
    ##   4                  100      0.9614465  0.07927084  0.7057447
    ##   4                  125      0.9639103  0.07642161  0.7070170
    ##   5                   25      0.9586924  0.08237065  0.7070126
    ##   5                   50      0.9588926  0.08149733  0.7081411
    ##   5                   75      0.9615281  0.07858522  0.7092461
    ##   5                  100      0.9640650  0.07611075  0.7113397
    ##   5                  125      0.9658459  0.07523048  0.7131757
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = select(testTransformed, -log.shares))

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse3, cv_rmse4, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Random Forest Model", "Boosted Tree Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Model 1      | 0.9760 |   0.0581 | 0.7014 |
| Linear Model 2      | 0.9711 |   0.0637 | 0.6988 |
| Random Forest Model | 0.9568 |   0.0837 | 0.6954 |
| Boosted Tree Model  | 0.9631 |   0.0721 | 0.6951 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the entertainment
channel is “need to automate this part”.

The best model fit to predict the number of shares

# Final Model

Automation is done with the modifications of the YAML header and the
render function.
