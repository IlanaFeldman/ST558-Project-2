Predictive Models for Popularity of Online News
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
-   [Final Model Fit with Entire
    Data](#final-model-fit-with-entire-data)

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

| Index | Attribute                    | Attribute Information                                  | Type    |
|-------|------------------------------|--------------------------------------------------------|---------|
| 1     | `shares` (target)            | Number of shares                                       | number  |
| 2     | `kw_avg_avg`                 | Average keyword (average shares)                       | number  |
| 3     | `LDA_02`                     | Closeness of current article to a LDA Topic 2          | ratio   |
| 4.1   | `weekday_is_monday`          | Was the article published on a Monday?                 | boolean |
| 4.2   | `weekday_is_tuesday`         | Was the article published on a Tuesday?                | boolean |
| 4.3   | `weekday_is_wednesday`       | Was the article published on a Wednesday?              | boolean |
| 4.4   | `weekday_is_thursday`        | Was the article published on a Thursday?               | boolean |
| 4.5   | `weekday_is_friday`          | Was the article published on a Friday?                 | boolean |
| 4.6   | `weekday_is_saturday`        | Was the article published on a Saturday?               | boolean |
| 4.7   | `weekday_is_sunday`          | Was the article published on a Sunday?                 | boolean |
| 5     | `self_reference_avg_sharess` | Avg. shares of popular news referenced in the articles | number  |
| 6     | `average_token_length`       | Average length of the words in the content             | number  |
| 7     | `n_tokens_content`           | Number of words in the content                         | number  |
| 8     | `n_tokens_title`             | Number of words in the title                           | number  |
| 9     | `global_subjectivity`        | Text subjectivity                                      | ratio   |
| 10    | `num_imgs`                   | Number of images                                       | number  |

``` r
library(tidyverse)
library(knitr)
library(caret)
library(corrplot)
library(ggplot2)
library(gbm)
library(vip)

allnews <- read_csv("C:/Users/peach/Documents/ST558/ST558_repos/St558-Project-2/_Data/OnlineNewsPopularity.csv", col_names = TRUE)

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
train
```

    ## # A tibble: 1,472 x 23
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tues~
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>            <dbl>
    ##  1            0    556       6.32       1         0  0.0201 0.115  0.0200 0.0200  0.825                 1                0
    ##  2            1   1900       7.55       1         0  0.0286 0.0286 0.0286 0.0287  0.885                 1                0
    ##  3            1   5700       8.65       1         0  0.437  0.200  0.0335 0.0334  0.295                 1                0
    ##  4            0    462       6.14       1         0  0.0200 0.0200 0.0200 0.0200  0.920                 1                0
    ##  5            1   3600       8.19       1         0  0.211  0.0255 0.0251 0.0251  0.713                 1                0
    ##  6            0    343       5.84       1         0  0.0201 0.0206 0.0205 0.121   0.818                 1                0
    ##  7            0    507       6.23       1         0  0.0250 0.160  0.0250 0.0250  0.765                 1                0
    ##  8            0    552       6.31       1         0  0.207  0.146  0.276  0.0251  0.346                 1                0
    ##  9            0   1200       7.09       2       885. 0.0202 0.133  0.120  0.0201  0.707                 0                1
    ## 10            1   1900       7.55       3      1207. 0.0335 0.217  0.0334 0.0335  0.683                 0                0
    ## # ... with 1,462 more rows, and 11 more variables: weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
    ## #   weekday_is_friday <dbl>, weekday_is_saturday <dbl>, weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
    ## #   average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>

``` r
train1 <- train %>% select(-class_shares, -shares, 
                           -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                           -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)

test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# Data

When a subset of data is selected for the lifestyle channel articles
which contain 2099 articles, the subset of data is then split into a
training set (70% of the subset data) and a test set (30% of the subset
data) based on the target variable, the number of shares. There are 1472
articles in the training set and 627 observations in the test set
regarding the lifestyle channel. The `createDataPartition` function from
the `caret` package is used to split the data into training and test
sets. We set a seed so that the analyses we implemented are
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

The lifestyle channel has 1472 articles collected. Now let us take a
look at the relationships between our response and the predictors with
some numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. We classified the number of shares greater than 1400 in a day
as “popular” and the number of shares less than 1400 in a day as
“unpopular”. We can see the number of articles from the lifestyle
channel classified into “popular” group or “unpopular” group on
different days of the week from January 7th, 2013 to January 7th, 2015
when the articles were published and retrieved by the study. Note, this
table may not reflect on the information contained in the data due to
dichotomizing the data.

Table 3 shows the average shares of the articles on different days of
the week. We can compare and determine which day of the week has the
most average number of shares for the lifestyle channel. Here, we can
see a potential problem for our analysis later. Median shares are all
very different from the average shares on any day of the week. Recall
that median is a robust measure for center. It is robust to outliers in
the data. On the contrary, mean is also a measure of center but it is
not robust to outliers. Mean measure can be influenced by potential
outliers.

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
lifestyle channel in mashable.com on different days of the week. This
table indicates the number of times *average keywords* shown in the
articles regarding the average number of shares, and the table is
showing the average number of those *average keywords* calculated for
each day of the week so that we can compare to see which day of the
week, the *average keywords* showed up the most or the worst according
to the average of shares in the lifestyle channel.

Table 5 shows the numerical summaries of average shares of referenced
articles in mashable.com on different days of the week. We calculated
the average number of shares of those articles that contained the
earlier popularity of news referenced for each day of the week so that
we can compare which day has the most or the worst average number of
shares when there were earlier popularity of news referenced in the
lifestylearticles.

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
| Unpopular |     91 |     107 |       121 |       99 |     91 |       19 |     44 |
| Popular   |    132 |     128 |       152 |      155 |    122 |       95 |    116 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   3486.045 |  5318.719 |          1600 |        7.5882 |       0.9883 |           7.3778 |
| Tuesday   |   4168.881 | 14793.301 |          1400 |        7.5302 |       0.9636 |           7.2442 |
| Wednesday |   3399.795 |  6374.130 |          1600 |        7.5289 |       0.9706 |           7.3778 |
| Thursday  |   3669.768 |  6002.006 |          1600 |        7.6033 |       0.9941 |           7.3778 |
| Friday    |   2939.681 |  4380.366 |          1500 |        7.5090 |       0.8830 |           7.3132 |
| Saturday  |   4073.597 |  5418.537 |          2300 |        7.8972 |       0.8131 |           7.7407 |
| Sunday    |   3517.994 |  4285.079 |          2000 |        7.7887 |       0.7859 |           7.6009 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    3246.340 |   1348.691 |       3185.484 |    1134.762 |
| Tuesday   |    3264.005 |   1032.268 |       3149.765 |    1135.409 |
| Wednesday |    3451.520 |   1800.231 |       3250.528 |    1292.849 |
| Thursday  |    3311.112 |   1136.251 |       3233.834 |    1295.447 |
| Friday    |    3196.760 |   1066.331 |       2996.297 |    1218.999 |
| Saturday  |    3737.214 |   1076.623 |       3546.271 |    1438.096 |
| Sunday    |    3817.900 |   1333.473 |       3752.708 |    1605.025 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      6696.825 |    10535.357 |             3000 |        5300.0 |
| Tuesday   |      5168.337 |     8723.726 |             2520 |        4295.2 |
| Wednesday |      7360.335 |    27000.613 |             2200 |        6200.0 |
| Thursday  |      5844.472 |    13768.595 |             2475 |        5830.0 |
| Friday    |      5673.178 |    12087.230 |             2100 |        4320.0 |
| Saturday  |      5377.458 |     7985.266 |             2500 |        4787.5 |
| Sunday    |      6853.131 |    20490.799 |             2350 |        3425.0 |

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
| Unpopular    |           0.4726 |          0.0853 |              0.4774 |
| Popular      |           0.4744 |          0.0983 |              0.4769 |

Table 6. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 7. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     4.4709 |    5.8961 |           1.0 |         1.2323 |        0.9277 |            0.6931 |
| Tuesday   |     4.0298 |    6.1153 |           1.0 |         1.1297 |        0.9070 |            0.6931 |
| Wednesday |     4.8388 |    8.9986 |           1.0 |         1.2006 |        0.9770 |            0.6931 |
| Thursday  |     4.1063 |    6.6595 |           1.0 |         1.1295 |        0.9133 |            0.6931 |
| Friday    |     4.3005 |    9.0238 |           1.0 |         1.1406 |        0.9025 |            0.6931 |
| Saturday  |     7.3947 |    7.8138 |           4.5 |         1.6343 |        1.0611 |            1.7006 |
| Sunday    |     7.4125 |    7.2271 |           6.0 |         1.6791 |        1.0422 |            1.9459 |

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

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

### Boxplot

Figure 2 shows the number of shares across different days of the week.
Here, due to the huge number of large-valued outliers, I capped the
number of shares to 8,000 so that we can see the medians and the
interquartile ranges clearly for different days of the week.

This is a boxplot with the days of the week on the x-axis and the number
of shares on the y-axis. We can inspect the trend of shares to see if
the shares are higher on a Monday, a Friday or a Sunday for the
lifestyle articles.

Figure 2 coincides with the findings in Table 3 that the variance of
shares is huge across days of the week, and the mean values of shares
across different days are driven by larged-valued outliers. Therefore,
those mean values of shares are not close to the median values of shares
for each day of the week.

Figure 3 is a boxplot that compares the word count of the content of
each article, grouped by the day of the week and additionally colored by
its popularity. Similarly to before, for ease of viewing, the word count
is capped at 2,000 since a small number of articles have a much larger
word count in some cases. This has the capacity to indicate behavior
regarding article length on different days of the week, article
popularity based on length, or a cross-section of the two, where
articles of some length may be more or less popular on some days of the
week than others.

``` r
ggplot(data = edadata, aes(x = day.week, y = shares)) + 
  geom_boxplot(fill = "white", outlier.shape = NA) + 
  coord_cartesian(ylim=c(0, 8000)) + 
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

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
ggplot(data = edadata, aes(x = day.week, y = n_tokens_content)) +
  geom_boxplot(fill = "white", outlier.shape = NA) +
  coord_cartesian(ylim=c(0, 2000)) + 
  geom_jitter(aes(color = class.shares), size = 1) + 
  guides(color = guide_legend(override.aes = list(size = 8))) + 
  labs(x = "Popularity", y = "Word Count of Article", 
       title = "Figure 3. Word Count of Article by Popularity") + 
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 14))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### Barplot

Figure 4 shows the popularity of the news articles in relations to their
closeness to a top LDA topic for the lifestyle channel on any day of the
week. The Latent Dirichlet Allocation (LDA) is an algorithm applied to
the Mashable texts of the articles in order to identify the five top
relevant topics and then measure the closeness of the current articles
to each topic, and there are five topics categories. Thus, each article
published on Mashable was measured for each of the topic categories.
Together, those LDA measures in ratios are added to 1 for each article.
Thus, these LDA topics variables are highly correlated with one another.

We calculated the mean ratios of these LDA topics variables for the
specific day of the week. These mean ratios are further classified into
a “popular” group and an “unpopular” group according to their number of
shares (&gt; 1400 or &lt; 1400) which is shown in Figure 4 barplot.
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
       title = "Figure 4. Popularity of Top LDA Topic on Day of the Week") + 
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

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

### Line Plot

Figure 5 is a line plot that shows the same measurements as in Figure 4
that we can see the patterns of the mean ratios of a LDA topic vary or
not vary across time in different popularity groups more clearly. Again,
some mean ratios of LDA topics do not seem to vary across time when the
corresponding lines are flattened while other mean ratios of LDA topics
vary across time when their lines are fluctuating. The patterns observed
in the “popular” group may not reflect on the same trend in the
“unpopular” group for articles in the lifestyle channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

ggplot(data = l.plot2, aes(x = day.week, y = avg.LDA, group = LDA.Topic)) + 
  geom_line(aes(color = LDA.Topic), lwd = 2) + 
  labs(x = "Day of the Week", y = "Closeness to LDA Topic", 
       title = "Figure 5. Popularity of LDA Topic on Day of the Week") + 
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

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### Scatterplots

Figure 6 shows the relationship between the average keyword and
log-transformed shares for articles in the lifestyle channel across
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
average keywords in the articles from the lifestyle channel.

Figure 7 is similar, except it compares the log-transformed number of
shares to the log-transformed images in the article. As noted
previously, both of these variables do not behave properly in a linear
model due to the existence of extreme outliers in the data. Here, a
negative correlation will indicate that shares sharply decreased for
articles containing more images, and a positive correlation will
indicate that shares sharply increased for articles containing more
images.

``` r
ggplot(data = edadata, aes(x = kw_avg_avg, y = log.shares, color = day.week)) + 
  geom_point(size = 2) + #aes(shape = class.shares)
  scale_color_discrete(name = "Day of the Week") + 
  coord_cartesian(xlim=c(0, 10000)) +
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "Average Keywords", y = "log(number of shares)", 
       title = "Figure 6. Average Keywords vs Log Number of Shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
ggplot(data = edadata, aes(x = log(num_imgs + 1), y = log.shares, color = day.week)) + 
  geom_point(size = 2) +
  scale_color_discrete(name = "Day of the Week") + 
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "log(number of images)", y = "log(number of shares)", 
       title = "Figure 7. Log Number of Images vs Log Number of Shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the lifestyle channel in figures 8a,
8b, 8c, and 8d. We’re aiming for something close to a straight line,
which would indicate that the data is approximately normal in its
distribution and does not need further standardization.

``` r
ggplot(edadata) + geom_qq(aes(sample = shares)) + geom_qq_line(aes(sample = shares)) + 
  labs(x = "Theoretical Quantiles", y = "Share Numbers", 
       title = "Figure 8a. QQ Plot for Non-Transformed Shares") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = log(shares))) + geom_qq_line(aes(sample = log(shares))) +
    labs(x = "Theoretical Quantiles", y = "Log(Share Numbers)", 
       title = "Figure 8b. QQ Plot for Log-Transformed Shares") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = num_imgs)) + geom_qq_line(aes(sample = num_imgs)) + 
  labs(x = "Theoretical Quantiles", y = "Image Numbers", 
       title = "Figure 8c. QQ Plot for Non-Transformed Image Numbers") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = log(num_imgs + 1))) + geom_qq_line(aes(sample = log(num_imgs + 1))) +
    labs(x = "Theoretical Quantiles", y = "Log(Image Numbers)", 
       title = "Figure 8d. QQ Plot for Log-Transformed Image Numbers") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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

The second linear regression model provided below applies a log
transformation to the variables `num_imgs` and
`self_reference_avg_sharess`, which are extremely skewed to the right
even after standardization, to see if this accounts for outliers in the
data and puts those variables in an appropriate context.

Table 8. The predictors in linear regression model 1

| Index | Parameter in Linear Regression 1           |
|-------|--------------------------------------------|
| 1     | `kw_avg_avg`                               |
| 2     | `LDA_02`                                   |
| 3     | `dayweek`                                  |
| 4     | `self_reference_avg_sharess`               |
| 5     | `average_token_length`                     |
| 6     | `n_tokens_content`                         |
| 7     | `n_tokens_title`                           |
| 8     | `global_subjectivity`                      |
| 9     | `num_imgs`                                 |
| 10    | `I(n_tokens_content^2)`                    |
| 11    | `kw_avg_avg:num_imgs`                      |
| 12    | `average_token_length:global_subjectivity` |
| 13    | `dayweek:self_reference_avg_sharess`       |

``` r
train1$dayweek <- as.factor(train1$dayweek)
test1$dayweek <- as.factor(test1$dayweek)
preProcValues <- preProcess(train1, method = c("center", "scale"))
trainTransformed1 <- predict(preProcValues, train1)
testTransformed1 <- predict(preProcValues, test1)
trainTransformed1
```

    ## # A tibble: 1,472 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_~ average_token_l~ n_tokens_content n_tokens_title global_subjecti~
    ##         <dbl> <fct>        <dbl>  <dbl>            <dbl>            <dbl>            <dbl>          <dbl>            <dbl>
    ##  1    -1.37   1            -2.56 -0.522        -0.186              0.116             0.644         -0.920           0.436 
    ##  2    -0.0588 1            -2.56 -0.441        -0.378              0.122            -0.820          0.134           0.0370
    ##  3     1.11   1            -2.56 -0.396        -0.0730             0.477            -0.979          0.660          -0.536 
    ##  4    -1.56   1            -2.56 -0.522        -0.378             -0.400            -0.714          0.134           0.474 
    ##  5     0.621  1            -2.56 -0.474        -0.378              0.149            -0.788         -0.920           1.91  
    ##  6    -1.88   1            -2.56 -0.517         0.000276          -0.403            -0.578          0.660           0.854 
    ##  7    -1.46   1            -2.56 -0.475        -0.186              0.0511            1.08           0.134           0.352 
    ##  8    -1.37   1            -2.56  1.88         -0.378              0.600            -0.466         -1.97           -0.803 
    ##  9    -0.548  2            -1.89  0.420        -0.299              0.932            -0.229          1.19           -0.843 
    ## 10    -0.0588 3            -1.65 -0.397         0.336             -0.0792           -0.752          0.660          -1.09  
    ## # ... with 1,462 more rows, and 1 more variable: num_imgs <dbl>

``` r
cv_fit3 <- train(log.shares ~ . + I(n_tokens_content^2) + kw_avg_avg:num_imgs + 
                   average_token_length:global_subjectivity + dayweek:self_reference_avg_sharess, 
                 data=trainTransformed1,
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
    ## -3.3175 -0.6352 -0.1953  0.4938  4.6046 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.0191608  0.0658291  -0.291  0.77104    
    ## dayweek2                                   -0.0818826  0.0910686  -0.899  0.36873    
    ## dayweek3                                   -0.1133024  0.0878858  -1.289  0.19753    
    ## dayweek4                                    0.0002381  0.0892309   0.003  0.99787    
    ## dayweek5                                   -0.0870931  0.0932195  -0.934  0.35032    
    ## dayweek6                                    0.2586752  0.1128843   2.292  0.02208 *  
    ## dayweek7                                    0.1221937  0.1014374   1.205  0.22855    
    ## kw_avg_avg                                  0.0906101  0.0287118   3.156  0.00163 ** 
    ## LDA_02                                     -0.0265484  0.0262063  -1.013  0.31120    
    ## self_reference_avg_sharess                  0.2936412  0.1022398   2.872  0.00414 ** 
    ## average_token_length                        0.0391978  0.0502118   0.781  0.43514    
    ## n_tokens_content                            0.0936724  0.0370860   2.526  0.01165 *  
    ## n_tokens_title                              0.0209063  0.0255567   0.818  0.41347    
    ## global_subjectivity                        -0.0015538  0.0304201  -0.051  0.95927    
    ## num_imgs                                    0.0290591  0.0326464   0.890  0.37355    
    ## `I(n_tokens_content^2)`                    -0.0102938  0.0047079  -2.186  0.02894 *  
    ## `kw_avg_avg:num_imgs`                       0.1171510  0.0260389   4.499 7.37e-06 ***
    ## `average_token_length:global_subjectivity`  0.0234899  0.0124507   1.887  0.05941 .  
    ## `dayweek2:self_reference_avg_sharess`      -0.2292420  0.1565084  -1.465  0.14321    
    ## `dayweek3:self_reference_avg_sharess`      -0.2949869  0.1082046  -2.726  0.00648 ** 
    ## `dayweek4:self_reference_avg_sharess`      -0.2563741  0.1251963  -2.048  0.04076 *  
    ## `dayweek5:self_reference_avg_sharess`      -0.1518996  0.1361101  -1.116  0.26461    
    ## `dayweek6:self_reference_avg_sharess`      -0.1499298  0.2129948  -0.704  0.48160    
    ## `dayweek7:self_reference_avg_sharess`      -0.2842474  0.1194082  -2.380  0.01742 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9669 on 1448 degrees of freedom
    ## Multiple R-squared:  0.07975,    Adjusted R-squared:  0.06513 
    ## F-statistic: 5.456 on 23 and 1448 DF,  p-value: 2.518e-15

``` r
cv_fit4 <- train(log.shares ~ . - num_imgs - self_reference_avg_sharess + I(log(num_imgs+1)) + I(n_tokens_content^2) +
                 I(log(self_reference_avg_sharess+1)) + kw_avg_avg:I(log(num_imgs + 1)) +
                 average_token_length:global_subjectivity, 
                 data=trainTransformed1,
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
    ## -3.2559 -0.6385 -0.1935  0.4877  4.5260 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0002400  0.0672616   0.004   0.9972    
    ## dayweek2                                   -0.0759217  0.0908469  -0.836   0.4035    
    ## dayweek3                                   -0.1120316  0.0879792  -1.273   0.2031    
    ## dayweek4                                    0.0039863  0.0893630   0.045   0.9644    
    ## dayweek5                                   -0.0886116  0.0933093  -0.950   0.3424    
    ## dayweek6                                    0.2528487  0.1125150   2.247   0.0248 *  
    ## dayweek7                                    0.1276618  0.1015658   1.257   0.2090    
    ## kw_avg_avg                                  0.1264231  0.0278104   4.546 5.92e-06 ***
    ## LDA_02                                     -0.0280738  0.0261918  -1.072   0.2840    
    ## average_token_length                        0.0446570  0.0497141   0.898   0.3692    
    ## n_tokens_content                            0.0965929  0.0376898   2.563   0.0105 *  
    ## n_tokens_title                              0.0209927  0.0255672   0.821   0.4117    
    ## global_subjectivity                        -0.0001502  0.0305780  -0.005   0.9961    
    ## `I(log(num_imgs + 1))`                      0.0303453  0.0426734   0.711   0.4771    
    ## `I(n_tokens_content^2)`                    -0.0069424  0.0045934  -1.511   0.1309    
    ## `I(log(self_reference_avg_sharess + 1))`    0.1520836  0.0628052   2.422   0.0156 *  
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.1539667  0.0340307   4.524 6.55e-06 ***
    ## `average_token_length:global_subjectivity`  0.0261706  0.0123421   2.120   0.0341 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9674 on 1454 degrees of freedom
    ## Multiple R-squared:  0.07499,    Adjusted R-squared:  0.06417 
    ## F-statistic: 6.934 on 17 and 1454 DF,  p-value: < 2.2e-16

``` r
pred3 <- predict(cv_fit3, newdata = testTransformed1)
pred4 <- predict(cv_fit4, newdata = testTransformed1)
cv_rmse3 <- postResample(pred3, obs = testTransformed1$log.shares)
cv_rmse4 <- postResample(pred4, obs = testTransformed1$log.shares)
#result2 <- rbind(cv_rmse3, cv_rmse4)
#row.names(result2) <- c("Model 1","Model 2")
#kable(result2, digits = 4, caption = "Table ###. Cross Validation - Model Predictions on Test Set")
```

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
preProcValues2 <- preProcess(train2, method = c("center", "scale"))
trainTransformed2 <- predict(preProcValues2, train2)
testTransformed2 <- predict(preProcValues2, test2)
trainTransformed2
```

    ## # A tibble: 1,472 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednes~ weekday_is_thurs~ weekday_is_frid~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>              <dbl>             <dbl>            <dbl>
    ##  1    -1.37        -2.56 -0.522             2.37              -0.436             -0.477            -0.457           -0.411
    ##  2    -0.0588      -2.56 -0.441             2.37              -0.436             -0.477            -0.457           -0.411
    ##  3     1.11        -2.56 -0.396             2.37              -0.436             -0.477            -0.457           -0.411
    ##  4    -1.56        -2.56 -0.522             2.37              -0.436             -0.477            -0.457           -0.411
    ##  5     0.621       -2.56 -0.474             2.37              -0.436             -0.477            -0.457           -0.411
    ##  6    -1.88        -2.56 -0.517             2.37              -0.436             -0.477            -0.457           -0.411
    ##  7    -1.46        -2.56 -0.475             2.37              -0.436             -0.477            -0.457           -0.411
    ##  8    -1.37        -2.56  1.88              2.37              -0.436             -0.477            -0.457           -0.411
    ##  9    -0.548       -1.89  0.420            -0.422              2.29              -0.477            -0.457           -0.411
    ## 10    -0.0588      -1.65 -0.397            -0.422             -0.436              2.09             -0.457           -0.411
    ## # ... with 1,462 more rows, and 8 more variables: weekday_is_saturday <dbl>, weekday_is_sunday <dbl>,
    ## #   self_reference_avg_sharess <dbl>, average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>,
    ## #   global_subjectivity <dbl>, num_imgs <dbl>

``` r
random_forest <- train(log.shares ~ ., data = trainTransformed2,
    method = "rf",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = data.frame(mtry = 1:5), importance = TRUE)

random_forest_predict <- predict(random_forest, newdata = testTransformed2)
rf_rmse <- postResample(random_forest_predict, obs = testTransformed2$log.shares)
random_forest
```

    ## Random Forest 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1179, 1177, 1179, 1177, 1176 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared    MAE      
    ##   1     0.9797739  0.04972620  0.7516122
    ##   2     0.9800207  0.04379828  0.7527251
    ##   3     0.9804460  0.04572803  0.7532795
    ##   4     0.9806098  0.04682730  0.7536991
    ##   5     0.9836604  0.04383146  0.7568901
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 1.

``` r
random_forest$results
```

    ## # A tibble: 5 x 7
    ##    mtry  RMSE Rsquared   MAE RMSESD RsquaredSD  MAESD
    ##   <int> <dbl>    <dbl> <dbl>  <dbl>      <dbl>  <dbl>
    ## 1     1 0.980   0.0497 0.752 0.0410     0.0240 0.0263
    ## 2     2 0.980   0.0438 0.753 0.0458     0.0246 0.0275
    ## 3     3 0.980   0.0457 0.753 0.0458     0.0251 0.0264
    ## 4     4 0.981   0.0468 0.754 0.0482     0.0232 0.0275
    ## 5     5 0.984   0.0438 0.757 0.0499     0.0232 0.0284

``` r
# mtry = random_forest$bestTune[[1]]
```

We then used 5 fold cross validation to search for the tuning parameter
value ranging from 1 to 5 that produces the optimal random forest model.
The optimal model chosen by cross validation produced the smallest RMSE
value when mtry = 1 and the lowest RMSE = 0.9797739 when training the
model on the training set.

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

We then used 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
model chosen by cross validation across all combinations of tuning
parameters values produces the lowest root mean squared error (RMSE).

``` r
#expand.grid(n.trees = c(25, 50, 75, 100, 125), interaction.depth = 1:5, shrinkage = 0.1, n.minobsinnode = 10)
boosted_tree <- train(log.shares ~ . , data = trainTransformed2,
      method = "gbm", 
      trControl = trainControl(method = "cv", number = 10), #method="repeatedcv", repeats=5
      tuneGrid = expand.grid(n.trees = c(25, 50, 75, 100, 125), interaction.depth = 1:5, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE)
boosted_tree_predict <- predict(boosted_tree, newdata = select(testTransformed2, -log.shares))
boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed2$log.shares)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1325, 1326, 1325, 1325, 1324, 1325, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9769836  0.04770778  0.7494999
    ##   1                   50      0.9776899  0.04358768  0.7488421
    ##   1                   75      0.9795196  0.04119715  0.7507544
    ##   1                  100      0.9797796  0.04168059  0.7496812
    ##   1                  125      0.9829633  0.03768929  0.7525519
    ##   2                   25      0.9749324  0.04846097  0.7486882
    ##   2                   50      0.9818082  0.03866540  0.7511064
    ##   2                   75      0.9849727  0.03717623  0.7522272
    ##   2                  100      0.9889651  0.03357259  0.7534582
    ##   2                  125      0.9902417  0.03380704  0.7544737
    ##   3                   25      0.9803310  0.03904670  0.7499339
    ##   3                   50      0.9880751  0.03251430  0.7537922
    ##   3                   75      0.9914815  0.03157919  0.7560888
    ##   3                  100      0.9972540  0.02750305  0.7617255
    ##   3                  125      1.0002613  0.02766825  0.7643401
    ##   4                   25      0.9832446  0.03685641  0.7519685
    ##   4                   50      0.9913671  0.03188742  0.7581352
    ##   4                   75      1.0022470  0.02307663  0.7678374
    ##   4                  100      1.0124373  0.01825297  0.7760306
    ##   4                  125      1.0191248  0.01701350  0.7813262
    ##   5                   25      0.9771076  0.04487131  0.7472190
    ##   5                   50      0.9847193  0.03823500  0.7529846
    ##   5                   75      0.9932727  0.03219481  0.7574308
    ##   5                  100      1.0001856  0.02953811  0.7654409
    ##   5                  125      1.0083956  0.02605447  0.7714343
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at
    ##  a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree$results
```

    ## # A tibble: 25 x 10
    ##    shrinkage interaction.depth n.minobsinnode n.trees  RMSE Rsquared   MAE RMSESD RsquaredSD  MAESD
    ##        <dbl>             <int>          <dbl>   <dbl> <dbl>    <dbl> <dbl>  <dbl>      <dbl>  <dbl>
    ##  1       0.1                 1             10      25 0.977   0.0477 0.749 0.0791     0.0356 0.0456
    ##  2       0.1                 2             10      25 0.975   0.0485 0.749 0.0773     0.0308 0.0429
    ##  3       0.1                 3             10      25 0.980   0.0390 0.750 0.0807     0.0286 0.0465
    ##  4       0.1                 4             10      25 0.983   0.0369 0.752 0.0816     0.0262 0.0469
    ##  5       0.1                 5             10      25 0.977   0.0449 0.747 0.0824     0.0315 0.0481
    ##  6       0.1                 1             10      50 0.978   0.0436 0.749 0.0813     0.0323 0.0490
    ##  7       0.1                 2             10      50 0.982   0.0387 0.751 0.0793     0.0257 0.0447
    ##  8       0.1                 3             10      50 0.988   0.0325 0.754 0.0833     0.0272 0.0479
    ##  9       0.1                 4             10      50 0.991   0.0319 0.758 0.0803     0.0220 0.0463
    ## 10       0.1                 5             10      50 0.985   0.0382 0.753 0.0814     0.0234 0.0458
    ## # ... with 15 more rows

``` r
# n.trees = boosted_tree$bestTune[[1]]
```

We then used 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
boosted tree model chosen by cross validation produced the smallest RMSE
value (0.9749324) when n.trees = 25, interaction.depth = 2, shrinkage =
0.1 and n.minobsinnode = 10 when training the model with the training
set.

## Model Comparisons

The best model fit to predict the number of shares for the lifestyle
channel can be determined by looking at the Root Mean Square Error, the
Mean Absolute Error, or the R-squared value using the test set. Table 9
shows these criteria measures for each candidate model. The approach
I’ve taken below picks whichever model is considered superior by the
majority of these three metrics, and if each of the metrics picks a
different model, then the one with the lowest RMSE will be chosen.

``` r
result2 <- rbind(cv_rmse3, cv_rmse4, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Regression 1", "Linear Regression 2", "Random Forest", "Boosted Tree")
kable(result2, digits = 4, caption = "Table 9. Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Regression 1 | 0.9906 |   0.0450 | 0.7382 |
| Linear Regression 2 | 0.9860 |   0.0511 | 0.7344 |
| Random Forest       | 0.9859 |   0.0688 | 0.7411 |
| Boosted Tree        | 0.9928 |   0.0371 | 0.7488 |

Table 9. Cross Validation - Comparisons of the models in test set

``` r
rmse_best <- names(which.min(result2[,1]))
rsq_best <- names(which.max(result2[,2]))
mae_best <- names(which.min(result2[,3]))
model_best <- table(c(rmse_best, rsq_best, mae_best))
final_model <- if_else(max(model_best) > 1, names(which.max(model_best)), rmse_best)
```

# Final Model Fit with Entire Data

The best model fit to predict the number of shares is the **Random
Forest** model for the lifestyle articles.

both training and the test sets have been standardized with same mean
and same standard deviation.

Figure 9. variable importance plot

``` r
linear1 <- function(...){
  
  data1comb <- rbind(trainTransformed1, testTransformed1, ...)
  linearfit1 <- train(log.shares ~ . + I(n_tokens_content^2) + kw_avg_avg:num_imgs + 
                        average_token_length:global_subjectivity + dayweek:self_reference_avg_sharess, 
                      data=data1comb, 
                      method = "lm", ...)
  pred <- predict(linearfit1, newdata = data1comb, ...)
  linear.rmse <- postResample(pred, obs = data1comb$log.shares, ...)
  #colnames(linear.rmse) <- "Linear Regression 1"
  linear_rmse <- kable(linear.rmse, digits = 4, caption = "10. Final model fit on entire data", ...)
  summary <- summary(linearfit1)
  important <- varImp(linearfit1, ...)
  plot_imp <- vip(linearfit1, ...)
  return(list(linear_rmse, summary, important, plot_imp))
  
}

linear2 <- function(...){
  
  data1comb <- rbind(trainTransformed1, testTransformed1, ...)
  linearfit2 <- train(log.shares ~ . - num_imgs - self_reference_avg_sharess + I(n_tokens_content^2) + 
                        I(log(num_imgs + 1)) + I(log(self_reference_avg_sharess+1)) + 
                        kw_avg_avg:I(log(num_imgs + 1)) + average_token_length:global_subjectivity, 
                      data=data1comb, 
                      method = "lm", ...)
  pred <- predict(linearfit2, newdata = data1comb, ...)
  linear.rmse <- postResample(pred, obs = data1comb$log.shares, ...)
  #row.names(linear.rmse) <- "Linear Regression 2"
  linear_rmse <- kable(linear.rmse, digits = 4, caption = "10. Final model fit on entire data", ...)
  summary <- summary(linearfit2)
  return(list(linear_rmse, summary, important, plot_imp))
  
}

randomf <- function(...){
  
  data2comb <- rbind(trainTransformed2, testTransformed2, ...)
  rffit <- train(log.shares ~ . , data = data2comb, 
                 method = "rf", 
                 tuneGrid = data.frame(mtry = random_forest$bestTune[[1]]), 
                 importance = TRUE, ...)
  important <- varImp(rffit , ...)
  plot_imp <- vip(rffit, ...)
  return(list(rffit, important, plot_imp))
  
}

boostedt <- function(...){
  
  data2comb <- rbind(trainTransformed2, testTransformed2, ...)
  btfit <- train(log.shares ~ . , data = data2comb, 
                 method = "gbm", 
                 tuneGrid = data.frame(n.trees = boosted_tree$bestTune[[1]], 
                                       interaction.depth = boosted_tree$bestTune[[2]], 
                                       shrinkage = boosted_tree$bestTune[[3]], 
                                       n.minobsinnode = boosted_tree$bestTune[[4]]), 
                 verbose = FALSE, ...)
  important <- varImp(btfit, ...)
  plot_imp <- vip(btfit, ...)
  return(list(btfit, important, plot_imp))
  
}

f_model <- if_else(final_model == "Random Forest", 3, 
           if_else(final_model == "Boosted Tree", 4, 
                   if_else(final_model == "Linear Regression 1", 1, 2)))

switch(f_model,
       
       modell = linear1(),
       model2 = linear2(),
       model3 = randomf(),
       model4 = boostedt()
)
```

    ## [[1]]
    ## Random Forest 
    ## 
    ## 2099 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 2099, 2099, 2099, 2099, 2099, 2099, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared    MAE      
    ##   0.9841459  0.05295993  0.7501863
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 1
    ## 
    ## [[2]]
    ## rf variable importance
    ## 
    ##                            Overall
    ## kw_avg_avg                 100.000
    ## num_imgs                    96.939
    ## weekday_is_saturday         96.333
    ## self_reference_avg_sharess  78.312
    ## weekday_is_sunday           74.758
    ## n_tokens_content            68.328
    ## average_token_length        57.266
    ## global_subjectivity         53.275
    ## weekday_is_wednesday        46.169
    ## LDA_02                      26.641
    ## weekday_is_tuesday          18.847
    ## weekday_is_friday           17.004
    ## weekday_is_thursday         13.297
    ## n_tokens_title               8.667
    ## weekday_is_monday            0.000
    ## 
    ## [[3]]

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/lifestyle_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
#imp <- varImp(boosted_tree)
#imp2 <- data.frame(imp[1])
#first <- rownames(imp2)[order(imp2$Overall, decreasing=TRUE)][1] # most important
#second <- rownames(imp2)[order(imp2$Overall, decreasing=TRUE)][2] #2nd
#third <- rownames(imp2)[order(imp2$Overall, decreasing=TRUE)][3] # 3rd

#print(paste0("The most important predictor is ", first, ", the second is ", second))
```
