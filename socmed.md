ST558 - Project 2 - Predictive Modeling
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
        -   [Scatterplot](#scatterplot)
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

| Attribute              | Attribute Information                     | Type                                              |         |
|------------------------|-------------------------------------------|---------------------------------------------------|---------|
| 1                      | `shares` (target)                         | Number of shares                                  | number  |
| 2                      | `kw_avg_avg`                              | Average keyword (average shares)                  | number  |
| 3                      | `LDA_02`                                  | Closeness to LDA topic 2                          | ratio   |
| 4                      | `weekday_is_monday`                       | Was the article published on a Monday?            | boolean |
| `weekday_is_tuesday`   | Was the article published on a Tuesday?   | boolean                                           |         |
| `weekday_is_wednesday` | Was the article published on a Wednesday? | boolean                                           |         |
| `weekday_is_thursday`  | Was the article published on a Thursday?  | boolean                                           |         |
| `weekday_is_friday`    | Was the article published on a Friday?    | boolean                                           |         |
| `weekday_is_saturday`  | Was the article published on a Saturday?  | boolean                                           |         |
| `weekday_is_sunday`    | Was the article published on a Sunday?    | boolean                                           |         |
| 5                      | `self_reference_avg_sharess`              | Average shares of referenced articles in Mashable | number  |
| 6                      | `average_token_length`                    | Average length of the words in the content        | number  |
| 7                      | `n_tokens_content`                        | Number of words in the content                    | number  |
| 8                      | `n_tokens_title`                          | Number of words in the title                      | number  |
| 9                      | `global_subjectivity`                     | Text subjectivity                                 | ratio   |
| 10                     | `num_imgs`                                | Number of images                                  | number  |

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

    ## # A tibble: 1,628 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg_sh~ average_token_len~ n_tokens_content n_tokens_title global_subjectiv~ num_imgs
    ##         <dbl>   <dbl>      <dbl>  <dbl>                  <dbl>              <dbl>            <dbl>          <dbl>             <dbl>    <dbl>
    ##  1       7.86       1         0  0.0224                  1775                4.64              257              8             0.4          0
    ##  2       6.54       1         0  0.0201                  3900                4.44              218              8             0.522       11
    ##  3       8.48       1         0  0.0288                  2858.               4.39             1226              9             0.408        1
    ##  4       6.75       1         0  0.0336                  2796.               4.79             1121             10             0.497        1
    ##  5       8.48       3       832. 0.0224                  6600                4.68              168              9             0.638       11
    ##  6       9.13       3      1072. 0.619                   1800                4.45              100              9             0.338        1
    ##  7       7.38       3      1564. 0.0336                  2300                4.63             1596             10             0.454        8
    ##  8       6.65       4      1862. 0.0299                  3500                4.79              518              7             0.599        1
    ##  9       9.81       5      2210. 0.0502                 10400                4.25              358              8             0.504        1
    ## 10       7.38       5      1398. 0.319                   1600                5.39              358              6             0.493        1
    ## # ... with 1,618 more rows

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# Data

When a subset of data is selected for the socmed channel articles which
contain 2323 articles, the subset of data is then split into a training
set (70% of the subset data) and a test set (30% of the subset data)
based on the target variable, the number of shares. There are 1628
articles in the training set and 695 observations in the test set
regarding the socmed channel. The `createDataPartition` function from
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
boolean format are needed when we run the ensemble models.

# Exploratory Data Analysis

The socmed channel has 1628 articles collected. Now let us take a look
at the relationships between our response and the predictors with some
numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from socmed channel falls into
different categories on different days of the week for 709 days.

Table 3 shows the average shares of the articles on different days of
the week. Here, we can see a potential problem for our analysis later.
Median shares are all very different from the average shares on any day
of the week. Recall that median is a robust measure for center. It is
robust to outliers in the data. On the contrary, mean is also a measure
of center but it is not robust to outliers. Mean measure can be
influenced by potential outliers.

In addition, Table 3 also shows the standard deviation of shares is huge
for any day of the week. They are potentially larger than the average
shares. This tells us the variance of shares for any day is huge. We
know a common variance stabilizing transformation to deal with
increasing variance of the response variable, that is, the
log-transformation, which could help us on this matter. Therefore, Table
3 again shows after the log-transformation of shares, the mean values
are similar to their corresponding median values, and their standard
deviations are much smaller than before relatively speaking.

Table 4 shows the numerical summaries of average keywords from socmed
channel in mashable.com on different days of the week. Table 5 shows the
numerical summaries of average shares of referenced articles in
mashable.com on different days of the week.

Table 5 checks the numerical summaries of the `global_subjectivity`
variable between popular and unpopular articles, to see if there’s any
difference or a higher variation in subjectivity in popular articles.
Text subjectivity is a value between 0 and 1, so there isn’t any need
for transformation.

Table 6 checks the numerical summaries of the image count per article on
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
| Unpopular |     59 |      89 |        77 |       88 |     52 |       21 |     10 |
| Popular   |    173 |     233 |       220 |      239 |    176 |      105 |     86 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   4191.319 |  6741.698 |          2200 |        7.8289 |       0.9155 |           7.6962 |
| Tuesday   |   3207.497 |  4044.715 |          1900 |        7.7050 |       0.7823 |           7.5496 |
| Wednesday |   3597.333 |  4869.488 |          2200 |        7.7999 |       0.7936 |           7.6962 |
| Thursday  |   2964.226 |  2971.900 |          2000 |        7.6782 |       0.7868 |           7.6009 |
| Friday    |   3848.167 |  5000.793 |          2300 |        7.8463 |       0.8310 |           7.7407 |
| Saturday  |   3427.770 |  3784.587 |          2300 |        7.8361 |       0.7186 |           7.7407 |
| Sunday    |   4355.875 |  6313.474 |          2500 |        7.9569 |       0.8287 |           7.8240 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    3158.086 |  1215.8199 |       3105.198 |   1094.3811 |
| Tuesday   |    3170.531 |   849.5774 |       3104.404 |    874.3928 |
| Wednesday |    3124.782 |   748.3088 |       3160.847 |    958.1442 |
| Thursday  |    3218.250 |   713.8441 |       3242.990 |    822.9191 |
| Friday    |    3396.239 |  2366.3300 |       3183.976 |    944.2154 |
| Saturday  |    3468.093 |  3074.0769 |       3136.249 |    887.6163 |
| Sunday    |    3467.503 |  1571.3500 |       3286.085 |   1088.4383 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      9791.491 |     27844.76 |         2925.000 |      6625.000 |
| Tuesday   |      8648.764 |     23554.57 |         3375.000 |      6110.306 |
| Wednesday |      7666.357 |     13003.55 |         3300.000 |      6407.375 |
| Thursday  |      7793.434 |     16128.44 |         3400.000 |      5744.347 |
| Friday    |      8424.720 |     25966.08 |         3500.000 |      5755.625 |
| Saturday  |      8591.587 |     17355.76 |         4034.469 |      6316.667 |
| Sunday    |     12189.867 |     70446.38 |         2466.700 |      5093.500 |

Table 5. Summary of Average shares of referenced articles in Mashable on
Day of the Week

``` r
edadata %>% group_by(class.shares) %>% summarize(
  Avg.subjectivity = mean(global_subjectivity), Sd.subjectivity = sd(global_subjectivity), 
  Median.subjectivity = median(global_subjectivity)) %>% kable(digits = 4, caption = "Table 5. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4682 |          0.1111 |              0.4654 |
| Popular      |           0.4570 |          0.0881 |              0.4558 |

Table 5. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 6. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     3.4138 |    5.6981 |             1 |         1.0431 |        0.8146 |            0.6931 |
| Tuesday   |     5.0559 |   10.8446 |             1 |         1.1139 |        0.9487 |            0.6931 |
| Wednesday |     3.0673 |    6.0434 |             1 |         0.9642 |        0.7759 |            0.6931 |
| Thursday  |     5.1437 |    9.0038 |             1 |         1.1668 |        1.0112 |            0.6931 |
| Friday    |     3.5570 |    6.8631 |             1 |         0.9939 |        0.8708 |            0.6931 |
| Saturday  |     5.8492 |    7.8944 |             1 |         1.4205 |        0.9772 |            0.6931 |
| Sunday    |     5.3854 |    7.9702 |             1 |         1.3084 |        0.9896 |            0.6931 |

Table 6. Comparing Image Counts by the Day of the Week

``` r
file.name <- paste0("../images/", params$channel, "1.png")
```

## Visualizations

Graphical presentation is a great tool used to visualize the
relationships between the predictors and the number of shares (or log
number of shares). Below we will see some plots that tell us stories
between those variables.

### Correlation Plot

Figure 1 shows the correlations between the variables, both the response
and the predictors, which will be used in the regression models as well
as the ensemble models for predicting the number of shares. Notice that
there may be some collinearity among the predictor variables.

``` r
# keep log-shares
#corplt <- train %>% select(-class_shares, -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday,
#                           -weekday_is_thursday, -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday) 
correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt", 
         title="Figure 1. Correlations Between the Variables",
         mar = c(0, 0, 2, 0))
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n")
```

![](../images/socmed1.pngunnamed-chunk-3-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "2.png")
```

### Boxplot

Figure 2 shows the number of shares across different days of the week.
Here, due to the huge number of large-valued outliers, I capped the
number of shares to 10,000 so that we can see the medians and the
interquartile ranges for different days of the week. Figure 2 coincides
with the findings in Table 2 that the variance of shares is huge across
days of the week, and the mean values of shares across different days
are driven by larged-valued outliers. Therefore, those mean values of
shares are not close to the median values of shares for each day of the
week. The median number of shares seems to be bigger during weekend than
weekdays.

``` r
boxplot1 <- ggplot(data = edadata, aes(x = day.week, y = shares))
boxplot1 + geom_boxplot(fill = "white", outlier.shape = NA) + 
  coord_cartesian(ylim=c(0, 10000)) + 
  geom_jitter(aes(color = day.week), size = 1) + 
  guides(color = guide_legend(override.aex = list(size = 6))) + 
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

![](../images/socmed2.pngunnamed-chunk-4-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "3.png")
```

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the socmed channel on mashable.com on any day of the week. The
measurements of the different LDA topics are in ratios, and these are
the mean ratios calculated for the specific day of thte week for that
topic across 709 days of collections of data in mashable.com. These mean
ratios are further classified into a “popular” group and an “unpopular”
group according to their number of shares.

Some mean ratios of a LDA topic do not seem to vary over the days of a
week while other mean ratios of LDA topics vary across different days of
the week. Note, when we dicotomize a continuous variable into different
groups, we lose information about that variable. Here, I just want to
show you whether or not the mean ratios of a LDA topic differ across
time for different levels of shares. The classified version of number of
shares will not be used to fit in a model later.

``` r
b.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

barplot1 <- ggplot(data = b.plot2, aes(x = day.week, y = avg.LDA, fill = LDA.Topic))
barplot1 + geom_bar(stat = "identity", position = "stack") + 
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

![](../images/socmed3.pngunnamed-chunk-5-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "4.png")
```

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the socmed channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

lineplot1 <- ggplot(data = l.plot2, aes(x = day.week, y = avg.LDA, group = LDA.Topic))
lineplot1 + geom_line(aes(color = LDA.Topic), lwd = 2) + 
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

![](../images/socmed4.pngunnamed-chunk-6-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "5.png")
```

### Scatterplot

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the socmed channel
across different days of the week. In the news popularity study, it says
average keyword is the most important predictor in the models they used
which accounted for the most variation in the data. Therefore, we are
interested to see how average keyword is correlated with log shares. The
different colored linear regression lines indicate different days of the
week. If it is an upward trend, it shows positive linear relationship.
If it is a downward trend, it shows a negative linear relationship. More
tilted the line is, much stronger the relationship is regardless of
positive or negative.

Figure 6 is similar, except it compares the log-transformed number of
shares to the log-transformed images in the article. As noted
previously, both of these variables do not behave properly in a linear
model due to the existence of extreme outliers in the data.

``` r
scatter1 <- ggplot(data = edadata, aes(x = kw_avg_avg, y = log.shares, color = day.week)) #y=kw_avg_max
scatter1 + geom_point(size = 2) + #aes(shape = class.shares)
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

![](../images/socmed5.pngunnamed-chunk-7-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "6.png")
```

``` r
scatter2 <- ggplot(data = edadata, aes(x = log(num_imgs + 1), y = log.shares, color = day.week))
scatter2 + geom_point(size = 2) +
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

![](../images/socmed6.pngunnamed-chunk-8-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7a.png")
```

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the socmed channel in figures 7a,
7b, 7c, and 7d. We’re aiming for something close to a straight line,
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

![](../images/socmed7a.pngunnamed-chunk-9-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7b.png")
```

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

![](../images/socmed7b.pngunnamed-chunk-10-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7c.png")
```

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

![](../images/socmed7c.pngunnamed-chunk-11-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7d.png")
```

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

![](../images/socmed7d.pngunnamed-chunk-12-1.png)<!-- -->

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

``` r
# using train1, dayweek is numeric, no class_shares
#train1 <- train %>% select(-class_shares, -shares) #keep log.shares
#test1 <- test %>% select(-class_shares, -shares) #keep log.shares
train1$dayweek <- as.factor(train1$dayweek)
test1$dayweek <- as.factor(test1$dayweek)
preProcValues <- preProcess(train1, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train1)
testTransformed <- predict(preProcValues, test1)

cv_fit3 <- train(log.shares ~ . + I(n_tokens_content^2) + I(self_reference_avg_sharess^2) + 
                 kw_avg_avg:num_imgs + average_token_length:global_subjectivity, 
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
    ## -6.7130 -0.6077 -0.1259  0.5142  3.9399 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0865098  0.0646293   1.339 0.180905    
    ## dayweek2                                   -0.1321449  0.0826865  -1.598 0.110207    
    ## dayweek3                                   -0.0158275  0.0839164  -0.189 0.850422    
    ## dayweek4                                   -0.1604719  0.0823802  -1.948 0.051595 .  
    ## dayweek5                                    0.0511307  0.0893966   0.572 0.567433    
    ## dayweek6                                   -0.0064937  0.1069122  -0.061 0.951575    
    ## dayweek7                                    0.2178460  0.1172627   1.858 0.063386 .  
    ## kw_avg_avg                                  0.0460258  0.0250854   1.835 0.066725 .  
    ## LDA_02                                     -0.0814245  0.0252847  -3.220 0.001306 ** 
    ## self_reference_avg_sharess                  0.3524698  0.0450301   7.827 8.96e-15 ***
    ## average_token_length                       -0.1020108  0.0375574  -2.716 0.006676 ** 
    ## n_tokens_content                            0.1684237  0.0402944   4.180 3.07e-05 ***
    ## n_tokens_title                              0.0001029  0.0239428   0.004 0.996573    
    ## global_subjectivity                        -0.0819431  0.0254386  -3.221 0.001302 ** 
    ## num_imgs                                   -0.1062641  0.0293070  -3.626 0.000297 ***
    ## `I(n_tokens_content^2)`                    -0.0253103  0.0143061  -1.769 0.077050 .  
    ## `I(self_reference_avg_sharess^2)`          -0.0170289  0.0025053  -6.797 1.50e-11 ***
    ## `kw_avg_avg:num_imgs`                       0.0876670  0.0273658   3.204 0.001384 ** 
    ## `average_token_length:global_subjectivity` -0.0261192  0.0100829  -2.590 0.009672 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9561 on 1609 degrees of freedom
    ## Multiple R-squared:  0.09605,    Adjusted R-squared:  0.08594 
    ## F-statistic: 9.498 on 18 and 1609 DF,  p-value: < 2.2e-16

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
    ## -6.7070 -0.6112 -0.1273  0.5208  3.9386 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.068984   0.065874   1.047 0.295164    
    ## dayweek2                                   -0.141248   0.082900  -1.704 0.088603 .  
    ## dayweek3                                   -0.017732   0.084170  -0.211 0.833171    
    ## dayweek4                                   -0.169223   0.082569  -2.049 0.040579 *  
    ## dayweek5                                    0.050891   0.089666   0.568 0.570414    
    ## dayweek6                                   -0.009651   0.107377  -0.090 0.928394    
    ## dayweek7                                    0.221975   0.117666   1.886 0.059409 .  
    ## kw_avg_avg                                  0.067503   0.025389   2.659 0.007922 ** 
    ## LDA_02                                     -0.094648   0.025020  -3.783 0.000161 ***
    ## self_reference_avg_sharess                  0.348576   0.045281   7.698  2.4e-14 ***
    ## average_token_length                       -0.103635   0.037674  -2.751 0.006011 ** 
    ## n_tokens_content                            0.155306   0.040725   3.814 0.000142 ***
    ## n_tokens_title                              0.001220   0.024007   0.051 0.959460    
    ## global_subjectivity                        -0.086663   0.025461  -3.404 0.000681 ***
    ## `I(log(num_imgs + 1))`                     -0.109142   0.047519  -2.297 0.021758 *  
    ## `I(n_tokens_content^2)`                    -0.029967   0.014305  -2.095 0.036344 *  
    ## `I(self_reference_avg_sharess^2)`          -0.016872   0.002519  -6.699  2.9e-11 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.105939   0.036440   2.907 0.003697 ** 
    ## `average_token_length:global_subjectivity` -0.027293   0.010105  -2.701 0.006986 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9588 on 1609 degrees of freedom
    ## Multiple R-squared:  0.09084,    Adjusted R-squared:  0.08067 
    ## F-statistic: 8.932 on 18 and 1609 DF,  p-value: < 2.2e-16

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
| Model 1 | 1.0706 |   0.0723 | 0.7845 |
| Model 2 | 1.0686 |   0.0757 | 0.7833 |

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

    ## # A tibble: 1,628 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       7.86         0  0.0224                 1                  0                    0                0                0                0
    ##  2       6.54         0  0.0201                 1                  0                    0                0                0                0
    ##  3       8.48         0  0.0288                 1                  0                    0                0                0                0
    ##  4       6.75         0  0.0336                 1                  0                    0                0                0                0
    ##  5       8.48       832. 0.0224                 0                  0                    1                0                0                0
    ##  6       9.13      1072. 0.619                  0                  0                    1                0                0                0
    ##  7       7.38      1564. 0.0336                 0                  0                    1                0                0                0
    ##  8       6.65      1862. 0.0299                 0                  0                    0                1                0                0
    ##  9       9.81      2210. 0.0502                 0                  0                    0                0                1                0
    ## 10       7.38      1398. 0.319                  0                  0                    0                0                1                0
    ## # ... with 1,618 more rows, and 7 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, average_token_length <dbl>,
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
    ## 1628 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1303, 1302, 1302, 1302, 1303 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared   MAE      
    ##   1     0.9606698  0.1182411  0.7316117
    ##   2     0.9419569  0.1217768  0.7108718
    ##   3     0.9431642  0.1156764  0.7114149
    ##   4     0.9434054  0.1156106  0.7105782
    ##   5     0.9450679  0.1120898  0.7121410
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
    ## 1628 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1464, 1466, 1464, 1466, 1466, 1466, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9600704  0.09697637  0.7325316
    ##   1                   50      0.9490041  0.11141773  0.7193253
    ##   1                   75      0.9438317  0.11716074  0.7115830
    ##   1                  100      0.9403562  0.12244627  0.7070380
    ##   1                  125      0.9414403  0.12015540  0.7071706
    ##   2                   25      0.9478173  0.11564570  0.7189607
    ##   2                   50      0.9430204  0.11796174  0.7100477
    ##   2                   75      0.9434647  0.11641078  0.7093996
    ##   2                  100      0.9430316  0.11718473  0.7075330
    ##   2                  125      0.9472454  0.11181961  0.7093632
    ##   3                   25      0.9477701  0.11093493  0.7168767
    ##   3                   50      0.9428291  0.11803665  0.7086590
    ##   3                   75      0.9435309  0.11691071  0.7093589
    ##   3                  100      0.9487740  0.11153564  0.7135919
    ##   3                  125      0.9515952  0.10734613  0.7148470
    ##   4                   25      0.9494257  0.10410244  0.7189138
    ##   4                   50      0.9510715  0.10264312  0.7177852
    ##   4                   75      0.9526193  0.10309786  0.7197286
    ##   4                  100      0.9566659  0.10109305  0.7231431
    ##   4                  125      0.9610599  0.09655032  0.7263360
    ##   5                   25      0.9507792  0.10177898  0.7171720
    ##   5                   50      0.9534757  0.10190991  0.7164681
    ##   5                   75      0.9584227  0.09675842  0.7214508
    ##   5                  100      0.9635010  0.09219644  0.7264407
    ##   5                  125      0.9687060  0.08839823  0.7317979
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
| Linear Model 1      | 1.0706 |   0.0723 | 0.7845 |
| Linear Model 2      | 1.0686 |   0.0757 | 0.7833 |
| Random Forest Model | 1.0417 |   0.1309 | 0.7562 |
| Boosted Tree Model  | 1.0379 |   0.1329 | 0.7563 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the socmed
channel is “need to automate this part”.

The best model fit to predict the number of shares

# Final Model

Automation is done with the modifications of the YAML header and the
render function.
