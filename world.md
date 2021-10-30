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

    ## # A tibble: 5,900 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg_sh~ average_token_len~ n_tokens_content n_tokens_title global_subjectiv~ num_imgs
    ##         <dbl>   <dbl>      <dbl>  <dbl>                  <dbl>              <dbl>            <dbl>          <dbl>             <dbl>    <dbl>
    ##  1       6.57       1         0   0.840                     0                5.09              231             10             0.314        1
    ##  2       7.70       1         0   0.401                     0                4.62             1248              9             0.482        1
    ##  3       7.38       1         0   0.867                     0                4.62              682             12             0.473        1
    ##  4       7.31       1         0   0.700                 16100                4.82              125             11             0.396        1
    ##  5       7.50       1         0   0.840                  1560.               5.24              317             11             0.375        1
    ##  6       7.09       1         0   0.485                     0                4.58              399             11             0.565        1
    ##  7       6.20       1         0   0.702                     0                5.01              443              9             0.420        1
    ##  8       6.63       2       804.  0.862                  3100                4.38              288             12             0.450        0
    ##  9       6.15       2       728.  0.700                     0                4.98              414             10             0.343        1
    ## 10       7.24       3      1047.  0.602                     0                4.15              540             12             0.387       10
    ## # ... with 5,890 more rows

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# Data

When a subset of data is selected for the world channel articles which
contain 8427 articles, the subset of data is then split into a training
set (70% of the subset data) and a test set (30% of the subset data)
based on the target variable, the number of shares. There are 5900
articles in the training set and 2527 observations in the test set
regarding the world channel. The `createDataPartition` function from the
`caret` package is used to split the data into training and test sets.
We set a seed so that the analyses we implemented are reproducible.

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

The world channel has 5900 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from world channel falls into
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

Table 4 shows the numerical summaries of average keywords from world
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
| Unpopular |    601 |     723 |       726 |      695 |    538 |      157 |    189 |
| Popular   |    372 |     378 |       366 |      376 |    352 |      216 |    211 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   2337.043 |  6543.355 |          1100 |        7.1990 |       0.8232 |           7.0031 |
| Tuesday   |   2426.688 |  6565.391 |          1100 |        7.1584 |       0.8682 |           7.0031 |
| Wednesday |   1925.386 |  3406.646 |          1100 |        7.1317 |       0.7749 |           7.0031 |
| Thursday  |   2133.721 |  4561.754 |          1100 |        7.1459 |       0.8352 |           7.0031 |
| Friday    |   2296.185 |  6716.310 |          1100 |        7.2045 |       0.8052 |           7.0031 |
| Saturday  |   2557.188 |  3558.843 |          1600 |        7.4388 |       0.8664 |           7.3778 |
| Sunday    |   2463.262 |  3821.872 |          1400 |        7.4483 |       0.7235 |           7.2442 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    2473.181 |   751.0025 |       2385.787 |    745.9407 |
| Tuesday   |    2498.915 |   912.8466 |       2377.593 |    709.0243 |
| Wednesday |    2524.790 |  1086.8147 |       2394.104 |    666.2618 |
| Thursday  |    2553.265 |   866.3338 |       2432.320 |    744.0002 |
| Friday    |    2540.271 |   809.9252 |       2412.769 |    721.8967 |
| Saturday  |    2543.164 |   789.7911 |       2404.729 |    669.8686 |
| Sunday    |    2551.390 |   741.6412 |       2436.715 |    649.0045 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      3916.641 |     9109.592 |         1500.000 |      2471.000 |
| Tuesday   |      3650.512 |    12051.108 |         1400.000 |      3100.000 |
| Wednesday |      3860.214 |    15032.993 |         1400.000 |      2458.000 |
| Thursday  |      3638.208 |     9786.237 |         1392.571 |      2377.000 |
| Friday    |      5484.440 |    29458.710 |         1489.196 |      2756.583 |
| Saturday  |      3208.286 |     6186.295 |         1300.000 |      2162.000 |
| Sunday    |      3337.209 |     8687.275 |         1500.000 |      2050.125 |

Table 5. Summary of Average shares of referenced articles in Mashable on
Day of the Week

``` r
edadata %>% group_by(class.shares) %>% summarize(
  Avg.subjectivity = mean(global_subjectivity), Sd.subjectivity = sd(global_subjectivity), 
  Median.subjectivity = median(global_subjectivity)) %>% kable(digits = 4, caption = "Table 5. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4002 |          0.1011 |              0.4090 |
| Popular      |           0.4072 |          0.1148 |              0.4233 |

Table 5. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 6. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     2.7256 |    4.8701 |             1 |         0.9631 |        0.7101 |            0.6931 |
| Tuesday   |     2.6576 |    4.4983 |             1 |         0.9628 |        0.6892 |            0.6931 |
| Wednesday |     2.7857 |    5.1872 |             1 |         0.9611 |        0.7171 |            0.6931 |
| Thursday  |     2.7591 |    5.3269 |             1 |         0.9784 |        0.6959 |            0.6931 |
| Friday    |     3.1719 |    6.2381 |             1 |         0.9965 |        0.7643 |            0.6931 |
| Saturday  |     2.3324 |    4.5487 |             1 |         0.8937 |        0.6490 |            0.6931 |
| Sunday    |     3.1125 |    4.8646 |             1 |         1.0199 |        0.7829 |            0.6931 |

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

![](../images/world1.pngunnamed-chunk-3-1.png)<!-- -->

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

![](../images/world2.pngunnamed-chunk-4-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "3.png")
```

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the world channel on mashable.com on any day of the week. The
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

![](../images/world3.pngunnamed-chunk-5-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "4.png")
```

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the world channel.

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

![](../images/world4.pngunnamed-chunk-6-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "5.png")
```

### Scatterplot

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the world channel
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

![](../images/world5.pngunnamed-chunk-7-1.png)<!-- -->

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

![](../images/world6.pngunnamed-chunk-8-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7a.png")
```

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the world channel in figures 7a, 7b,
7c, and 7d. We’re aiming for something close to a straight line, which
would indicate that the data is approximately normal in its distribution
and does not need further standardization.

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

![](../images/world7a.pngunnamed-chunk-9-1.png)<!-- -->

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

![](../images/world7b.pngunnamed-chunk-10-1.png)<!-- -->

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

![](../images/world7c.pngunnamed-chunk-11-1.png)<!-- -->

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

![](../images/world7d.pngunnamed-chunk-12-1.png)<!-- -->

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
    ## -4.4660 -0.5427 -0.1597  0.3692  5.3608 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0244710  0.0317180   0.772  0.44043    
    ## dayweek2                                   -0.0488882  0.0420504  -1.163  0.24504    
    ## dayweek3                                   -0.0796494  0.0420971  -1.892  0.05853 .  
    ## dayweek4                                   -0.0631973  0.0422953  -1.494  0.13518    
    ## dayweek5                                   -0.0054166  0.0443622  -0.122  0.90282    
    ## dayweek6                                    0.3167958  0.0582247   5.441 5.51e-08 ***
    ## dayweek7                                    0.3076476  0.0568577   5.411 6.52e-08 ***
    ## kw_avg_avg                                  0.0833156  0.0129738   6.422 1.45e-10 ***
    ## LDA_02                                     -0.1156664  0.0129655  -8.921  < 2e-16 ***
    ## self_reference_avg_sharess                  0.1661917  0.0232168   7.158 9.17e-13 ***
    ## average_token_length                       -0.2981887  0.0440656  -6.767 1.44e-11 ***
    ## n_tokens_content                           -0.0459741  0.0179386  -2.563  0.01041 *  
    ## n_tokens_title                              0.0196072  0.0126162   1.554  0.12021    
    ## global_subjectivity                         0.0963430  0.0163096   5.907 3.68e-09 ***
    ## num_imgs                                    0.1117624  0.0145662   7.673 1.96e-14 ***
    ## `I(n_tokens_content^2)`                     0.0103693  0.0035200   2.946  0.00323 ** 
    ## `I(self_reference_avg_sharess^2)`          -0.0044174  0.0007789  -5.671 1.48e-08 ***
    ## `kw_avg_avg:num_imgs`                       0.0050688  0.0111307   0.455  0.64884    
    ## `average_token_length:global_subjectivity` -0.0592001  0.0128446  -4.609 4.13e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9541 on 5881 degrees of freedom
    ## Multiple R-squared:  0.09251,    Adjusted R-squared:  0.08973 
    ## F-statistic: 33.31 on 18 and 5881 DF,  p-value: < 2.2e-16

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
    ## -4.4759 -0.5455 -0.1615  0.3727  5.2865 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0650738  0.0322706   2.016 0.043793 *  
    ## dayweek2                                   -0.0504457  0.0420823  -1.199 0.230678    
    ## dayweek3                                   -0.0784558  0.0421269  -1.862 0.062601 .  
    ## dayweek4                                   -0.0647099  0.0423303  -1.529 0.126395    
    ## dayweek5                                   -0.0012788  0.0443848  -0.029 0.977015    
    ## dayweek6                                    0.3185164  0.0582790   5.465 4.81e-08 ***
    ## dayweek7                                    0.3065117  0.0568999   5.387 7.45e-08 ***
    ## kw_avg_avg                                  0.0850509  0.0130539   6.515 7.85e-11 ***
    ## LDA_02                                     -0.1152786  0.0129775  -8.883  < 2e-16 ***
    ## self_reference_avg_sharess                  0.1672371  0.0232299   7.199 6.81e-13 ***
    ## average_token_length                       -0.2994483  0.0440909  -6.792 1.22e-11 ***
    ## n_tokens_content                           -0.0557111  0.0184449  -3.020 0.002535 ** 
    ## n_tokens_title                              0.0175189  0.0126219   1.388 0.165197    
    ## global_subjectivity                         0.0957307  0.0163187   5.866 4.70e-09 ***
    ## `I(log(num_imgs + 1))`                      0.2036179  0.0268274   7.590 3.70e-14 ***
    ## `I(n_tokens_content^2)`                     0.0121745  0.0035529   3.427 0.000615 ***
    ## `I(self_reference_avg_sharess^2)`          -0.0044549  0.0007793  -5.716 1.14e-08 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`          -0.0016706  0.0199035  -0.084 0.933111    
    ## `average_token_length:global_subjectivity` -0.0594312  0.0128653  -4.619 3.93e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9548 on 5881 degrees of freedom
    ## Multiple R-squared:  0.09112,    Adjusted R-squared:  0.08834 
    ## F-statistic: 32.76 on 18 and 5881 DF,  p-value: < 2.2e-16

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
| Model 1 | 0.9794 |   0.0845 | 0.6941 |
| Model 2 | 0.9793 |   0.0846 | 0.6942 |

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

    ## # A tibble: 5,900 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       6.57         0   0.840                 1                  0                    0                0                0                0
    ##  2       7.70         0   0.401                 1                  0                    0                0                0                0
    ##  3       7.38         0   0.867                 1                  0                    0                0                0                0
    ##  4       7.31         0   0.700                 1                  0                    0                0                0                0
    ##  5       7.50         0   0.840                 1                  0                    0                0                0                0
    ##  6       7.09         0   0.485                 1                  0                    0                0                0                0
    ##  7       6.20         0   0.702                 1                  0                    0                0                0                0
    ##  8       6.63       804.  0.862                 0                  1                    0                0                0                0
    ##  9       6.15       728.  0.700                 0                  1                    0                0                0                0
    ## 10       7.24      1047.  0.602                 0                  0                    1                0                0                0
    ## # ... with 5,890 more rows, and 7 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, average_token_length <dbl>,
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
    ## 5900 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 4720, 4719, 4721, 4719, 4721 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared   MAE      
    ##   1     0.9583946  0.1164469  0.6771352
    ##   2     0.9421747  0.1141223  0.6659247
    ##   3     0.9434439  0.1102331  0.6696683
    ##   4     0.9463028  0.1058137  0.6729843
    ##   5     0.9471606  0.1051826  0.6752052
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
    ## 5900 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 5310, 5310, 5312, 5309, 5310, 5310, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9635549  0.08589626  0.6836610
    ##   1                   50      0.9512170  0.10188332  0.6736086
    ##   1                   75      0.9454919  0.10846582  0.6685364
    ##   1                  100      0.9430208  0.11121932  0.6666788
    ##   1                  125      0.9412566  0.11409712  0.6654915
    ##   2                   25      0.9552392  0.09645564  0.6760220
    ##   2                   50      0.9448452  0.10921791  0.6672135
    ##   2                   75      0.9412286  0.11446236  0.6649608
    ##   2                  100      0.9399847  0.11649667  0.6641316
    ##   2                  125      0.9401382  0.11618951  0.6639579
    ##   3                   25      0.9507596  0.10289985  0.6726348
    ##   3                   50      0.9435272  0.11019261  0.6664037
    ##   3                   75      0.9412401  0.11417189  0.6647448
    ##   3                  100      0.9413001  0.11437480  0.6649426
    ##   3                  125      0.9417480  0.11369590  0.6645094
    ##   4                   25      0.9483031  0.10621153  0.6714922
    ##   4                   50      0.9421649  0.11258064  0.6666425
    ##   4                   75      0.9395653  0.11773687  0.6646818
    ##   4                  100      0.9403284  0.11665282  0.6651844
    ##   4                  125      0.9430764  0.11246958  0.6670272
    ##   5                   25      0.9480456  0.10419737  0.6707364
    ##   5                   50      0.9449025  0.10738534  0.6683985
    ##   5                   75      0.9448860  0.10819393  0.6688973
    ##   5                  100      0.9463410  0.10653747  0.6702411
    ##   5                  125      0.9488864  0.10314806  0.6721808
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 75, interaction.depth = 4, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = select(testTransformed, -log.shares))

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse3, cv_rmse4, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Random Forest Model", "Boosted Tree Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Model 1      | 0.9794 |   0.0845 | 0.6941 |
| Linear Model 2      | 0.9793 |   0.0846 | 0.6942 |
| Random Forest Model | 0.9701 |   0.1021 | 0.6855 |
| Boosted Tree Model  | 0.9701 |   0.1023 | 0.6866 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the world channel
is “need to automate this part”.

The best model fit to predict the number of shares

# Final Model

Automation is done with the modifications of the YAML header and the
render function.
