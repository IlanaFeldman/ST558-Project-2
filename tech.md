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

    ## # A tibble: 5,145 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg_sh~ average_token_len~ n_tokens_content n_tokens_title global_subjectiv~ num_imgs
    ##         <dbl>   <dbl>      <dbl>  <dbl>                  <dbl>              <dbl>            <dbl>          <dbl>             <dbl>    <dbl>
    ##  1       6.22       1          0 0.0286                  3151.               4.68             1072             13             0.514       20
    ##  2       6.75       1          0 0.0222                  8500                4.36              370             10             0.437        0
    ##  3       9.75       1          0 0.0250                  2830.               4.72             1207              8             0.539       42
    ##  4       7.94       1          0 0.0200                  3151.               4.69             1248             13             0.507       20
    ##  5       6.10       1          0 0.0286                  3151.               4.63             1154             11             0.534       20
    ##  6       6.66       1          0 0.0202                   924                4.26              266              8             0.313        1
    ##  7       7.31       1          0 0.133                   2500                4.78              331              8             0.561        1
    ##  8       7.50       1          0 0.0222                  3036.               4.64             1225             12             0.481       28
    ##  9       8.27       1          0 0.0286                     0                4.99              633             10             0.532       19
    ## 10       8.95       1          0 0.0200                  3713.               4.75             1244             10             0.512       20
    ## # ... with 5,135 more rows

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# Data

When a subset of data is selected for the tech channel articles which
contain 7346 articles, the subset of data is then split into a training
set (70% of the subset data) and a test set (30% of the subset data)
based on the target variable, the number of shares. There are 5145
articles in the training set and 2201 observations in the test set
regarding the tech channel. The `createDataPartition` function from the
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

The tech channel has 5145 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from tech channel falls into
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

Table 4 shows the numerical summaries of average keywords from tech
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
| Unpopular |    326 |     413 |       430 |      362 |    221 |       59 |     51 |
| Popular   |    503 |     614 |       606 |      535 |    472 |      323 |    230 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   2908.466 |  4253.489 |          1600 |        7.5448 |       0.8311 |           7.3778 |
| Tuesday   |   2899.634 |  5162.654 |          1600 |        7.5184 |       0.8208 |           7.3778 |
| Wednesday |   3562.874 | 21010.806 |          1600 |        7.5367 |       0.8637 |           7.3778 |
| Thursday  |   2747.845 |  4294.657 |          1600 |        7.4975 |       0.7968 |           7.3778 |
| Friday    |   3164.838 |  5920.596 |          1800 |        7.6145 |       0.8060 |           7.4955 |
| Saturday  |   3750.105 |  6010.826 |          2300 |        7.8771 |       0.7426 |           7.7407 |
| Sunday    |   3692.751 |  4119.956 |          2300 |        7.8667 |       0.7663 |           7.7407 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    2681.119 |   812.6908 |       2659.370 |    804.1680 |
| Tuesday   |    2769.896 |   693.9189 |       2708.336 |    727.7346 |
| Wednesday |    2708.965 |   776.8135 |       2658.664 |    731.7624 |
| Thursday  |    2775.331 |   836.8488 |       2721.222 |    703.3169 |
| Friday    |    2760.874 |   615.6428 |       2701.512 |    720.6229 |
| Saturday  |    2868.073 |   633.2257 |       2817.810 |    746.2075 |
| Sunday    |    2791.215 |   564.6896 |       2767.627 |    691.5831 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      5783.178 |    13788.070 |         2833.333 |      3750.000 |
| Tuesday   |      7051.978 |    37302.274 |         2800.000 |      3750.000 |
| Wednesday |      6922.696 |    29799.508 |         2671.372 |      3594.682 |
| Thursday  |      8171.512 |    38021.828 |         2816.667 |      4080.000 |
| Friday    |      9233.143 |    49155.884 |         2900.000 |      3972.750 |
| Saturday  |      6766.594 |    20936.535 |         2922.000 |      3453.756 |
| Sunday    |      4432.695 |     5580.107 |         2660.000 |      3892.667 |

Table 5. Summary of Average shares of referenced articles in Mashable on
Day of the Week

``` r
edadata %>% group_by(class.shares) %>% summarize(
  Avg.subjectivity = mean(global_subjectivity), Sd.subjectivity = sd(global_subjectivity), 
  Median.subjectivity = median(global_subjectivity)) %>% kable(digits = 4, caption = "Table 5. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4536 |          0.0752 |              0.4560 |
| Popular      |           0.4579 |          0.0748 |              0.4606 |

Table 5. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 6. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     5.4572 |    8.8006 |             1 |         1.2199 |        1.0447 |            0.6931 |
| Tuesday   |     4.3262 |    6.9044 |             1 |         1.1509 |        0.9281 |            0.6931 |
| Wednesday |     4.3465 |    6.7674 |             1 |         1.1528 |        0.9351 |            0.6931 |
| Thursday  |     4.1159 |    7.0783 |             1 |         1.0978 |        0.9153 |            0.6931 |
| Friday    |     3.9351 |    6.2886 |             1 |         1.0635 |        0.9396 |            0.6931 |
| Saturday  |     5.2356 |    6.8153 |             2 |         1.3446 |        0.9888 |            1.0986 |
| Sunday    |     4.0107 |    6.2607 |             1 |         1.0753 |        0.9737 |            0.6931 |

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

![](../images/tech1.pngunnamed-chunk-3-1.png)<!-- -->

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

![](../images/tech2.pngunnamed-chunk-4-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "3.png")
```

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the tech channel on mashable.com on any day of the week. The
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

![](../images/tech3.pngunnamed-chunk-5-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "4.png")
```

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the tech channel.

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

![](../images/tech4.pngunnamed-chunk-6-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "5.png")
```

### Scatterplot

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the tech channel across
different days of the week. In the news popularity study, it says
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

![](../images/tech5.pngunnamed-chunk-7-1.png)<!-- -->

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

![](../images/tech6.pngunnamed-chunk-8-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7a.png")
```

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the tech channel in figures 7a, 7b,
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

![](../images/tech7a.pngunnamed-chunk-9-1.png)<!-- -->

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

![](../images/tech7b.pngunnamed-chunk-10-1.png)<!-- -->

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

![](../images/tech7c.pngunnamed-chunk-11-1.png)<!-- -->

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

![](../images/tech7d.pngunnamed-chunk-12-1.png)<!-- -->

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
    ## -4.7280 -0.6298 -0.1741  0.4828  6.6578 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0109446  0.0342979   0.319   0.7497    
    ## dayweek2                                   -0.0531213  0.0451936  -1.175   0.2399    
    ## dayweek3                                   -0.0356037  0.0450996  -0.789   0.4299    
    ## dayweek4                                   -0.0898530  0.0466314  -1.927   0.0541 .  
    ## dayweek5                                    0.0665131  0.0498366   1.335   0.1821    
    ## dayweek6                                    0.3181926  0.0599748   5.305 1.17e-07 ***
    ## dayweek7                                    0.3452883  0.0668671   5.164 2.51e-07 ***
    ## kw_avg_avg                                  0.1433292  0.0136555  10.496  < 2e-16 ***
    ## LDA_02                                     -0.0176936  0.0139002  -1.273   0.2031    
    ## self_reference_avg_sharess                  0.2578742  0.0370148   6.967 3.65e-12 ***
    ## average_token_length                        0.0004136  0.0182154   0.023   0.9819    
    ## n_tokens_content                            0.1605048  0.0229253   7.001 2.86e-12 ***
    ## n_tokens_title                             -0.0316179  0.0136082  -2.323   0.0202 *  
    ## global_subjectivity                         0.0155429  0.0141009   1.102   0.2704    
    ## num_imgs                                   -0.0041664  0.0163919  -0.254   0.7994    
    ## `I(n_tokens_content^2)`                    -0.0139974  0.0057229  -2.446   0.0145 *  
    ## `I(self_reference_avg_sharess^2)`          -0.0133321  0.0021746  -6.131 9.40e-10 ***
    ## `kw_avg_avg:num_imgs`                       0.0605058  0.0140011   4.321 1.58e-05 ***
    ## `average_token_length:global_subjectivity`  0.0110712  0.0045466   2.435   0.0149 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9642 on 5126 degrees of freedom
    ## Multiple R-squared:  0.07354,    Adjusted R-squared:  0.07029 
    ## F-statistic:  22.6 on 18 and 5126 DF,  p-value: < 2.2e-16

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
    ## -4.7236 -0.6292 -0.1758  0.4840  6.6621 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0059743  0.0343445   0.174   0.8619    
    ## dayweek2                                   -0.0510768  0.0451149  -1.132   0.2576    
    ## dayweek3                                   -0.0332413  0.0450108  -0.739   0.4602    
    ## dayweek4                                   -0.0863413  0.0465576  -1.855   0.0637 .  
    ## dayweek5                                    0.0684074  0.0497736   1.374   0.1694    
    ## dayweek6                                    0.3198482  0.0598889   5.341 9.66e-08 ***
    ## dayweek7                                    0.3494385  0.0667765   5.233 1.73e-07 ***
    ## kw_avg_avg                                  0.1733491  0.0148601  11.665  < 2e-16 ***
    ## LDA_02                                     -0.0176858  0.0139211  -1.270   0.2040    
    ## self_reference_avg_sharess                  0.2591397  0.0369965   7.004 2.80e-12 ***
    ## average_token_length                       -0.0009267  0.0182105  -0.051   0.9594    
    ## n_tokens_content                            0.1585392  0.0238432   6.649 3.25e-11 ***
    ## n_tokens_title                             -0.0310462  0.0136035  -2.282   0.0225 *  
    ## global_subjectivity                         0.0152918  0.0140842   1.086   0.2776    
    ## `I(log(num_imgs + 1))`                     -0.0047626  0.0243197  -0.196   0.8447    
    ## `I(n_tokens_content^2)`                    -0.0137116  0.0058391  -2.348   0.0189 *  
    ## `I(self_reference_avg_sharess^2)`          -0.0133731  0.0021735  -6.153 8.19e-10 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.0949261  0.0188638   5.032 5.01e-07 ***
    ## `average_token_length:global_subjectivity`  0.0108680  0.0045442   2.392   0.0168 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9637 on 5126 degrees of freedom
    ## Multiple R-squared:  0.07461,    Adjusted R-squared:  0.07136 
    ## F-statistic: 22.96 on 18 and 5126 DF,  p-value: < 2.2e-16

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
| Model 1 | 0.9327 |   0.0438 | 0.7224 |
| Model 2 | 0.9328 |   0.0442 | 0.7222 |

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

    ## # A tibble: 5,145 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       6.22          0 0.0286                 1                  0                    0                0                0                0
    ##  2       6.75          0 0.0222                 1                  0                    0                0                0                0
    ##  3       9.75          0 0.0250                 1                  0                    0                0                0                0
    ##  4       7.94          0 0.0200                 1                  0                    0                0                0                0
    ##  5       6.10          0 0.0286                 1                  0                    0                0                0                0
    ##  6       6.66          0 0.0202                 1                  0                    0                0                0                0
    ##  7       7.31          0 0.133                  1                  0                    0                0                0                0
    ##  8       7.50          0 0.0222                 1                  0                    0                0                0                0
    ##  9       8.27          0 0.0286                 1                  0                    0                0                0                0
    ## 10       8.95          0 0.0200                 1                  0                    0                0                0                0
    ## # ... with 5,135 more rows, and 7 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, average_token_length <dbl>,
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
    ## 5145 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 4117, 4116, 4115, 4117, 4115 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared    MAE      
    ##   1     0.9678298  0.09305648  0.7372635
    ##   2     0.9532411  0.09286516  0.7261887
    ##   3     0.9540791  0.09046151  0.7281295
    ##   4     0.9557310  0.08867220  0.7301363
    ##   5     0.9592102  0.08392537  0.7328861
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
    ## 5145 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4630, 4631, 4630, 4630, 4631, 4630, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9677709  0.07560556  0.7394042
    ##   1                   50      0.9600277  0.08330768  0.7321140
    ##   1                   75      0.9573736  0.08616326  0.7293200
    ##   1                  100      0.9568564  0.08661070  0.7285798
    ##   1                  125      0.9564751  0.08702578  0.7277437
    ##   2                   25      0.9598980  0.08622664  0.7326185
    ##   2                   50      0.9540789  0.09287852  0.7268021
    ##   2                   75      0.9538536  0.09197109  0.7259325
    ##   2                  100      0.9534890  0.09238571  0.7256380
    ##   2                  125      0.9532163  0.09305200  0.7257256
    ##   3                   25      0.9564313  0.09114211  0.7287591
    ##   3                   50      0.9524720  0.09456191  0.7254204
    ##   3                   75      0.9525379  0.09384730  0.7251303
    ##   3                  100      0.9527156  0.09418413  0.7246607
    ##   3                  125      0.9544042  0.09133586  0.7258212
    ##   4                   25      0.9575582  0.08759331  0.7295336
    ##   4                   50      0.9532724  0.09324465  0.7251196
    ##   4                   75      0.9527398  0.09408741  0.7245854
    ##   4                  100      0.9531114  0.09416916  0.7240543
    ##   4                  125      0.9546012  0.09242341  0.7250301
    ##   5                   25      0.9548732  0.09205278  0.7283430
    ##   5                   50      0.9526365  0.09432718  0.7257803
    ##   5                   75      0.9552225  0.09001609  0.7275190
    ##   5                  100      0.9566218  0.08881838  0.7283730
    ##   5                  125      0.9576450  0.08773931  0.7282879
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = select(testTransformed, -log.shares))

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse3, cv_rmse4, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Random Forest Model", "Boosted Tree Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Model 1      | 0.9327 |   0.0438 | 0.7224 |
| Linear Model 2      | 0.9328 |   0.0442 | 0.7222 |
| Random Forest Model | 0.9081 |   0.0854 | 0.7062 |
| Boosted Tree Model  | 0.9140 |   0.0746 | 0.7117 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the tech channel
is “need to automate this part”.

The best model fit to predict the number of shares

# Final Model

Automation is done with the modifications of the YAML header and the
render function.
