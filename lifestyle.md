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

    | Attribute                     | Attribute Information                              | Type   

— \| —————————– \| ————————————————– \| ——– 1 \| `shares` (target) \|
Number of shares \| number 2 \| `kw_avg_avg` \| Average keyword (average
shares) \| number 3 \| `LDA_02` \| Closeness to LDA topic 2 \| ratio 4.1
\| `weekday_is_monday` \| Was the article published on a Monday? \|
boolean 4.2 \| `weekday_is_tuesday` \| Was the article published on a
Tuesday? \| boolean 4.3 \| `weekday_is_wednesday` \| Was the article
published on a Wednesday? \| boolean 4.4 \| `weekday_is_thursday` \| Was
the article published on a Thursday? \| boolean 4.5 \|
`weekday_is_friday` \| Was the article published on a Friday? \| boolean
4.6 \| `weekday_is_saturday` \| Was the article published on a Saturday?
\| boolean 4.7 \| `weekday_is_sunday` \| Was the article published on a
Sunday? \| boolean 5 \| `self_reference_avg_sharess` \| Average shares
of referenced articles in Mashable \| number 6 \| `average_token_length`
\| Average length of the words in the content \| number 7 \|
`n_tokens_content` \| Number of words in the content \| number 8 \|
`n_tokens_title` \| Number of words in the title \| number 9 \|
`global_subjectivity` \| Text subjectivity \| ratio 10 \| `num_imgs` \|
Number of images \| number

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

    ## # A tibble: 1,472 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg_sh~ average_token_len~ n_tokens_content n_tokens_title global_subjectiv~ num_imgs
    ##         <dbl>   <dbl>      <dbl>  <dbl>                  <dbl>              <dbl>            <dbl>          <dbl>             <dbl>    <dbl>
    ##  1       6.32       1         0  0.0200                  3151.               4.65              960              8             0.514       20
    ##  2       7.55       1         0  0.0286                     0                4.66              187             10             0.477        1
    ##  3       8.65       1         0  0.0335                  5000                4.84              103             11             0.424        1
    ##  4       6.14       1         0  0.0200                     0                4.38              243             10             0.518        0
    ##  5       8.19       1         0  0.0251                     0                4.67              204              8             0.652        1
    ##  6       5.84       1         0  0.0205                  6200                4.38              315             11             0.554        1
    ##  7       6.23       1         0  0.0250                  3151.               4.62             1190             10             0.507       20
    ##  8       6.31       1         0  0.276                      0                4.91              374              6             0.399        1
    ##  9       7.09       2       885. 0.120                   1300                5.08              499             12             0.395        1
    ## 10       7.55       3      1207. 0.0334                 11700                4.55              223             11             0.372        0
    ## # ... with 1,462 more rows

``` r
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
boolean format are needed when we run the ensemble models.

# Exploratory Data Analysis

The lifestyle channel has 1472 articles collected. Now let us take a
look at the relationships between our response and the predictors with
some numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from lifestyle channel falls
into different categories on different days of the week for 709 days.

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

Table 4 shows the numerical summaries of average keywords from lifestyle
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
  Median.subjectivity = median(global_subjectivity)) %>% kable(digits = 4, caption = "Table 5. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4726 |          0.0853 |              0.4774 |
| Popular      |           0.4744 |          0.0983 |              0.4769 |

Table 5. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 6. Comparing Image Counts by the Day of the Week")
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

Table 6. Comparing Image Counts by the Day of the Week

``` r
#file.name <- paste0("../images/", params$channel, "1.png")
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

![](../images/lifestyleunnamed-chunk-3-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "2.png")
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
ggplot(data = edadata, aes(x = day.week, y = shares)) + 
  geom_boxplot(fill = "white", outlier.shape = NA) + 
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

![](../images/lifestyleunnamed-chunk-4-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "3.png")
```

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the lifestyle channel on mashable.com on any day of the week. The
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

![](../images/lifestyleunnamed-chunk-5-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "4.png")
```

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the lifestyle channel.

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

![](../images/lifestyleunnamed-chunk-6-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "5.png")
```

### Scatterplots

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the lifestyle channel
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

![](../images/lifestyleunnamed-chunk-7-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "6.png")
```

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

![](../images/lifestyleunnamed-chunk-8-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "7a.png")
```

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the lifestyle channel in figures 7a,
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

![](../images/lifestyleunnamed-chunk-9-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "7b.png")
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

![](../images/lifestyleunnamed-chunk-10-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "7c.png")
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

![](../images/lifestyleunnamed-chunk-11-1.png)<!-- -->

``` r
#file.name <- paste0("../images/", params$channel, "7d.png")
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

![](../images/lifestyleunnamed-chunk-12-1.png)<!-- -->

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
    ## -3.2979 -0.6390 -0.1937  0.4880  4.6284 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.0117707  0.0657779  -0.179  0.85801    
    ## dayweek2                                   -0.0830738  0.0908471  -0.914  0.36064    
    ## dayweek3                                   -0.1123462  0.0880283  -1.276  0.20207    
    ## dayweek4                                   -0.0006957  0.0892649  -0.008  0.99378    
    ## dayweek5                                   -0.0913123  0.0932089  -0.980  0.32742    
    ## dayweek6                                    0.2539322  0.1125410   2.256  0.02420 *  
    ## dayweek7                                    0.1219040  0.1014786   1.201  0.22984    
    ## kw_avg_avg                                  0.0901333  0.0288990   3.119  0.00185 ** 
    ## LDA_02                                     -0.0277557  0.0261959  -1.060  0.28953    
    ## self_reference_avg_sharess                  0.1076954  0.0441942   2.437  0.01493 *  
    ## average_token_length                        0.0361232  0.0499040   0.724  0.46927    
    ## n_tokens_content                            0.0930429  0.0370020   2.515  0.01203 *  
    ## n_tokens_title                              0.0221490  0.0255415   0.867  0.38599    
    ## global_subjectivity                        -0.0005028  0.0303908  -0.017  0.98680    
    ## num_imgs                                    0.0276859  0.0326272   0.849  0.39627    
    ## `I(n_tokens_content^2)`                    -0.0094252  0.0046839  -2.012  0.04438 *  
    ## `I(self_reference_avg_sharess^2)`          -0.0050261  0.0026706  -1.882  0.06003 .  
    ## `kw_avg_avg:num_imgs`                       0.1083358  0.0258405   4.192 2.93e-05 ***
    ## `average_token_length:global_subjectivity`  0.0233206  0.0123988   1.881  0.06019 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9671 on 1453 degrees of freedom
    ## Multiple R-squared:  0.07607,    Adjusted R-squared:  0.06463 
    ## F-statistic: 6.646 on 18 and 1453 DF,  p-value: 2.834e-16

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
    ## -3.2669 -0.6411 -0.1884  0.4965  4.5295 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.0123873  0.0671324  -0.185   0.8536    
    ## dayweek2                                   -0.0762782  0.0908620  -0.839   0.4013    
    ## dayweek3                                   -0.1105440  0.0881049  -1.255   0.2098    
    ## dayweek4                                    0.0018717  0.0893502   0.021   0.9833    
    ## dayweek5                                   -0.0908233  0.0932962  -0.973   0.3305    
    ## dayweek6                                    0.2533755  0.1125524   2.251   0.0245 *  
    ## dayweek7                                    0.1254995  0.1015641   1.236   0.2168    
    ## kw_avg_avg                                  0.1249407  0.0279278   4.474 8.29e-06 ***
    ## LDA_02                                     -0.0285191  0.0262023  -1.088   0.2766    
    ## self_reference_avg_sharess                  0.1085914  0.0441934   2.457   0.0141 *  
    ## average_token_length                        0.0429064  0.0497222   0.863   0.3883    
    ## n_tokens_content                            0.0958624  0.0377036   2.543   0.0111 *  
    ## n_tokens_title                              0.0219488  0.0255564   0.859   0.3906    
    ## global_subjectivity                         0.0002912  0.0306288   0.010   0.9924    
    ## `I(log(num_imgs + 1))`                      0.0339971  0.0427089   0.796   0.4262    
    ## `I(n_tokens_content^2)`                    -0.0068921  0.0045945  -1.500   0.1338    
    ## `I(self_reference_avg_sharess^2)`          -0.0051216  0.0026715  -1.917   0.0554 .  
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.1518899  0.0341319   4.450 9.24e-06 ***
    ## `average_token_length:global_subjectivity`  0.0254698  0.0123332   2.065   0.0391 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9676 on 1453 degrees of freedom
    ## Multiple R-squared:  0.07511,    Adjusted R-squared:  0.06365 
    ## F-statistic: 6.556 on 18 and 1453 DF,  p-value: 5.457e-16

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
| Model 1 | 0.9848 |   0.0530 | 0.7351 |
| Model 2 | 0.9866 |   0.0501 | 0.7363 |

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

    ## # A tibble: 1,472 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       6.32         0  0.0200                 1                  0                    0                0                0                0
    ##  2       7.55         0  0.0286                 1                  0                    0                0                0                0
    ##  3       8.65         0  0.0335                 1                  0                    0                0                0                0
    ##  4       6.14         0  0.0200                 1                  0                    0                0                0                0
    ##  5       8.19         0  0.0251                 1                  0                    0                0                0                0
    ##  6       5.84         0  0.0205                 1                  0                    0                0                0                0
    ##  7       6.23         0  0.0250                 1                  0                    0                0                0                0
    ##  8       6.31         0  0.276                  1                  0                    0                0                0                0
    ##  9       7.09       885. 0.120                  0                  1                    0                0                0                0
    ## 10       7.55      1207. 0.0334                 0                  0                    1                0                0                0
    ## # ... with 1,462 more rows, and 7 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, average_token_length <dbl>,
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
    ## 1472 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1177, 1176, 1179, 1177, 1179 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared    MAE      
    ##   1     0.9811805  0.04748241  0.7523925
    ##   2     0.9776730  0.04793521  0.7492909
    ##   3     0.9831137  0.04277405  0.7549683
    ##   4     0.9854245  0.04193763  0.7580638
    ##   5     0.9870863  0.04122588  0.7594749
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
    ## 1472 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1325, 1325, 1325, 1324, 1325, 1324, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9786030  0.04806353  0.7503070
    ##   1                   50      0.9789112  0.04437681  0.7503609
    ##   1                   75      0.9817670  0.04323250  0.7508578
    ##   1                  100      0.9840472  0.04136500  0.7535686
    ##   1                  125      0.9860287  0.03944421  0.7535652
    ##   2                   25      0.9794022  0.04721379  0.7517722
    ##   2                   50      0.9843330  0.03826902  0.7545293
    ##   2                   75      0.9902303  0.03332521  0.7576280
    ##   2                  100      0.9936666  0.03184637  0.7594043
    ##   2                  125      0.9977829  0.02990572  0.7636433
    ##   3                   25      0.9769114  0.05006842  0.7493482
    ##   3                   50      0.9837673  0.04264823  0.7522812
    ##   3                   75      0.9843967  0.04285284  0.7517982
    ##   3                  100      0.9878111  0.04235377  0.7534320
    ##   3                  125      0.9915350  0.03894286  0.7561844
    ##   4                   25      0.9828674  0.03672445  0.7518923
    ##   4                   50      0.9902483  0.03491000  0.7550739
    ##   4                   75      0.9930752  0.03574079  0.7582130
    ##   4                  100      0.9999453  0.03038588  0.7642757
    ##   4                  125      1.0003429  0.03115895  0.7649667
    ##   5                   25      0.9848068  0.03759095  0.7536839
    ##   5                   50      0.9984326  0.02532238  0.7641501
    ##   5                   75      1.0062245  0.02076886  0.7697086
    ##   5                  100      1.0082835  0.02464954  0.7702843
    ##   5                  125      1.0089440  0.02619955  0.7738755
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = select(testTransformed, -log.shares))

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse3, cv_rmse4, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Random Forest Model", "Boosted Tree Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Model 1      | 0.9848 |   0.0530 | 0.7351 |
| Linear Model 2      | 0.9866 |   0.0501 | 0.7363 |
| Random Forest Model | 0.9773 |   0.0682 | 0.7343 |
| Boosted Tree Model  | 0.9916 |   0.0397 | 0.7465 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the lifestyle
channel is “need to automate this part”.

The best model fit to predict the number of shares

# Final Model

Automation is done with the modifications of the YAML header and the
render function.
