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

    ## # A tibble: 4,382 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg_sh~ average_token_len~ n_tokens_content n_tokens_title global_subjectiv~ num_imgs
    ##         <dbl>   <dbl>      <dbl>  <dbl>                  <dbl>              <dbl>            <dbl>          <dbl>             <dbl>    <dbl>
    ##  1       6.57       1         0  0.0501                     0                4.91              255              9             0.341        1
    ##  2       8.04       1         0  0.0333                     0                5.45              397              8             0.374        1
    ##  3       6.75       1         0  0.0500                  2800                4.42              244             13             0.332        1
    ##  4       8.07       1         0  0.0286                  6100                4.62              708              8             0.491        1
    ##  5       6.35       1         0  0.239                      0                4.27              142             10             0.443        1
    ##  6       6.71       1         0  0.0200                   997.               4.81              444             12             0.462       23
    ##  7       7.60       3       802. 0.0500                  2000                4.64              233              9             0.183        1
    ##  8       7.55       3       642. 0.0400                  1200.               4.53              468             10             0.438        1
    ##  9       7.55       3       955. 0.0200                  2100                4.94              173             11             0.622        1
    ## 10       6.47       3       930. 0.0286                  4700                4.66              330              9             0.355        1
    ## # ... with 4,372 more rows

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# Data

When a subset of data is selected for the bus channel articles which
contain 6258 articles, the subset of data is then split into a training
set (70% of the subset data) and a test set (30% of the subset data)
based on the target variable, the number of shares. There are 4382
articles in the training set and 1876 observations in the test set
regarding the bus channel. The `createDataPartition` function from the
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

The bus channel has 4382 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from bus channel falls into
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

Table 4 shows the numerical summaries of average keywords from bus
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
| Unpopular |    393 |     424 |       491 |      421 |    264 |       16 |     52 |
| Popular   |    428 |     401 |       418 |      430 |    312 |      149 |    183 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   4191.832 | 33278.159 |          1400 |        7.4109 |       0.8955 |           7.2442 |
| Tuesday   |   2934.463 | 12066.613 |          1300 |        7.3526 |       0.8401 |           7.1701 |
| Wednesday |   2714.173 |  8858.365 |          1300 |        7.3091 |       0.8298 |           7.1701 |
| Thursday  |   2997.910 | 15204.307 |          1400 |        7.3471 |       0.7992 |           7.2442 |
| Friday    |   2467.287 |  5969.008 |          1450 |        7.3906 |       0.7835 |           7.2787 |
| Saturday  |   4951.697 | 12055.111 |          2700 |        8.0058 |       0.7943 |           7.9010 |
| Sunday    |   3500.425 |  5376.677 |          2100 |        7.7978 |       0.7243 |           7.6497 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    2935.819 |   1861.488 |       2755.417 |   1066.8790 |
| Tuesday   |    2922.067 |   1114.094 |       2753.349 |    971.1080 |
| Wednesday |    2885.239 |   1070.991 |       2746.641 |    910.7842 |
| Thursday  |    2896.107 |   1245.945 |       2727.643 |   1004.7583 |
| Friday    |    2996.119 |   1969.579 |       2742.287 |    901.6783 |
| Saturday  |    3609.733 |   2763.920 |       3353.134 |   1187.8104 |
| Sunday    |    3193.893 |   1040.154 |       3129.464 |   1179.2366 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      7003.652 |    34130.026 |         1950.000 |      3766.500 |
| Tuesday   |      5369.814 |    14573.626 |         2066.667 |      3230.000 |
| Wednesday |      7903.172 |    40798.364 |         2100.000 |      3789.000 |
| Thursday  |      5449.221 |    17830.870 |         2050.000 |      4100.000 |
| Friday    |      5637.258 |    17078.782 |         2009.071 |      3107.750 |
| Saturday  |      3550.522 |     8456.240 |         1300.000 |      3900.000 |
| Sunday    |      3113.325 |     5203.998 |         1510.667 |      3441.667 |

Table 5. Summary of Average shares of referenced articles in Mashable on
Day of the Week

``` r
edadata %>% group_by(class.shares) %>% summarize(
  Avg.subjectivity = mean(global_subjectivity), Sd.subjectivity = sd(global_subjectivity), 
  Median.subjectivity = median(global_subjectivity)) %>% kable(digits = 4, caption = "Table 5. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4268 |          0.0840 |              0.4281 |
| Popular      |           0.4447 |          0.0838 |              0.4498 |

Table 5. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 6. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     1.8794 |    3.9185 |             1 |         0.7841 |        0.5778 |            0.6931 |
| Tuesday   |     1.8158 |    3.5890 |             1 |         0.7778 |        0.5676 |            0.6931 |
| Wednesday |     1.6458 |    3.0517 |             1 |         0.7629 |        0.5222 |            0.6931 |
| Thursday  |     1.8273 |    3.2398 |             1 |         0.7956 |        0.5694 |            0.6931 |
| Friday    |     1.8038 |    3.6596 |             1 |         0.7776 |        0.5573 |            0.6931 |
| Saturday  |     2.2545 |    4.9271 |             1 |         0.8832 |        0.6022 |            0.6931 |
| Sunday    |     2.1277 |    4.3344 |             1 |         0.8647 |        0.5951 |            0.6931 |

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

![](../images/bus1.pngunnamed-chunk-3-1.png)<!-- -->

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

![](../images/bus2.pngunnamed-chunk-4-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "3.png")
```

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the bus channel on mashable.com on any day of the week. The measurements
of the different LDA topics are in ratios, and these are the mean ratios
calculated for the specific day of thte week for that topic across 709
days of collections of data in mashable.com. These mean ratios are
further classified into a “popular” group and an “unpopular” group
according to their number of shares.

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

![](../images/bus3.pngunnamed-chunk-5-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "4.png")
```

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the bus channel.

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

![](../images/bus4.pngunnamed-chunk-6-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "5.png")
```

### Scatterplot

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the bus channel across
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

![](../images/bus5.pngunnamed-chunk-7-1.png)<!-- -->

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

![](../images/bus6.pngunnamed-chunk-8-1.png)<!-- -->

``` r
file.name <- paste0("../images/", params$channel, "7a.png")
```

### QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the bus channel in figures 7a, 7b,
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

![](../images/bus7a.pngunnamed-chunk-9-1.png)<!-- -->

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

![](../images/bus7b.pngunnamed-chunk-10-1.png)<!-- -->

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

![](../images/bus7c.pngunnamed-chunk-11-1.png)<!-- -->

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

![](../images/bus7d.pngunnamed-chunk-12-1.png)<!-- -->

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
    ## -9.2511 -0.5863 -0.1536  0.4297  6.8695 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 1.650e-02  3.348e-02   0.493 0.622249    
    ## dayweek2                                   -5.905e-02  4.634e-02  -1.274 0.202650    
    ## dayweek3                                   -9.887e-02  4.526e-02  -2.184 0.028987 *  
    ## dayweek4                                   -5.407e-02  4.602e-02  -1.175 0.240080    
    ## dayweek5                                   -2.661e-02  5.110e-02  -0.521 0.602516    
    ## dayweek6                                    5.467e-01  8.125e-02   6.729 1.94e-11 ***
    ## dayweek7                                    3.877e-01  7.079e-02   5.476 4.59e-08 ***
    ## kw_avg_avg                                  1.573e-01  1.556e-02  10.108  < 2e-16 ***
    ## LDA_02                                     -1.796e-02  1.433e-02  -1.253 0.210159    
    ## self_reference_avg_sharess                  2.368e-01  3.068e-02   7.719 1.45e-14 ***
    ## average_token_length                       -7.358e-02  2.201e-02  -3.343 0.000835 ***
    ## n_tokens_content                            1.424e-01  2.081e-02   6.845 8.74e-12 ***
    ## n_tokens_title                             -7.969e-05  1.439e-02  -0.006 0.995581    
    ## global_subjectivity                         8.696e-02  1.536e-02   5.663 1.58e-08 ***
    ## num_imgs                                    3.554e-02  1.472e-02   2.414 0.015839 *  
    ## `I(n_tokens_content^2)`                    -3.302e-03  5.162e-03  -0.640 0.522359    
    ## `I(self_reference_avg_sharess^2)`          -1.053e-02  1.614e-03  -6.527 7.49e-11 ***
    ## `kw_avg_avg:num_imgs`                       7.202e-02  1.926e-02   3.740 0.000187 ***
    ## `average_token_length:global_subjectivity` -4.605e-04  5.808e-03  -0.079 0.936811    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.939 on 4363 degrees of freedom
    ## Multiple R-squared:  0.1218, Adjusted R-squared:  0.1182 
    ## F-statistic: 33.62 on 18 and 4363 DF,  p-value: < 2.2e-16

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
    ## -9.1793 -0.5861 -0.1525  0.4236  6.9393 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 3.554e-02  3.388e-02   1.049 0.294211    
    ## dayweek2                                   -5.927e-02  4.630e-02  -1.280 0.200607    
    ## dayweek3                                   -1.001e-01  4.522e-02  -2.214 0.026859 *  
    ## dayweek4                                   -5.528e-02  4.598e-02  -1.202 0.229306    
    ## dayweek5                                   -2.577e-02  5.105e-02  -0.505 0.613640    
    ## dayweek6                                    5.332e-01  8.118e-02   6.567 5.72e-11 ***
    ## dayweek7                                    3.851e-01  7.075e-02   5.444 5.50e-08 ***
    ## kw_avg_avg                                  1.911e-01  1.943e-02   9.834  < 2e-16 ***
    ## LDA_02                                     -1.757e-02  1.432e-02  -1.227 0.219902    
    ## self_reference_avg_sharess                  2.404e-01  3.068e-02   7.837 5.75e-15 ***
    ## average_token_length                       -7.418e-02  2.199e-02  -3.374 0.000747 ***
    ## n_tokens_content                            1.324e-01  2.113e-02   6.265 4.10e-10 ***
    ## n_tokens_title                              1.334e-03  1.438e-02   0.093 0.926086    
    ## global_subjectivity                         8.746e-02  1.535e-02   5.698 1.29e-08 ***
    ## `I(log(num_imgs + 1))`                      1.041e-01  3.320e-02   3.135 0.001729 ** 
    ## `I(n_tokens_content^2)`                    -2.490e-03  5.158e-03  -0.483 0.629271    
    ## `I(self_reference_avg_sharess^2)`          -1.074e-02  1.614e-03  -6.655 3.18e-11 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           1.255e-01  3.228e-02   3.889 0.000102 ***
    ## `average_token_length:global_subjectivity` -5.183e-05  5.799e-03  -0.009 0.992869    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9382 on 4363 degrees of freedom
    ## Multiple R-squared:  0.1234, Adjusted R-squared:  0.1198 
    ## F-statistic: 34.12 on 18 and 4363 DF,  p-value: < 2.2e-16

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
| Model 1 | 0.9290 |   0.1292 | 0.6861 |
| Model 2 | 0.9286 |   0.1300 | 0.6851 |

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

    ## # A tibble: 4,382 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       6.57         0  0.0501                 1                  0                    0                0                0                0
    ##  2       8.04         0  0.0333                 1                  0                    0                0                0                0
    ##  3       6.75         0  0.0500                 1                  0                    0                0                0                0
    ##  4       8.07         0  0.0286                 1                  0                    0                0                0                0
    ##  5       6.35         0  0.239                  1                  0                    0                0                0                0
    ##  6       6.71         0  0.0200                 1                  0                    0                0                0                0
    ##  7       7.60       802. 0.0500                 0                  0                    1                0                0                0
    ##  8       7.55       642. 0.0400                 0                  0                    1                0                0                0
    ##  9       7.55       955. 0.0200                 0                  0                    1                0                0                0
    ## 10       6.47       930. 0.0286                 0                  0                    1                0                0                0
    ## # ... with 4,372 more rows, and 7 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, average_token_length <dbl>,
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
    ## 4382 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 3506, 3505, 3506, 3505, 3506 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared   MAE      
    ##   1     0.9425512  0.1528951  0.6840804
    ##   2     0.9204825  0.1549393  0.6597427
    ##   3     0.9200469  0.1533686  0.6590971
    ##   4     0.9227270  0.1497970  0.6605752
    ##   5     0.9260267  0.1454681  0.6621405
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 3.

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
    ## 4382 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 3944, 3944, 3943, 3944, 3943, 3945, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared   MAE      
    ##   1                   25      0.9430243  0.1259381  0.6844696
    ##   1                   50      0.9288943  0.1437167  0.6716308
    ##   1                   75      0.9235522  0.1499076  0.6671662
    ##   1                  100      0.9231783  0.1494978  0.6659253
    ##   1                  125      0.9228617  0.1504148  0.6641560
    ##   2                   25      0.9325462  0.1389254  0.6733977
    ##   2                   50      0.9243914  0.1473441  0.6652644
    ##   2                   75      0.9209872  0.1534044  0.6614978
    ##   2                  100      0.9204215  0.1546858  0.6605791
    ##   2                  125      0.9207667  0.1542223  0.6595001
    ##   3                   25      0.9271224  0.1479288  0.6691127
    ##   3                   50      0.9226711  0.1511548  0.6627568
    ##   3                   75      0.9211107  0.1539756  0.6612418
    ##   3                  100      0.9201528  0.1561909  0.6600586
    ##   3                  125      0.9217594  0.1538696  0.6598955
    ##   4                   25      0.9257059  0.1478073  0.6668220
    ##   4                   50      0.9201169  0.1552136  0.6603124
    ##   4                   75      0.9184935  0.1590051  0.6572393
    ##   4                  100      0.9199378  0.1571475  0.6568558
    ##   4                  125      0.9208548  0.1562988  0.6566704
    ##   5                   25      0.9283737  0.1411771  0.6688243
    ##   5                   50      0.9236709  0.1493505  0.6632973
    ##   5                   75      0.9219827  0.1549202  0.6608557
    ##   5                  100      0.9248074  0.1512966  0.6613869
    ##   5                  125      0.9252987  0.1517203  0.6603614
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
| Linear Model 1      | 0.9290 |   0.1292 | 0.6861 |
| Linear Model 2      | 0.9286 |   0.1300 | 0.6851 |
| Random Forest Model | 0.8917 |   0.1996 | 0.6555 |
| Boosted Tree Model  | 0.8941 |   0.1938 | 0.6584 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the bus channel
is “need to automate this part”.

The best model fit to predict the number of shares

# Final Model

Automation is done with the modifications of the YAML header and the
render function.
