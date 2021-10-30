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
-   [Modeling](#modeling)
    -   [Linear Regression](#linear-regression)
    -   [Random Forest](#random-forest)
    -   [Boosted Tree](#boosted-tree)
-   [Model Comparisons](#model-comparisons)
-   [Automation](#automation)

# Introduction

briefly describes the data briefly describes the variables you have to
work with (describe what you want to use)

purpose of the analysis methods you will use to model the response (more
details in modeling section)

61 variables (only 58 predictive variables, 2 non-predictive), target
response is “shares”.

# Data

| Predictors                   | Attribute Information                             | Type    |
|------------------------------|---------------------------------------------------|---------|
| `kw_avg_avg`                 | Average keyword (average shares)                  | number  |
| `LDA_02`                     | Closeness to LDA topic 2                          | ratio   |
| `weekday_is_monday`          | Was the article published on a Monday?            | boolean |
| `weekday_is_tuesday`         | Was the article published on a Tuesday?           | boolean |
| `weekday_is_wednesday`       | Was the article published on a Wednesday?         | boolean |
| `weekday_is_thursday`        | Was the article published on a Thursday?          | boolean |
| `weekday_is_friday`          | Was the article published on a Friday?            | boolean |
| `weekday_is_saturday`        | Was the article published on a Saturday?          | boolean |
| `weekday_is_sunday`          | Was the article published on a Sunday?            | boolean |
| `self_reference_avg_sharess` | Average shares of referenced articles in Mashable | number  |
| `n_non_stop_unique_tokens`   | Rate of unique non-stop words in the content      | ratio   |
| `average_token_length`       | Average length of the words in the content        | number  |
| `n_tokens_content`           | Number of words in the content                    | number  |
| `n_tokens_title`             | Number of words in the title                      | number  |
| `global_subjectivity`        | Text subjectivity                                 | ratio   |
| `num_imgs`                   | Number of images                                  | number  |

I created a binary response variable, 0 if shares &lt; 1400, 1 if shares
&gt; 1400. “class\_shares” (can use it in EDA)

I created a categorical variable grouped all binary variables, monday,
tuesday, …, sunday, together. “dayweek” if dayweek = 1, it’s Monday, 2
is tuesday, 3 is wednesday, …, 7 is sunday.

I created a log(shares) variable and use it as response instead of
shares. In office hour, a lot of people say this improved fit a little
better.

This analysis is based on the bus channel popularity.

``` r
library(tidyverse)
library(knitr)
library(caret)
library(corrplot)
library(ggplot2)
library(gbm)

allnews <- read_csv("../_Data/OnlineNewsPopularity.csv", 
                 col_names = TRUE)

########KNIT with parameters!!!!!!!!!channels is in quotes!!!!Need to use it with quotes!!!!!!!!!!!!!!!!!!!!!!!!

channels <- paste0("data_channel_is_", params$channel)
subnews <- allnews[allnews[, channels] == 1, ]

news <- subnews %>% select(
  -data_channel_is_lifestyle, -data_channel_is_entertainment, -data_channel_is_bus, -data_channel_is_socmed, 
  -data_channel_is_tech, -data_channel_is_world, -url, -timedelta)
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dim(news)
```

    ## [1] 6258   53

``` r
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
                               n_non_stop_unique_tokens, average_token_length, 
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

    ## # A tibble: 4,382 x 11
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_a~ n_non_stop_uniq~ average_token_l~ n_tokens_content n_tokens_title global_subjecti~
    ##         <dbl>   <dbl>      <dbl>  <dbl>             <dbl>            <dbl>            <dbl>            <dbl>          <dbl>            <dbl>
    ##  1       6.57       1         0  0.0501                0             0.792             4.91              255              9            0.341
    ##  2       8.04       1         0  0.0333                0             0.806             5.45              397              8            0.374
    ##  3       6.75       1         0  0.0500             2800             0.680             4.42              244             13            0.332
    ##  4       8.07       1         0  0.0286             6100             0.688             4.62              708              8            0.491
    ##  5       6.35       1         0  0.239                 0             0.792             4.27              142             10            0.443
    ##  6       6.71       1         0  0.0200              997.            0.755             4.81              444             12            0.462
    ##  7       7.60       3       802. 0.0500             2000             0.714             4.64              233              9            0.183
    ##  8       7.55       3       642. 0.0400             1200.            0.722             4.53              468             10            0.438
    ##  9       7.55       3       955. 0.0200             2100             0.804             4.94              173             11            0.622
    ## 10       6.47       3       930. 0.0286             4700             0.585             4.66              330              9            0.355
    ## # ... with 4,372 more rows, and 1 more variable: num_imgs <dbl>

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04) #keep log.shares
```

# Exploratory Data Analysis

The bus channel has 4382 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## Numerical Summaries

Table 1 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from bus channel falls into
different categories on different days of the week for 709 days.

Table 2 shows the average shares of the articles on different days of
the week. Here, we can see a potential problem for our analysis later.
Median shares are all very different from the average shares on any day
of the week. Recall that median is a robust measure for center. It is
robust to outliers in the data. On the contrary, mean is also a measure
of center but it is not robust to outliers. Mean measure can be
influenced by potential outliers.

In addition, Table 2 also shows the standard deviation of shares is huge
for any day of the week. They are potentially larger than the average
shares. This tells us the variance of shares for any day is huge. We
know a common variance stabilizing transformation to deal with
increasing variance of the response variable, that is, the
log-transformation, which could help us on this matter. Therefore, Table
2 again shows after the log-transformation of shares, the mean values
are similar to their corresponding median values, and their standard
deviations are much smaller than before relatively speaking.

Table 3 shows the numerical summaries of average keywords from bus
channel in mashable.com on different days of the week. Table 4 shows the
numerical summaries of average shares of referenced articles in
mashable.com on different days of the week.

``` r
# contingency table
edadata <- train
edadata$class.shares <- cut(edadata$class_shares, 2, c("Unpopular","Popular"))
edadata$day.week <- cut(edadata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
table(edadata$class.shares, edadata$day.week) %>% kable(caption = "Table 1. Popularity on Day of the Week")
```

|           | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|:----------|-------:|--------:|----------:|---------:|-------:|---------:|-------:|
| Unpopular |    393 |     424 |       491 |      421 |    264 |       16 |     52 |
| Popular   |    428 |     401 |       418 |      430 |    312 |      149 |    183 |

Table 1. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 2. Average Shares vs. Average Log(shares) on Day of the Week")
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

Table 2. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 3. Summary of Average Keywords on Day of the Week")
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

Table 3. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average shares of referenced articles in Mashable on Day of the Week")
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

Table 4. Summary of Average shares of referenced articles in Mashable on
Day of the Week

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
file.name <- paste0("../images/", params$channel, 1, ".png")
png(filename = file.name)

correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt")
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n", 
         cex = 0.8,
         title="Figure 1. Correlations Between the Variables")
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
file.name <- paste0("../images/", params$channel, 2, ".png")
png(filename = file.name)

boxplot1 <- ggplot(data = edadata, aes(x = day.week, y = shares))
boxplot1 + geom_boxplot(fill = "white", outlier.shape = NA) + 
  coord_cartesian(ylim=c(0, 10000)) + 
  geom_jitter(aes(color = day.week), size = 1) + 
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
file.name <- paste0("../images/", params$channel, 3, ".png")
png(filename = file.name)

b.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
b.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares LDA_0  LDA_1  LDA_2  LDA_3  LDA_4
    ##    <fct>     <fct>        <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ##  1 Monday    Unpopular    0.628 0.0811 0.0882 0.0593 0.143 
    ##  2 Monday    Popular      0.679 0.0707 0.0799 0.0695 0.101 
    ##  3 Tuesday   Unpopular    0.631 0.0841 0.0936 0.0707 0.120 
    ##  4 Tuesday   Popular      0.662 0.0743 0.0756 0.0754 0.113 
    ##  5 Wednesday Unpopular    0.632 0.0826 0.0886 0.0618 0.135 
    ##  6 Wednesday Popular      0.655 0.0767 0.0788 0.0717 0.118 
    ##  7 Thursday  Unpopular    0.640 0.0778 0.0813 0.0567 0.144 
    ##  8 Thursday  Popular      0.675 0.0663 0.0770 0.0716 0.110 
    ##  9 Friday    Unpopular    0.595 0.0894 0.100  0.0587 0.157 
    ## 10 Friday    Popular      0.663 0.0766 0.0860 0.0699 0.105 
    ## 11 Saturday  Unpopular    0.574 0.106  0.0836 0.111  0.125 
    ## 12 Saturday  Popular      0.753 0.0553 0.0696 0.0522 0.0703
    ## 13 Sunday    Unpopular    0.715 0.0704 0.0530 0.0675 0.0942
    ## 14 Sunday    Popular      0.715 0.0682 0.0588 0.0691 0.0890

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.628 
    ##  2 Monday   Unpopular    LDA_1      0.0811
    ##  3 Monday   Unpopular    LDA_2      0.0882
    ##  4 Monday   Unpopular    LDA_3      0.0593
    ##  5 Monday   Unpopular    LDA_4      0.143 
    ##  6 Monday   Popular      LDA_0      0.679 
    ##  7 Monday   Popular      LDA_1      0.0707
    ##  8 Monday   Popular      LDA_2      0.0799
    ##  9 Monday   Popular      LDA_3      0.0695
    ## 10 Monday   Popular      LDA_4      0.101 
    ## # ... with 60 more rows

``` r
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

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the bus channel.

``` r
file.name <- paste0("../images/", params$channel, 4, ".png")
png(filename = file.name)

l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares LDA_0  LDA_1  LDA_2  LDA_3  LDA_4
    ##    <fct>     <fct>        <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ##  1 Monday    Unpopular    0.628 0.0811 0.0882 0.0593 0.143 
    ##  2 Monday    Popular      0.679 0.0707 0.0799 0.0695 0.101 
    ##  3 Tuesday   Unpopular    0.631 0.0841 0.0936 0.0707 0.120 
    ##  4 Tuesday   Popular      0.662 0.0743 0.0756 0.0754 0.113 
    ##  5 Wednesday Unpopular    0.632 0.0826 0.0886 0.0618 0.135 
    ##  6 Wednesday Popular      0.655 0.0767 0.0788 0.0717 0.118 
    ##  7 Thursday  Unpopular    0.640 0.0778 0.0813 0.0567 0.144 
    ##  8 Thursday  Popular      0.675 0.0663 0.0770 0.0716 0.110 
    ##  9 Friday    Unpopular    0.595 0.0894 0.100  0.0587 0.157 
    ## 10 Friday    Popular      0.663 0.0766 0.0860 0.0699 0.105 
    ## 11 Saturday  Unpopular    0.574 0.106  0.0836 0.111  0.125 
    ## 12 Saturday  Popular      0.753 0.0553 0.0696 0.0522 0.0703
    ## 13 Sunday    Unpopular    0.715 0.0704 0.0530 0.0675 0.0942
    ## 14 Sunday    Popular      0.715 0.0682 0.0588 0.0691 0.0890

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.628 
    ##  2 Monday   Unpopular    LDA_1      0.0811
    ##  3 Monday   Unpopular    LDA_2      0.0882
    ##  4 Monday   Unpopular    LDA_3      0.0593
    ##  5 Monday   Unpopular    LDA_4      0.143 
    ##  6 Monday   Popular      LDA_0      0.679 
    ##  7 Monday   Popular      LDA_1      0.0707
    ##  8 Monday   Popular      LDA_2      0.0799
    ##  9 Monday   Popular      LDA_3      0.0695
    ## 10 Monday   Popular      LDA_4      0.101 
    ## # ... with 60 more rows

``` r
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

``` r
file.name <- paste0("../images/", params$channel, 5, ".png")
png(filename = file.name)

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

cv_fit1 <- train(log.shares ~ . , 
                 data=trainTransformed,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
summary(cv_fit1)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.2776 -0.5913 -0.1558  0.4300  6.8202 
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 0.0005424  0.0329744   0.016  0.98688    
    ## dayweek2                   -0.0529441  0.0465947  -1.136  0.25591    
    ## dayweek3                   -0.0975052  0.0455120  -2.142  0.03222 *  
    ## dayweek4                   -0.0478733  0.0462419  -1.035  0.30060    
    ## dayweek5                   -0.0192387  0.0513658  -0.375  0.70802    
    ## dayweek6                    0.5376443  0.0813914   6.606 4.43e-11 ***
    ## dayweek7                    0.3959351  0.0708014   5.592 2.38e-08 ***
    ## kw_avg_avg                  0.1543592  0.0150646  10.246  < 2e-16 ***
    ## LDA_02                     -0.0196921  0.0144077  -1.367  0.17176    
    ## self_reference_avg_sharess  0.0591097  0.0148995   3.967 7.39e-05 ***
    ## n_non_stop_unique_tokens    0.0388413  0.0199398   1.948  0.05149 .  
    ## average_token_length       -0.0847992  0.0162107  -5.231 1.76e-07 ***
    ## n_tokens_content            0.1520085  0.0186059   8.170 4.00e-16 ***
    ## n_tokens_title             -0.0005527  0.0143989  -0.038  0.96938    
    ## global_subjectivity         0.0853938  0.0151165   5.649 1.72e-08 ***
    ## num_imgs                    0.0481934  0.0150338   3.206  0.00136 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9444 on 4366 degrees of freedom
    ## Multiple R-squared:  0.1112, Adjusted R-squared:  0.1082 
    ## F-statistic: 36.43 on 15 and 4366 DF,  p-value: < 2.2e-16

``` r
cv_fit2 <- train(log.shares ~ . +I(n_tokens_content^2)+ kw_avg_avg:num_imgs + 
                   average_token_length:global_subjectivity + 
                   dayweek:self_reference_avg_sharess ,
                 data=trainTransformed,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
summary(cv_fit2)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.2680 -0.5866 -0.1584  0.4212  6.8197 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0079253  0.0335847   0.236 0.813459    
    ## dayweek2                                   -0.0495578  0.0465329  -1.065 0.286932    
    ## dayweek3                                   -0.0947038  0.0454336  -2.084 0.037178 *  
    ## dayweek4                                   -0.0486077  0.0461741  -1.053 0.292535    
    ## dayweek5                                   -0.0252221  0.0512797  -0.492 0.622847    
    ## dayweek6                                    0.5282009  0.0843711   6.260 4.21e-10 ***
    ## dayweek7                                    0.4101331  0.0789047   5.198 2.11e-07 ***
    ## kw_avg_avg                                  0.1693157  0.0161416  10.489  < 2e-16 ***
    ## LDA_02                                     -0.0182724  0.0143818  -1.271 0.203967    
    ## self_reference_avg_sharess                  0.0997818  0.0263468   3.787 0.000154 ***
    ## n_non_stop_unique_tokens                    0.0575922  0.0237381   2.426 0.015300 *  
    ## average_token_length                       -0.0786786  0.0221025  -3.560 0.000375 ***
    ## n_tokens_content                            0.1857464  0.0282917   6.565 5.80e-11 ***
    ## n_tokens_title                              0.0009904  0.0144513   0.069 0.945366    
    ## global_subjectivity                         0.0835534  0.0155605   5.370 8.30e-08 ***
    ## num_imgs                                    0.0456526  0.0151628   3.011 0.002620 ** 
    ## `I(n_tokens_content^2)`                    -0.0083690  0.0055780  -1.500 0.133593    
    ## `kw_avg_avg:num_imgs`                       0.0701775  0.0193472   3.627 0.000290 ***
    ## `average_token_length:global_subjectivity`  0.0055683  0.0065669   0.848 0.396518    
    ## `dayweek2:self_reference_avg_sharess`       0.0393344  0.0652624   0.603 0.546731    
    ## `dayweek3:self_reference_avg_sharess`      -0.0765505  0.0331089  -2.312 0.020820 *  
    ## `dayweek4:self_reference_avg_sharess`       0.0484500  0.0547025   0.886 0.375828    
    ## `dayweek5:self_reference_avg_sharess`      -0.1511824  0.0681108  -2.220 0.026494 *  
    ## `dayweek6:self_reference_avg_sharess`      -0.1101370  0.2331959  -0.472 0.636741    
    ## `dayweek7:self_reference_avg_sharess`       0.2255069  0.3169530   0.711 0.476823    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.942 on 4357 degrees of freedom
    ## Multiple R-squared:  0.1175, Adjusted R-squared:  0.1126 
    ## F-statistic: 24.16 on 24 and 4357 DF,  p-value: < 2.2e-16

``` r
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
    ## -9.2582 -0.5875 -0.1498  0.4269  6.8307 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 1.985e-02  3.349e-02   0.593 0.553492    
    ## dayweek2                                   -5.606e-02  4.634e-02  -1.210 0.226428    
    ## dayweek3                                   -9.742e-02  4.524e-02  -2.153 0.031344 *  
    ## dayweek4                                   -5.457e-02  4.599e-02  -1.187 0.235456    
    ## dayweek5                                   -2.850e-02  5.108e-02  -0.558 0.576925    
    ## dayweek6                                    5.451e-01  8.121e-02   6.712 2.17e-11 ***
    ## dayweek7                                    3.873e-01  7.075e-02   5.474 4.64e-08 ***
    ## kw_avg_avg                                  1.574e-01  1.555e-02  10.118  < 2e-16 ***
    ## LDA_02                                     -1.835e-02  1.432e-02  -1.281 0.200293    
    ## self_reference_avg_sharess                  2.354e-01  3.067e-02   7.675 2.03e-14 ***
    ## n_non_stop_unique_tokens                    5.633e-02  2.363e-02   2.383 0.017198 *  
    ## average_token_length                       -7.412e-02  2.200e-02  -3.369 0.000761 ***
    ## n_tokens_content                            1.877e-01  2.817e-02   6.663 3.02e-11 ***
    ## n_tokens_title                             -8.132e-05  1.438e-02  -0.006 0.995489    
    ## global_subjectivity                         8.176e-02  1.550e-02   5.274 1.40e-07 ***
    ## num_imgs                                    4.356e-02  1.510e-02   2.886 0.003926 ** 
    ## `I(n_tokens_content^2)`                    -8.222e-03  5.557e-03  -1.480 0.139040    
    ## `I(self_reference_avg_sharess^2)`          -1.048e-02  1.613e-03  -6.495 9.25e-11 ***
    ## `kw_avg_avg:num_imgs`                       7.305e-02  1.925e-02   3.794 0.000150 ***
    ## `average_token_length:global_subjectivity`  6.702e-03  6.537e-03   1.025 0.305307    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9385 on 4362 degrees of freedom
    ## Multiple R-squared:  0.123,  Adjusted R-squared:  0.1191 
    ## F-statistic: 32.18 on 19 and 4362 DF,  p-value: < 2.2e-16

``` r
result_tab <- data.frame(t(cv_fit1$results),t(cv_fit2$results), t(cv_fit3$results))
colnames(result_tab) <- c("Model 1","Model 2", "Model 3")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 | Model 3 |
|:-----------|--------:|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |  1.0000 |
| RMSE       |  0.9454 |  0.9526 |  0.9387 |
| Rsquared   |  0.1117 |  0.0965 |  0.1192 |
| MAE        |  0.6850 |  0.6857 |  0.6801 |
| RMSESD     |  0.0751 |  0.0514 |  0.0711 |
| RsquaredSD |  0.0475 |  0.0182 |  0.0376 |
| MAESD      |  0.0307 |  0.0283 |  0.0253 |

Cross Validation - Comparisons of the models in training set

``` r
pred1 <- predict(cv_fit1, newdata = testTransformed)
pred2 <- predict(cv_fit2, newdata = testTransformed)
pred3 <- predict(cv_fit3, newdata = testTransformed)
cv_rmse1 <- postResample(pred1, obs = testTransformed$log.shares)
cv_rmse2 <- postResample(pred2, obs = testTransformed$log.shares)
cv_rmse3 <- postResample(pred3, obs = testTransformed$log.shares)
result2 <- rbind(cv_rmse1, cv_rmse2, cv_rmse3)
row.names(result2) <- c("Model 1","Model 2", "Model 3")
kable(result2, digits = 4, caption = "Table ###. Cross Validation - Model Predictions on Test Set")
```

|         |   RMSE | Rsquared |    MAE |
|:--------|-------:|---------:|-------:|
| Model 1 | 0.9311 |   0.1258 | 0.6902 |
| Model 2 | 0.9304 |   0.1269 | 0.6891 |
| Model 3 | 0.9291 |   0.1291 | 0.6874 |

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

    ## # A tibble: 4,382 x 17
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
    ## # ... with 4,372 more rows, and 8 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
    ## #   n_non_stop_unique_tokens <dbl>, average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>,
    ## #   num_imgs <dbl>

``` r
preProcValues <- preProcess(train2, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train2)
testTransformed <- predict(preProcValues, test2)

random_forest <- train(log.shares ~ ., data = trainTransformed,
    method = "rf",
    trControl = trainControl(method = "cv", number = 10),
    tuneGrid = data.frame(mtry = 1:15))

random_forest_predict <- predict(random_forest, newdata = testTransformed)
rf_rmse <- postResample(random_forest_predict, obs = testTransformed$log.shares)
```

## Boosted Tree

short but reasonably thorough explanation of the ensemble model you are
using

| Predictors                   | Attribute Information                             | Type    |
|------------------------------|---------------------------------------------------|---------|
| `kw_avg_avg`                 | Average keyword (average shares)                  | number  |
| `LDA_02`                     | Closeness to LDA topic 2                          | ratio   |
| `weekday_is_monday`          | Was the article published on a Monday?            | boolean |
| `weekday_is_tuesday`         | Was the article published on a Tuesday?           | boolean |
| `weekday_is_wednesday`       | Was the article published on a Wednesday?         | boolean |
| `weekday_is_thursday`        | Was the article published on a Thursday?          | boolean |
| `weekday_is_friday`          | Was the article published on a Friday?            | boolean |
| `weekday_is_saturday`        | Was the article published on a Saturday?          | boolean |
| `weekday_is_sunday`          | Was the article published on a Sunday?            | boolean |
| `self_reference_avg_sharess` | Average shares of referenced articles in Mashable | number  |
| `n_non_stop_unique_tokens`   | Rate of unique non-stop words in the content      | ratio   |
| `average_token_length`       | Average length of the words in the content        | number  |
| `n_tokens_content`           | Number of words in the content                    | number  |
| `n_tokens_title`             | Number of words in the title                      | number  |
| `global_subjectivity`        | Text subjectivity                                 | ratio   |
| `num_imgs`                   | Number of images                                  | number  |

``` r
#expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10)
boosted_tree <- train(log.shares ~ . , data = trainTransformed,
      method = "gbm", 
      trControl = trainControl(method = "cv", number = 10), #method="repeatedcv", repeats=5
      tuneGrid = expand.grid(n.trees = c(25, 50, 75, 100), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 4382 samples
    ##   16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 3944, 3943, 3944, 3944, 3944, 3945, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared   MAE      
    ##   1                   25      0.9448338  0.1222602  0.6856697
    ##   1                   50      0.9318327  0.1379367  0.6733338
    ##   1                   75      0.9266331  0.1433695  0.6682175
    ##   1                  100      0.9248999  0.1457956  0.6659584
    ##   2                   25      0.9344493  0.1367285  0.6752057
    ##   2                   50      0.9258026  0.1447875  0.6665741
    ##   2                   75      0.9231369  0.1489332  0.6628012
    ##   2                  100      0.9224968  0.1502069  0.6612477
    ##   3                   25      0.9327844  0.1350509  0.6722491
    ##   3                   50      0.9264823  0.1434581  0.6664957
    ##   3                   75      0.9261829  0.1439738  0.6654035
    ##   3                  100      0.9282587  0.1419245  0.6660285
    ##   4                   25      0.9298638  0.1405720  0.6695194
    ##   4                   50      0.9233931  0.1489078  0.6626812
    ##   4                   75      0.9247785  0.1477202  0.6606953
    ##   4                  100      0.9266786  0.1455600  0.6617817
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse3, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Random Forest Model", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Model 1      | 0.9311 |   0.1258 | 0.6902 |
| Linear Model 2      | 0.9291 |   0.1291 | 0.6874 |
| Random Forest Model | 0.8900 |   0.2034 | 0.6539 |
| Boosted Model       | 0.8973 |   0.1884 | 0.6598 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the bus channel
is “need to automate this part”.

The best model fit to predict the number of shares

# Automation

Automation is done with the modifications of the YAML header and the
render function.
