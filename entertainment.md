ST558 - Project 2 - Predictive Modeling
================
Jasmine Wang & Ilana Feldman
10/31/2021

-   [Introduction - done by Jasmine](#introduction---done-by-jasmine)
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

# Introduction - done by Jasmine

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

This analysis is based on the entertainment channel popularity.

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

    ## [1] 7057   53

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
sel_data
```

    ## # A tibble: 7,057 x 24
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesd~ weekday_is_wedn~
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>             <dbl>            <dbl>
    ##  1            0    593       6.39       1         0  0.500  0.378  0.0400 0.0413 0.0401                 1                 0                0
    ##  2            0   1200       7.09       1         0  0.0286 0.419  0.495  0.0289 0.0286                 1                 0                0
    ##  3            1   2100       7.65       1         0  0.0334 0.0345 0.215  0.684  0.0333                 1                 0                0
    ##  4            0   1200       7.09       1         0  0.126  0.0203 0.0200 0.814  0.0200                 1                 0                0
    ##  5            1   4600       8.43       1         0  0.200  0.340  0.0333 0.393  0.0333                 1                 0                0
    ##  6            0   1200       7.09       1         0  0.0240 0.665  0.0225 0.266  0.0223                 1                 0                0
    ##  7            0    631       6.45       1         0  0.456  0.482  0.0200 0.0213 0.0200                 1                 0                0
    ##  8            0   1300       7.17       2      1114. 0.0500 0.525  0.324  0.0510 0.0500                 0                 1                0
    ##  9            1   1700       7.44       2       714. 0.0400 0.840  0.0400 0.0401 0.0400                 0                 1                0
    ## 10            0    455       6.12       3       707. 0.0334 0.409  0.0333 0.491  0.0333                 0                 0                1
    ## # ... with 7,047 more rows, and 11 more variables: weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
    ## #   weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, average_token_length <dbl>,
    ## #   n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>

``` r
set.seed(388588)
sharesIndex <- createDataPartition(sel_data$shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]

train1 <- train %>% select(-class_shares, -shares, 
                           -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                           -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04) #keep log.shares
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04) #keep log.shares
```

# Exploratory Data Analysis

Now let us take a look at the relationships between our response and the
predictors with some numerical summaries and plots.

## Numerical Summaries

Table 1 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from entertainment channel falls
into different categories on different days of the week for 709 days.

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

Table 3 shows the numerical summaries of average keywords from
entertainment channel in mashable.com on different days of the week.
Table 4 shows the numerical summaries of average shares of referenced
articles in mashable.com on different days of the week.

``` r
# contingency table
edadata <- train
edadata$class.shares <- cut(edadata$class_shares, 2, c("Unpopular","Popular"))
edadata$day.week <- cut(edadata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
table(edadata$class.shares, edadata$day.week) %>% kable(caption = "Table 1. Popularity on Day of the Week")
```

|           | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|:----------|-------:|--------:|----------:|---------:|-------:|---------:|-------:|
| Unpopular |    614 |     566 |       576 |      536 |    392 |       97 |    134 |
| Popular   |    355 |     345 |       325 |      323 |    269 |      172 |    237 |

Table 1. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 2. Average Shares vs. Average Log(shares) on Day of the Week")
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

Table 2. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 3. Summary of Average Keywords on Day of the Week")
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

Table 3. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average shares of referenced articles in Mashable on Day of the Week")
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
correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt")
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n", 
         cex = 0.8,
         title="Figure 1. Correlations Between the Variables")
```

![](../images/unnamed-chunk-3-1.png)<!-- -->

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

![](../images/unnamed-chunk-4-1.png)<!-- -->

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the entertainment channel on mashable.com on any day of the week. The
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
b.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0 LDA_1  LDA_2 LDA_3  LDA_4
    ##    <fct>     <fct>         <dbl> <dbl>  <dbl> <dbl>  <dbl>
    ##  1 Monday    Unpopular    0.0648 0.433 0.0890 0.348 0.0649
    ##  2 Monday    Popular      0.0603 0.437 0.0757 0.366 0.0610
    ##  3 Tuesday   Unpopular    0.0642 0.449 0.0922 0.333 0.0619
    ##  4 Tuesday   Popular      0.0706 0.400 0.0695 0.389 0.0681
    ##  5 Wednesday Unpopular    0.0636 0.431 0.0900 0.349 0.0660
    ##  6 Wednesday Popular      0.0696 0.428 0.0846 0.352 0.0659
    ##  7 Thursday  Unpopular    0.0713 0.450 0.0944 0.318 0.0659
    ##  8 Thursday  Popular      0.0609 0.419 0.0863 0.374 0.0592
    ##  9 Friday    Unpopular    0.0605 0.441 0.0990 0.340 0.0599
    ## 10 Friday    Popular      0.0690 0.400 0.0861 0.381 0.0646
    ## 11 Saturday  Unpopular    0.0733 0.367 0.154  0.354 0.0518
    ## 12 Saturday  Popular      0.0650 0.425 0.0919 0.347 0.0713
    ## 13 Sunday    Unpopular    0.0662 0.331 0.0919 0.440 0.0708
    ## 14 Sunday    Popular      0.0847 0.351 0.0918 0.413 0.0592

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.0648
    ##  2 Monday   Unpopular    LDA_1      0.433 
    ##  3 Monday   Unpopular    LDA_2      0.0890
    ##  4 Monday   Unpopular    LDA_3      0.348 
    ##  5 Monday   Unpopular    LDA_4      0.0649
    ##  6 Monday   Popular      LDA_0      0.0603
    ##  7 Monday   Popular      LDA_1      0.437 
    ##  8 Monday   Popular      LDA_2      0.0757
    ##  9 Monday   Popular      LDA_3      0.366 
    ## 10 Monday   Popular      LDA_4      0.0610
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

![](../images/unnamed-chunk-5-1.png)<!-- -->

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the entertainment channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0 LDA_1  LDA_2 LDA_3  LDA_4
    ##    <fct>     <fct>         <dbl> <dbl>  <dbl> <dbl>  <dbl>
    ##  1 Monday    Unpopular    0.0648 0.433 0.0890 0.348 0.0649
    ##  2 Monday    Popular      0.0603 0.437 0.0757 0.366 0.0610
    ##  3 Tuesday   Unpopular    0.0642 0.449 0.0922 0.333 0.0619
    ##  4 Tuesday   Popular      0.0706 0.400 0.0695 0.389 0.0681
    ##  5 Wednesday Unpopular    0.0636 0.431 0.0900 0.349 0.0660
    ##  6 Wednesday Popular      0.0696 0.428 0.0846 0.352 0.0659
    ##  7 Thursday  Unpopular    0.0713 0.450 0.0944 0.318 0.0659
    ##  8 Thursday  Popular      0.0609 0.419 0.0863 0.374 0.0592
    ##  9 Friday    Unpopular    0.0605 0.441 0.0990 0.340 0.0599
    ## 10 Friday    Popular      0.0690 0.400 0.0861 0.381 0.0646
    ## 11 Saturday  Unpopular    0.0733 0.367 0.154  0.354 0.0518
    ## 12 Saturday  Popular      0.0650 0.425 0.0919 0.347 0.0713
    ## 13 Sunday    Unpopular    0.0662 0.331 0.0919 0.440 0.0708
    ## 14 Sunday    Popular      0.0847 0.351 0.0918 0.413 0.0592

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.0648
    ##  2 Monday   Unpopular    LDA_1      0.433 
    ##  3 Monday   Unpopular    LDA_2      0.0890
    ##  4 Monday   Unpopular    LDA_3      0.348 
    ##  5 Monday   Unpopular    LDA_4      0.0649
    ##  6 Monday   Popular      LDA_0      0.0603
    ##  7 Monday   Popular      LDA_1      0.437 
    ##  8 Monday   Popular      LDA_2      0.0757
    ##  9 Monday   Popular      LDA_3      0.366 
    ## 10 Monday   Popular      LDA_4      0.0610
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

![](../images/unnamed-chunk-6-1.png)<!-- -->

### Scatterplot

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the entertainment
channel across different days of the week. In the news popularity study,
it says average keyword is the most important predictor in the models
they used which accounted for the most variation in the data. Therefore,
we are interested to see how average keyword is correlated with log
shares. The different colored linear regression lines indicate different
days of the week. If it is an upward trend, it shows positive linear
relationship. If it is a downward trend, it shows a negative linear
relationship. More tilted the line is, much stronger the relationship is
regardless of positive or negative.

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

![](../images/unnamed-chunk-7-1.png)<!-- -->

# Modeling

## Linear Regression

a short but thorough explanation of the idea of a linear regression
model.

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
    ## -3.9193 -0.6003 -0.2370  0.3643  4.4995 
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                -0.04436    0.03090  -1.436  0.15111    
    ## dayweek2                   -0.03221    0.04440  -0.726  0.46817    
    ## dayweek3                   -0.04062    0.04453  -0.912  0.36171    
    ## dayweek4                   -0.01017    0.04508  -0.226  0.82151    
    ## dayweek5                    0.04336    0.04852   0.894  0.37148    
    ## dayweek6                    0.33557    0.06641   5.053 4.50e-07 ***
    ## dayweek7                    0.47154    0.05877   8.023 1.28e-15 ***
    ## kw_avg_avg                  0.18056    0.01415  12.756  < 2e-16 ***
    ## LDA_02                     -0.03261    0.01395  -2.338  0.01945 *  
    ## self_reference_avg_sharess  0.09005    0.01394   6.462 1.13e-10 ***
    ## n_non_stop_unique_tokens    0.02308    0.01374   1.680  0.09306 .  
    ## average_token_length       -0.03115    0.01840  -1.693  0.09043 .  
    ## n_tokens_content           -0.01980    0.01568  -1.263  0.20664    
    ## n_tokens_title             -0.00467    0.01375  -0.339  0.73426    
    ## global_subjectivity         0.05757    0.01835   3.137  0.00172 ** 
    ## num_imgs                    0.04848    0.01547   3.134  0.00174 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9614 on 4925 degrees of freedom
    ## Multiple R-squared:  0.07857,    Adjusted R-squared:  0.07576 
    ## F-statistic: 27.99 on 15 and 4925 DF,  p-value: < 2.2e-16

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
    ## -3.9681 -0.5985 -0.2383  0.3704  4.5117 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.0629559  0.0320851  -1.962 0.049801 *  
    ## dayweek2                                   -0.0330453  0.0443006  -0.746 0.455744    
    ## dayweek3                                   -0.0410796  0.0444284  -0.925 0.355207    
    ## dayweek4                                   -0.0124044  0.0449895  -0.276 0.782777    
    ## dayweek5                                    0.0408271  0.0484515   0.843 0.399472    
    ## dayweek6                                    0.3405765  0.0662854   5.138 2.88e-07 ***
    ## dayweek7                                    0.4752498  0.0586909   8.098 7.01e-16 ***
    ## kw_avg_avg                                  0.1784387  0.0141973  12.569  < 2e-16 ***
    ## LDA_02                                     -0.0334613  0.0139389  -2.401 0.016406 *  
    ## self_reference_avg_sharess                  0.1213338  0.0316334   3.836 0.000127 ***
    ## n_non_stop_unique_tokens                    0.0248505  0.0137240   1.811 0.070243 .  
    ## average_token_length                        0.0186052  0.0438872   0.424 0.671634    
    ## n_tokens_content                           -0.0287057  0.0209177  -1.372 0.170028    
    ## n_tokens_title                             -0.0019196  0.0137736  -0.139 0.889166    
    ## global_subjectivity                         0.0630246  0.0188140   3.350 0.000815 ***
    ## num_imgs                                    0.0382324  0.0158350   2.414 0.015796 *  
    ## `I(n_tokens_content^2)`                     0.0065954  0.0054429   1.212 0.225670    
    ## `kw_avg_avg:num_imgs`                       0.0497198  0.0162574   3.058 0.002238 ** 
    ## `average_token_length:global_subjectivity`  0.0154639  0.0123334   1.254 0.209963    
    ## `dayweek2:self_reference_avg_sharess`       0.0007804  0.0489428   0.016 0.987279    
    ## `dayweek3:self_reference_avg_sharess`       0.0146644  0.0470760   0.312 0.755430    
    ## `dayweek4:self_reference_avg_sharess`       0.0276575  0.0452784   0.611 0.541339    
    ## `dayweek5:self_reference_avg_sharess`      -0.0446623  0.0563137  -0.793 0.427759    
    ## `dayweek6:self_reference_avg_sharess`      -0.1365807  0.0455444  -2.999 0.002724 ** 
    ## `dayweek7:self_reference_avg_sharess`      -0.1139098  0.0553453  -2.058 0.039627 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.959 on 4916 degrees of freedom
    ## Multiple R-squared:  0.08472,    Adjusted R-squared:  0.08025 
    ## F-statistic: 18.96 on 24 and 4916 DF,  p-value: < 2.2e-16

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
    ## -3.9134 -0.5940 -0.2396  0.3708  4.5122 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.056348   0.032104  -1.755  0.07930 .  
    ## dayweek2                                   -0.035456   0.044287  -0.801  0.42340    
    ## dayweek3                                   -0.042243   0.044418  -0.951  0.34163    
    ## dayweek4                                   -0.011961   0.044978  -0.266  0.79030    
    ## dayweek5                                    0.042173   0.048395   0.871  0.38357    
    ## dayweek6                                    0.350621   0.066343   5.285 1.31e-07 ***
    ## dayweek7                                    0.473919   0.058668   8.078 8.21e-16 ***
    ## kw_avg_avg                                  0.175351   0.014253  12.303  < 2e-16 ***
    ## LDA_02                                     -0.031601   0.013931  -2.268  0.02335 *  
    ## self_reference_avg_sharess                  0.157060   0.021133   7.432 1.25e-13 ***
    ## n_non_stop_unique_tokens                    0.024666   0.013720   1.798  0.07227 .  
    ## average_token_length                        0.019482   0.043855   0.444  0.65689    
    ## n_tokens_content                           -0.028416   0.020891  -1.360  0.17382    
    ## n_tokens_title                             -0.002401   0.013762  -0.174  0.86153    
    ## global_subjectivity                         0.061796   0.018798   3.287  0.00102 ** 
    ## num_imgs                                    0.037405   0.015830   2.363  0.01817 *  
    ## `I(n_tokens_content^2)`                     0.006647   0.005439   1.222  0.22169    
    ## `I(self_reference_avg_sharess^2)`          -0.007507   0.001779  -4.221 2.48e-05 ***
    ## `kw_avg_avg:num_imgs`                       0.048536   0.016243   2.988  0.00282 ** 
    ## `average_token_length:global_subjectivity`  0.016431   0.012335   1.332  0.18291    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9588 on 4921 degrees of freedom
    ## Multiple R-squared:  0.08425,    Adjusted R-squared:  0.08071 
    ## F-statistic: 23.83 on 19 and 4921 DF,  p-value: < 2.2e-16

``` r
result_tab <- data.frame(t(cv_fit1$results),t(cv_fit2$results), t(cv_fit3$results))
colnames(result_tab) <- c("Model 1","Model 2", "Model 3")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 | Model 3 |
|:-----------|--------:|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |  1.0000 |
| RMSE       |  1.3792 |  1.8517 |  1.7959 |
| Rsquared   |  0.0695 |  0.0710 |  0.0645 |
| MAE        |  0.7322 |  0.7526 |  0.7494 |
| RMSESD     |  1.3456 |  2.8218 |  2.6527 |
| RsquaredSD |  0.0398 |  0.0349 |  0.0340 |
| MAESD      |  0.0902 |  0.1450 |  0.1445 |

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
| Model 1 | 0.9813 |   0.0497 | 0.7037 |
| Model 2 | 0.9758 |   0.0584 | 0.7012 |
| Model 3 | 0.9732 |   0.0609 | 0.6995 |

Table \#\#\#. Cross Validation - Model Predictions on Test Set

## Random Forest

Ilana

short but reasonably thorough explanation of the ensemble model you are
using

Categorical variables have to be divided their levels into dummy
variables style.

``` r
train2 <- train %>% select(-class_shares, -shares, -dayweek, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
test2 <- test %>% select(-class_shares, -shares, -dayweek, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
train2
```

    ## # A tibble: 4,941 x 17
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
    ## # ... with 4,931 more rows, and 8 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
    ## #   n_non_stop_unique_tokens <dbl>, average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>,
    ## #   num_imgs <dbl>

``` r
preProcValues <- preProcess(train2, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train2)
testTransformed <- predict(preProcValues, test2)
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
    ## 4941 samples
    ##   16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4447, 4449, 4447, 4446, 4446, 4447, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9658893  0.07464788  0.7140226
    ##   1                   50      0.9595639  0.08116757  0.7073590
    ##   1                   75      0.9577154  0.08280655  0.7056202
    ##   1                  100      0.9577698  0.08247236  0.7054139
    ##   2                   25      0.9605362  0.08035490  0.7084321
    ##   2                   50      0.9566896  0.08472053  0.7041160
    ##   2                   75      0.9574753  0.08290929  0.7042801
    ##   2                  100      0.9569497  0.08388002  0.7047055
    ##   3                   25      0.9609741  0.07730143  0.7087700
    ##   3                   50      0.9577675  0.08205197  0.7055517
    ##   3                   75      0.9584468  0.08146307  0.7055401
    ##   3                  100      0.9594165  0.08051896  0.7063919
    ##   4                   25      0.9589043  0.08171526  0.7063486
    ##   4                   50      0.9584947  0.08127019  0.7052882
    ##   4                   75      0.9596207  0.08065970  0.7052690
    ##   4                  100      0.9612653  0.07868383  0.7065377
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse3, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                |   RMSE | Rsquared |    MAE |
|:---------------|-------:|---------:|-------:|
| Linear Model 1 | 0.9813 |   0.0497 | 0.7037 |
| Linear Model 2 | 0.9732 |   0.0609 | 0.6995 |
| Boosted Model  | 0.9634 |   0.0710 | 0.6970 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the entertainment
channel is “need to automate this part”.

The best model fit to predict the number of shares

# Automation

Automation is done with the modifications of the YAML header and the
render function.
