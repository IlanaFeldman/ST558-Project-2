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

This analysis is based on the tech channel popularity.

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

    ## [1] 7346   53

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

    ## # A tibble: 7,346 x 24
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesd~ weekday_is_wedn~
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>             <dbl>            <dbl>
    ##  1            0    505       6.22       1          0 0.0286 0.0288 0.0286 0.0286  0.885                 1                 0                0
    ##  2            0    855       6.75       1          0 0.0222 0.307  0.0222 0.0222  0.627                 1                 0                0
    ##  3            0    891       6.79       1          0 0.0222 0.151  0.243  0.0222  0.561                 1                 0                0
    ##  4            1   3600       8.19       1          0 0.458  0.0290 0.0287 0.0297  0.454                 1                 0                0
    ##  5            1  17100       9.75       1          0 0.0250 0.0252 0.0250 0.0250  0.900                 1                 0                0
    ##  6            1   2800       7.94       1          0 0.0201 0.0200 0.0200 0.0200  0.920                 1                 0                0
    ##  7            0    445       6.10       1          0 0.312  0.233  0.0286 0.0286  0.398                 1                 0                0
    ##  8            0    783       6.66       1          0 0.0200 0.367  0.0202 0.0200  0.572                 1                 0                0
    ##  9            1   1500       7.31       1          0 0.0222 0.0226 0.133  0.133   0.689                 1                 0                0
    ## 10            1   1800       7.50       1          0 0.0223 0.362  0.0222 0.0223  0.571                 1                 0                0
    ## # ... with 7,336 more rows, and 11 more variables: weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
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
We can see the total number of articles from tech channel falls into
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

Table 3 shows the numerical summaries of average keywords from tech
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
| Unpopular |    326 |     413 |       430 |      362 |    221 |       59 |     51 |
| Popular   |    503 |     614 |       606 |      535 |    472 |      323 |    230 |

Table 1. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 2. Average Shares vs. Average Log(shares) on Day of the Week")
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

Table 2. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 3. Summary of Average Keywords on Day of the Week")
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

Table 3. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average shares of referenced articles in Mashable on Day of the Week")
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
b.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0  LDA_1  LDA_2  LDA_3 LDA_4
    ##    <fct>     <fct>         <dbl>  <dbl>  <dbl>  <dbl> <dbl>
    ##  1 Monday    Unpopular    0.0695 0.0843 0.118  0.0510 0.677
    ##  2 Monday    Popular      0.0754 0.0569 0.112  0.0610 0.695
    ##  3 Tuesday   Unpopular    0.0776 0.0668 0.124  0.0570 0.674
    ##  4 Tuesday   Popular      0.0728 0.0634 0.0984 0.0674 0.698
    ##  5 Wednesday Unpopular    0.0715 0.0718 0.110  0.0535 0.693
    ##  6 Wednesday Popular      0.0787 0.0590 0.112  0.0666 0.683
    ##  7 Thursday  Unpopular    0.0760 0.0763 0.106  0.0650 0.676
    ##  8 Thursday  Popular      0.0650 0.0639 0.112  0.0595 0.700
    ##  9 Friday    Unpopular    0.0622 0.0682 0.130  0.0705 0.669
    ## 10 Friday    Popular      0.0728 0.0681 0.0987 0.0642 0.696
    ## 11 Saturday  Unpopular    0.0815 0.0613 0.106  0.0482 0.703
    ## 12 Saturday  Popular      0.0835 0.0512 0.109  0.0600 0.696
    ## 13 Sunday    Unpopular    0.0877 0.0579 0.108  0.0529 0.694
    ## 14 Sunday    Popular      0.0818 0.0571 0.108  0.0710 0.682

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.0695
    ##  2 Monday   Unpopular    LDA_1      0.0843
    ##  3 Monday   Unpopular    LDA_2      0.118 
    ##  4 Monday   Unpopular    LDA_3      0.0510
    ##  5 Monday   Unpopular    LDA_4      0.677 
    ##  6 Monday   Popular      LDA_0      0.0754
    ##  7 Monday   Popular      LDA_1      0.0569
    ##  8 Monday   Popular      LDA_2      0.112 
    ##  9 Monday   Popular      LDA_3      0.0610
    ## 10 Monday   Popular      LDA_4      0.695 
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
and popularity groups for articles in the tech channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0  LDA_1  LDA_2  LDA_3 LDA_4
    ##    <fct>     <fct>         <dbl>  <dbl>  <dbl>  <dbl> <dbl>
    ##  1 Monday    Unpopular    0.0695 0.0843 0.118  0.0510 0.677
    ##  2 Monday    Popular      0.0754 0.0569 0.112  0.0610 0.695
    ##  3 Tuesday   Unpopular    0.0776 0.0668 0.124  0.0570 0.674
    ##  4 Tuesday   Popular      0.0728 0.0634 0.0984 0.0674 0.698
    ##  5 Wednesday Unpopular    0.0715 0.0718 0.110  0.0535 0.693
    ##  6 Wednesday Popular      0.0787 0.0590 0.112  0.0666 0.683
    ##  7 Thursday  Unpopular    0.0760 0.0763 0.106  0.0650 0.676
    ##  8 Thursday  Popular      0.0650 0.0639 0.112  0.0595 0.700
    ##  9 Friday    Unpopular    0.0622 0.0682 0.130  0.0705 0.669
    ## 10 Friday    Popular      0.0728 0.0681 0.0987 0.0642 0.696
    ## 11 Saturday  Unpopular    0.0815 0.0613 0.106  0.0482 0.703
    ## 12 Saturday  Popular      0.0835 0.0512 0.109  0.0600 0.696
    ## 13 Sunday    Unpopular    0.0877 0.0579 0.108  0.0529 0.694
    ## 14 Sunday    Popular      0.0818 0.0571 0.108  0.0710 0.682

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.0695
    ##  2 Monday   Unpopular    LDA_1      0.0843
    ##  3 Monday   Unpopular    LDA_2      0.118 
    ##  4 Monday   Unpopular    LDA_3      0.0510
    ##  5 Monday   Unpopular    LDA_4      0.677 
    ##  6 Monday   Popular      LDA_0      0.0754
    ##  7 Monday   Popular      LDA_1      0.0569
    ##  8 Monday   Popular      LDA_2      0.112 
    ##  9 Monday   Popular      LDA_3      0.0610
    ## 10 Monday   Popular      LDA_4      0.695 
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
    ## -4.7669 -0.6387 -0.1788  0.4847  6.6464 
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                -0.02219    0.03382  -0.656 0.511717    
    ## dayweek2                   -0.05257    0.04540  -1.158 0.246933    
    ## dayweek3                   -0.02581    0.04532  -0.569 0.569109    
    ## dayweek4                   -0.08529    0.04690  -1.819 0.069045 .  
    ## dayweek5                    0.06413    0.05010   1.280 0.200633    
    ## dayweek6                    0.33700    0.06023   5.595 2.32e-08 ***
    ## dayweek7                    0.34963    0.06723   5.201 2.06e-07 ***
    ## kw_avg_avg                  0.15338    0.01365  11.238  < 2e-16 ***
    ## LDA_02                     -0.00808    0.01387  -0.582 0.560314    
    ## self_reference_avg_sharess  0.04584    0.01358   3.374 0.000745 ***
    ## n_non_stop_unique_tokens   -0.01676    0.01985  -0.844 0.398478    
    ## average_token_length       -0.02376    0.01477  -1.609 0.107607    
    ## n_tokens_content            0.10696    0.01846   5.794 7.27e-09 ***
    ## n_tokens_title             -0.03764    0.01364  -2.759 0.005815 ** 
    ## global_subjectivity         0.01066    0.01391   0.767 0.443340    
    ## num_imgs                   -0.01260    0.01701  -0.741 0.458790    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.97 on 5129 degrees of freedom
    ## Multiple R-squared:  0.0618, Adjusted R-squared:  0.05906 
    ## F-statistic: 22.52 on 15 and 5129 DF,  p-value: < 2.2e-16

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
    ## -4.7588 -0.6351 -0.1723  0.4799  6.7143 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0087609  0.0344967   0.254  0.79953    
    ## dayweek2                                   -0.0678003  0.0454013  -1.493  0.13541    
    ## dayweek3                                   -0.0440251  0.0453262  -0.971  0.33145    
    ## dayweek4                                   -0.0967833  0.0468758  -2.065  0.03900 *  
    ## dayweek5                                    0.0548575  0.0500587   1.096  0.27319    
    ## dayweek6                                    0.3129173  0.0602039   5.198  2.1e-07 ***
    ## dayweek7                                    0.4144082  0.0727697   5.695  1.3e-08 ***
    ## kw_avg_avg                                  0.1488658  0.0136570  10.900  < 2e-16 ***
    ## LDA_02                                     -0.0142601  0.0139288  -1.024  0.30598    
    ## self_reference_avg_sharess                  0.2530984  0.0809844   3.125  0.00179 ** 
    ## n_non_stop_unique_tokens                    0.0139753  0.0217398   0.643  0.52035    
    ## average_token_length                       -0.0021880  0.0183241  -0.119  0.90496    
    ## n_tokens_content                            0.1674222  0.0280928   5.960  2.7e-09 ***
    ## n_tokens_title                             -0.0315454  0.0136504  -2.311  0.02088 *  
    ## global_subjectivity                         0.0141977  0.0141542   1.003  0.31587    
    ## num_imgs                                    0.0001904  0.0173718   0.011  0.99126    
    ## `I(n_tokens_content^2)`                    -0.0148072  0.0060460  -2.449  0.01436 *  
    ## `kw_avg_avg:num_imgs`                       0.0591123  0.0140411   4.210  2.6e-05 ***
    ## `average_token_length:global_subjectivity`  0.0112103  0.0047905   2.340  0.01932 *  
    ## `dayweek2:self_reference_avg_sharess`      -0.2078529  0.0852562  -2.438  0.01480 *  
    ## `dayweek3:self_reference_avg_sharess`      -0.1636885  0.0875752  -1.869  0.06166 .  
    ## `dayweek4:self_reference_avg_sharess`      -0.2310449  0.0857194  -2.695  0.00705 ** 
    ## `dayweek5:self_reference_avg_sharess`      -0.2327730  0.0846493  -2.750  0.00598 ** 
    ## `dayweek6:self_reference_avg_sharess`      -0.2169843  0.1126931  -1.925  0.05423 .  
    ## `dayweek7:self_reference_avg_sharess`       0.8304727  0.3526353   2.355  0.01856 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9665 on 5120 degrees of freedom
    ## Multiple R-squared:  0.0703, Adjusted R-squared:  0.06594 
    ## F-statistic: 16.13 on 24 and 5120 DF,  p-value: < 2.2e-16

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
    ## -4.7299 -0.6306 -0.1725  0.4819  6.6751 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0115712  0.0343149   0.337   0.7360    
    ## dayweek2                                   -0.0525586  0.0452054  -1.163   0.2450    
    ## dayweek3                                   -0.0351322  0.0451088  -0.779   0.4361    
    ## dayweek4                                   -0.0888279  0.0466635  -1.904   0.0570 .  
    ## dayweek5                                    0.0670387  0.0498468   1.345   0.1787    
    ## dayweek6                                    0.3185596  0.0599814   5.311 1.14e-07 ***
    ## dayweek7                                    0.3445472  0.0668818   5.152 2.68e-07 ***
    ## kw_avg_avg                                  0.1429396  0.0136707  10.456  < 2e-16 ***
    ## LDA_02                                     -0.0178045  0.0139022  -1.281   0.2004    
    ## self_reference_avg_sharess                  0.2580130  0.0370177   6.970 3.57e-12 ***
    ## n_non_stop_unique_tokens                    0.0134393  0.0216775   0.620   0.5353    
    ## average_token_length                       -0.0005375  0.0182810  -0.029   0.9765    
    ## n_tokens_content                            0.1705022  0.0280298   6.083 1.27e-09 ***
    ## n_tokens_title                             -0.0318396  0.0136137  -2.339   0.0194 *  
    ## global_subjectivity                         0.0151301  0.0141175   1.072   0.2839    
    ## num_imgs                                   -0.0006839  0.0173285  -0.039   0.9685    
    ## `I(n_tokens_content^2)`                    -0.0151777  0.0060316  -2.516   0.0119 *  
    ## `I(self_reference_avg_sharess^2)`          -0.0133335  0.0021747  -6.131 9.38e-10 ***
    ## `kw_avg_avg:num_imgs`                       0.0607535  0.0140077   4.337 1.47e-05 ***
    ## `average_token_length:global_subjectivity`  0.0119854  0.0047800   2.507   0.0122 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9643 on 5125 degrees of freedom
    ## Multiple R-squared:  0.07361,    Adjusted R-squared:  0.07017 
    ## F-statistic: 21.43 on 19 and 5125 DF,  p-value: < 2.2e-16

``` r
result_tab <- data.frame(t(cv_fit1$results),t(cv_fit2$results), t(cv_fit3$results))
colnames(result_tab) <- c("Model 1","Model 2", "Model 3")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 | Model 3 |
|:-----------|--------:|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |  1.0000 |
| RMSE       |  0.9714 |  0.9730 |  0.9672 |
| Rsquared   |  0.0580 |  0.0617 |  0.0660 |
| MAE        |  0.7370 |  0.7376 |  0.7340 |
| RMSESD     |  0.0287 |  0.0537 |  0.0297 |
| RsquaredSD |  0.0150 |  0.0218 |  0.0194 |
| MAESD      |  0.0140 |  0.0268 |  0.0101 |

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
| Model 1 | 0.9273 |   0.0481 | 0.7202 |
| Model 2 | 1.0367 |   0.0147 | 0.7315 |
| Model 3 | 0.9325 |   0.0441 | 0.7222 |

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

    ## # A tibble: 5,145 x 17
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
    ## # ... with 5,135 more rows, and 8 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
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
    ## 5145 samples
    ##   16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4631, 4631, 4630, 4630, 4631, 4630, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9674977  0.07597769  0.7387925
    ##   1                   50      0.9594363  0.08429885  0.7311373
    ##   1                   75      0.9576172  0.08485910  0.7289083
    ##   1                  100      0.9569137  0.08548935  0.7285715
    ##   2                   25      0.9604576  0.08568509  0.7322726
    ##   2                   50      0.9545656  0.09119302  0.7274996
    ##   2                   75      0.9530990  0.09281867  0.7257595
    ##   2                  100      0.9537185  0.09137827  0.7257389
    ##   3                   25      0.9576053  0.08825503  0.7308629
    ##   3                   50      0.9534208  0.09256431  0.7267907
    ##   3                   75      0.9515291  0.09576184  0.7241228
    ##   3                  100      0.9512199  0.09661809  0.7240487
    ##   4                   25      0.9564538  0.08914914  0.7290427
    ##   4                   50      0.9533939  0.09215119  0.7262581
    ##   4                   75      0.9512342  0.09628521  0.7248452
    ##   4                  100      0.9529065  0.09389567  0.7251516
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse3, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                |   RMSE | Rsquared |    MAE |
|:---------------|-------:|---------:|-------:|
| Linear Model 1 | 0.9273 |   0.0481 | 0.7202 |
| Linear Model 2 | 0.9325 |   0.0441 | 0.7222 |
| Boosted Model  | 0.9152 |   0.0759 | 0.7103 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the tech channel
is “need to automate this part”.

The best model fit to predict the number of shares

# Automation

Automation is done with the modifications of the YAML header and the
render function.
