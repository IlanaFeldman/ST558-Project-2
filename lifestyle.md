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

This analysis is based on the lifestyle channel popularity.

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

    ## [1] 2099   53

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

    ## # A tibble: 2,099 x 24
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesd~ weekday_is_wedn~
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>             <dbl>            <dbl>
    ##  1            0    556       6.32       1         0  0.0201 0.115  0.0200 0.0200  0.825                 1                 0                0
    ##  2            1   1900       7.55       1         0  0.0286 0.0286 0.0286 0.0287  0.885                 1                 0                0
    ##  3            1   5700       8.65       1         0  0.437  0.200  0.0335 0.0334  0.295                 1                 0                0
    ##  4            0    462       6.14       1         0  0.0200 0.0200 0.0200 0.0200  0.920                 1                 0                0
    ##  5            1   3600       8.19       1         0  0.211  0.0255 0.0251 0.0251  0.713                 1                 0                0
    ##  6            0    343       5.84       1         0  0.0201 0.0206 0.0205 0.121   0.818                 1                 0                0
    ##  7            0    507       6.23       1         0  0.0250 0.160  0.0250 0.0250  0.765                 1                 0                0
    ##  8            0    552       6.31       1         0  0.207  0.146  0.276  0.0251  0.346                 1                 0                0
    ##  9            0   1200       7.09       2       885. 0.0202 0.133  0.120  0.0201  0.707                 0                 1                0
    ## 10            1   1900       7.55       3      1207. 0.0335 0.217  0.0334 0.0335  0.683                 0                 0                1
    ## # ... with 2,089 more rows, and 11 more variables: weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
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
We can see the total number of articles from lifestyle channel falls
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

Table 3 shows the numerical summaries of average keywords from lifestyle
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
| Unpopular |     91 |     107 |       121 |       99 |     91 |       19 |     44 |
| Popular   |    132 |     128 |       152 |      155 |    122 |       95 |    116 |

Table 1. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 2. Average Shares vs. Average Log(shares) on Day of the Week")
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

Table 2. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 3. Summary of Average Keywords on Day of the Week")
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

Table 3. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average shares of referenced articles in Mashable on Day of the Week")
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

![](/images/unnamed-chunk-3-1.png)<!-- -->

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

![](/images/unnamed-chunk-4-1.png)<!-- -->

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
b.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##    <fct>     <fct>         <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ##  1 Monday    Unpopular    0.204  0.0600 0.0772 0.106 0.553
    ##  2 Monday    Popular      0.174  0.0718 0.0584 0.115 0.580
    ##  3 Tuesday   Unpopular    0.183  0.0736 0.0742 0.106 0.564
    ##  4 Tuesday   Popular      0.215  0.0724 0.0745 0.130 0.508
    ##  5 Wednesday Unpopular    0.207  0.0692 0.0695 0.102 0.553
    ##  6 Wednesday Popular      0.158  0.0684 0.0886 0.122 0.563
    ##  7 Thursday  Unpopular    0.226  0.0796 0.0917 0.139 0.463
    ##  8 Thursday  Popular      0.210  0.0770 0.0885 0.127 0.497
    ##  9 Friday    Unpopular    0.161  0.0606 0.0917 0.111 0.575
    ## 10 Friday    Popular      0.187  0.0551 0.0878 0.134 0.536
    ## 11 Saturday  Unpopular    0.0518 0.0667 0.0761 0.242 0.563
    ## 12 Saturday  Popular      0.162  0.0769 0.0550 0.259 0.447
    ## 13 Sunday    Unpopular    0.127  0.0672 0.0638 0.173 0.569
    ## 14 Sunday    Popular      0.116  0.0478 0.0500 0.269 0.517

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.204 
    ##  2 Monday   Unpopular    LDA_1      0.0600
    ##  3 Monday   Unpopular    LDA_2      0.0772
    ##  4 Monday   Unpopular    LDA_3      0.106 
    ##  5 Monday   Unpopular    LDA_4      0.553 
    ##  6 Monday   Popular      LDA_0      0.174 
    ##  7 Monday   Popular      LDA_1      0.0718
    ##  8 Monday   Popular      LDA_2      0.0584
    ##  9 Monday   Popular      LDA_3      0.115 
    ## 10 Monday   Popular      LDA_4      0.580 
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

![](/images/unnamed-chunk-5-1.png)<!-- -->

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
l.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##    <fct>     <fct>         <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ##  1 Monday    Unpopular    0.204  0.0600 0.0772 0.106 0.553
    ##  2 Monday    Popular      0.174  0.0718 0.0584 0.115 0.580
    ##  3 Tuesday   Unpopular    0.183  0.0736 0.0742 0.106 0.564
    ##  4 Tuesday   Popular      0.215  0.0724 0.0745 0.130 0.508
    ##  5 Wednesday Unpopular    0.207  0.0692 0.0695 0.102 0.553
    ##  6 Wednesday Popular      0.158  0.0684 0.0886 0.122 0.563
    ##  7 Thursday  Unpopular    0.226  0.0796 0.0917 0.139 0.463
    ##  8 Thursday  Popular      0.210  0.0770 0.0885 0.127 0.497
    ##  9 Friday    Unpopular    0.161  0.0606 0.0917 0.111 0.575
    ## 10 Friday    Popular      0.187  0.0551 0.0878 0.134 0.536
    ## 11 Saturday  Unpopular    0.0518 0.0667 0.0761 0.242 0.563
    ## 12 Saturday  Popular      0.162  0.0769 0.0550 0.259 0.447
    ## 13 Sunday    Unpopular    0.127  0.0672 0.0638 0.173 0.569
    ## 14 Sunday    Popular      0.116  0.0478 0.0500 0.269 0.517

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.204 
    ##  2 Monday   Unpopular    LDA_1      0.0600
    ##  3 Monday   Unpopular    LDA_2      0.0772
    ##  4 Monday   Unpopular    LDA_3      0.106 
    ##  5 Monday   Unpopular    LDA_4      0.553 
    ##  6 Monday   Popular      LDA_0      0.174 
    ##  7 Monday   Popular      LDA_1      0.0718
    ##  8 Monday   Popular      LDA_2      0.0584
    ##  9 Monday   Popular      LDA_3      0.115 
    ## 10 Monday   Popular      LDA_4      0.580 
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

![](/images/unnamed-chunk-6-1.png)<!-- -->

### Scatterplot

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

![](/images/unnamed-chunk-7-1.png)<!-- -->

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
    ## -3.2289 -0.6368 -0.1880  0.5188  4.5760 
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 0.002641   0.065526   0.040   0.9678    
    ## dayweek2                   -0.057646   0.091382  -0.631   0.5283    
    ## dayweek3                   -0.094740   0.088425  -1.071   0.2842    
    ## dayweek4                    0.014446   0.089893   0.161   0.8724    
    ## dayweek5                   -0.074032   0.093826  -0.789   0.4302    
    ## dayweek6                    0.250096   0.113162   2.210   0.0273 *  
    ## dayweek7                    0.119445   0.102199   1.169   0.2427    
    ## kw_avg_avg                  0.143701   0.027247   5.274 1.54e-07 ***
    ## LDA_02                     -0.018331   0.026372  -0.695   0.4871    
    ## self_reference_avg_sharess  0.048951   0.025876   1.892   0.0587 .  
    ## n_non_stop_unique_tokens    0.006191   0.036437   0.170   0.8651    
    ## average_token_length       -0.036571   0.032693  -1.119   0.2635    
    ## n_tokens_content            0.020612   0.029588   0.697   0.4861    
    ## n_tokens_title              0.018692   0.025704   0.727   0.4672    
    ## global_subjectivity        -0.023134   0.029170  -0.793   0.4279    
    ## num_imgs                    0.079470   0.031325   2.537   0.0113 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.976 on 1456 degrees of freedom
    ## Multiple R-squared:  0.05715,    Adjusted R-squared:  0.04743 
    ## F-statistic: 5.883 on 15 and 1456 DF,  p-value: 5.365e-12

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
    ## -3.3603 -0.6325 -0.1848  0.4919  4.5416 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.018607   0.065773  -0.283  0.77729    
    ## dayweek2                                   -0.082428   0.090991  -0.906  0.36515    
    ## dayweek3                                   -0.116766   0.087830  -1.329  0.18391    
    ## dayweek4                                   -0.001824   0.089161  -0.020  0.98368    
    ## dayweek5                                   -0.089129   0.093146  -0.957  0.33879    
    ## dayweek6                                    0.258107   0.112788   2.288  0.02226 *  
    ## dayweek7                                    0.127843   0.101395   1.261  0.20757    
    ## kw_avg_avg                                  0.092180   0.028699   3.212  0.00135 ** 
    ## LDA_02                                     -0.024538   0.026206  -0.936  0.34925    
    ## self_reference_avg_sharess                  0.292465   0.102154   2.863  0.00426 ** 
    ## n_non_stop_unique_tokens                    0.082632   0.044220   1.869  0.06187 .  
    ## average_token_length                        0.045339   0.050276   0.902  0.36732    
    ## n_tokens_content                            0.145309   0.046223   3.144  0.00170 ** 
    ## n_tokens_title                              0.020565   0.025535   0.805  0.42076    
    ## global_subjectivity                        -0.001375   0.030394  -0.045  0.96392    
    ## num_imgs                                    0.053228   0.035089   1.517  0.12950    
    ## `I(n_tokens_content^2)`                    -0.014108   0.005128  -2.751  0.00601 ** 
    ## `kw_avg_avg:num_imgs`                       0.113766   0.026079   4.362 1.38e-05 ***
    ## `average_token_length:global_subjectivity`  0.036838   0.014345   2.568  0.01033 *  
    ## `dayweek2:self_reference_avg_sharess`      -0.229025   0.156374  -1.465  0.14325    
    ## `dayweek3:self_reference_avg_sharess`      -0.294082   0.108113  -2.720  0.00660 ** 
    ## `dayweek4:self_reference_avg_sharess`      -0.254540   0.125093  -2.035  0.04205 *  
    ## `dayweek5:self_reference_avg_sharess`      -0.157732   0.136029  -1.160  0.24642    
    ## `dayweek6:self_reference_avg_sharess`      -0.146304   0.212821  -0.687  0.49191    
    ## `dayweek7:self_reference_avg_sharess`      -0.285283   0.119307  -2.391  0.01692 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9661 on 1447 degrees of freedom
    ## Multiple R-squared:  0.08197,    Adjusted R-squared:  0.06674 
    ## F-statistic: 5.383 on 24 and 1447 DF,  p-value: 1.411e-15

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
    ## -3.3418 -0.6427 -0.1821  0.4819  4.5638 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.0112459  0.0657185  -0.171  0.86415    
    ## dayweek2                                   -0.0836419  0.0907648  -0.922  0.35693    
    ## dayweek3                                   -0.1158250  0.0879668  -1.317  0.18815    
    ## dayweek4                                   -0.0028247  0.0891905  -0.032  0.97474    
    ## dayweek5                                   -0.0931975  0.0931291  -1.001  0.31712    
    ## dayweek6                                    0.2531655  0.1124391   2.252  0.02450 *  
    ## dayweek7                                    0.1276219  0.1014303   1.258  0.20851    
    ## kw_avg_avg                                  0.0916917  0.0288842   3.174  0.00153 ** 
    ## LDA_02                                     -0.0257274  0.0261935  -0.982  0.32616    
    ## self_reference_avg_sharess                  0.1066226  0.0441575   2.415  0.01588 *  
    ## n_non_stop_unique_tokens                    0.0844622  0.0441910   1.911  0.05616 .  
    ## average_token_length                        0.0423815  0.0499659   0.848  0.39646    
    ## n_tokens_content                            0.1456421  0.0460870   3.160  0.00161 ** 
    ## n_tokens_title                              0.0217905  0.0255190   0.854  0.39330    
    ## global_subjectivity                        -0.0003802  0.0303632  -0.013  0.99001    
    ## num_imgs                                    0.0524340  0.0350750   1.495  0.13515    
    ## `I(n_tokens_content^2)`                    -0.0133109  0.0051022  -2.609  0.00918 ** 
    ## `I(self_reference_avg_sharess^2)`          -0.0050271  0.0026681  -1.884  0.05975 .  
    ## `kw_avg_avg:num_imgs`                       0.1049891  0.0258762   4.057 5.23e-05 ***
    ## `average_token_length:global_subjectivity`  0.0369476  0.0142928   2.585  0.00983 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9663 on 1452 degrees of freedom
    ## Multiple R-squared:  0.07839,    Adjusted R-squared:  0.06633 
    ## F-statistic:   6.5 on 19 and 1452 DF,  p-value: < 2.2e-16

``` r
result_tab <- data.frame(t(cv_fit1$results),t(cv_fit2$results), t(cv_fit3$results))
colnames(result_tab) <- c("Model 1","Model 2", "Model 3")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 | Model 3 |
|:-----------|--------:|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |  1.0000 |
| RMSE       |  0.9817 |  0.9732 |  0.9802 |
| Rsquared   |  0.0435 |  0.0568 |  0.0530 |
| MAE        |  0.7542 |  0.7456 |  0.7487 |
| RMSESD     |  0.0528 |  0.0599 |  0.0475 |
| RsquaredSD |  0.0298 |  0.0353 |  0.0462 |
| MAESD      |  0.0326 |  0.0362 |  0.0199 |

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
| Model 1 | 0.9949 |   0.0381 | 0.7468 |
| Model 2 | 0.9897 |   0.0466 | 0.7377 |
| Model 3 | 0.9840 |   0.0547 | 0.7343 |

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

    ## # A tibble: 1,472 x 17
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
    ## # ... with 1,462 more rows, and 8 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
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
    ## 1472 samples
    ##   16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1325, 1326, 1324, 1324, 1325, 1326, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9799056  0.04080666  0.7523847
    ##   1                   50      0.9812772  0.03751165  0.7525142
    ##   1                   75      0.9849988  0.03528134  0.7543478
    ##   1                  100      0.9861487  0.03586993  0.7546050
    ##   2                   25      0.9777712  0.04226639  0.7508166
    ##   2                   50      0.9799891  0.04168048  0.7514737
    ##   2                   75      0.9814034  0.04347557  0.7519968
    ##   2                  100      0.9862036  0.03970495  0.7547282
    ##   3                   25      0.9828033  0.03906998  0.7523992
    ##   3                   50      0.9849012  0.04030301  0.7532025
    ##   3                   75      0.9885755  0.03992793  0.7554085
    ##   3                  100      0.9916412  0.03980989  0.7586587
    ##   4                   25      0.9816170  0.04158678  0.7522120
    ##   4                   50      0.9798736  0.04923190  0.7529326
    ##   4                   75      0.9842059  0.04793681  0.7562425
    ##   4                  100      0.9916832  0.04278703  0.7607390
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse3, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                |   RMSE | Rsquared |    MAE |
|:---------------|-------:|---------:|-------:|
| Linear Model 1 | 0.9949 |   0.0381 | 0.7468 |
| Linear Model 2 | 0.9840 |   0.0547 | 0.7343 |
| Boosted Model  | 0.9985 |   0.0266 | 0.7449 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the lifestyle
channel is “need to automate this part”.

The best model fit to predict the number of shares

# Automation

Automation is done with the modifications of the YAML header and the
render function.
