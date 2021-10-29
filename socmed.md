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

This analysis is based on the socmed channel popularity.

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

    ## [1] 2323   53

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

    ## # A tibble: 2,323 x 24
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesd~ weekday_is_wedn~
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>             <dbl>            <dbl>
    ##  1            1   2600       7.86       1         0  0.439  0.0225 0.0224 0.0233 0.493                  1                 0                0
    ##  2            0    690       6.54       1         0  0.199  0.248  0.0201 0.513  0.0202                 1                 0                0
    ##  3            1   4800       8.48       1         0  0.0298 0.194  0.0288 0.718  0.0293                 1                 0                0
    ##  4            0    851       6.75       1         0  0.0355 0.0338 0.0336 0.863  0.0341                 1                 0                0
    ##  5            1   4800       8.48       3       832. 0.0231 0.0223 0.0224 0.910  0.0226                 0                 0                1
    ##  6            1   9200       9.13       3      1072. 0.0356 0.0334 0.619  0.0337 0.279                  0                 0                1
    ##  7            1   1600       7.38       3      1564. 0.865  0.0334 0.0336 0.0347 0.0336                 0                 0                1
    ##  8            0    775       6.65       4      1862. 0.509  0.0287 0.0299 0.404  0.0286                 0                 0                0
    ##  9            1  18200       9.81       5      2210. 0.800  0.0500 0.0502 0.0501 0.0500                 0                 0                0
    ## 10            1   1600       7.38       5      1398. 0.189  0.0250 0.319  0.0254 0.441                  0                 0                0
    ## # ... with 2,313 more rows, and 11 more variables: weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
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
We can see the total number of articles from socmed channel falls into
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

Table 3 shows the numerical summaries of average keywords from socmed
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
| Unpopular |     59 |      89 |        77 |       88 |     52 |       21 |     10 |
| Popular   |    173 |     233 |       220 |      239 |    176 |      105 |     86 |

Table 1. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 2. Average Shares vs. Average Log(shares) on Day of the Week")
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

Table 2. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 3. Summary of Average Keywords on Day of the Week")
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

Table 3. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average shares of referenced articles in Mashable on Day of the Week")
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
b.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares LDA_0  LDA_1 LDA_2 LDA_3 LDA_4
    ##    <fct>     <fct>        <dbl>  <dbl> <dbl> <dbl> <dbl>
    ##  1 Monday    Unpopular    0.279 0.167  0.176 0.229 0.148
    ##  2 Monday    Popular      0.408 0.0785 0.180 0.195 0.139
    ##  3 Tuesday   Unpopular    0.290 0.124  0.208 0.195 0.183
    ##  4 Tuesday   Popular      0.382 0.0661 0.219 0.157 0.176
    ##  5 Wednesday Unpopular    0.335 0.105  0.198 0.179 0.182
    ##  6 Wednesday Popular      0.475 0.0598 0.182 0.145 0.138
    ##  7 Thursday  Unpopular    0.280 0.0813 0.255 0.231 0.153
    ##  8 Thursday  Popular      0.462 0.0664 0.163 0.156 0.153
    ##  9 Friday    Unpopular    0.295 0.0841 0.236 0.214 0.170
    ## 10 Friday    Popular      0.434 0.0731 0.185 0.171 0.137
    ## 11 Saturday  Unpopular    0.189 0.125  0.301 0.193 0.193
    ## 12 Saturday  Popular      0.416 0.0804 0.184 0.137 0.183
    ## 13 Sunday    Unpopular    0.199 0.120  0.373 0.172 0.136
    ## 14 Sunday    Popular      0.318 0.0798 0.212 0.232 0.158

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.279 
    ##  2 Monday   Unpopular    LDA_1      0.167 
    ##  3 Monday   Unpopular    LDA_2      0.176 
    ##  4 Monday   Unpopular    LDA_3      0.229 
    ##  5 Monday   Unpopular    LDA_4      0.148 
    ##  6 Monday   Popular      LDA_0      0.408 
    ##  7 Monday   Popular      LDA_1      0.0785
    ##  8 Monday   Popular      LDA_2      0.180 
    ##  9 Monday   Popular      LDA_3      0.195 
    ## 10 Monday   Popular      LDA_4      0.139 
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
and popularity groups for articles in the socmed channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares LDA_0  LDA_1 LDA_2 LDA_3 LDA_4
    ##    <fct>     <fct>        <dbl>  <dbl> <dbl> <dbl> <dbl>
    ##  1 Monday    Unpopular    0.279 0.167  0.176 0.229 0.148
    ##  2 Monday    Popular      0.408 0.0785 0.180 0.195 0.139
    ##  3 Tuesday   Unpopular    0.290 0.124  0.208 0.195 0.183
    ##  4 Tuesday   Popular      0.382 0.0661 0.219 0.157 0.176
    ##  5 Wednesday Unpopular    0.335 0.105  0.198 0.179 0.182
    ##  6 Wednesday Popular      0.475 0.0598 0.182 0.145 0.138
    ##  7 Thursday  Unpopular    0.280 0.0813 0.255 0.231 0.153
    ##  8 Thursday  Popular      0.462 0.0664 0.163 0.156 0.153
    ##  9 Friday    Unpopular    0.295 0.0841 0.236 0.214 0.170
    ## 10 Friday    Popular      0.434 0.0731 0.185 0.171 0.137
    ## 11 Saturday  Unpopular    0.189 0.125  0.301 0.193 0.193
    ## 12 Saturday  Popular      0.416 0.0804 0.184 0.137 0.183
    ## 13 Sunday    Unpopular    0.199 0.120  0.373 0.172 0.136
    ## 14 Sunday    Popular      0.318 0.0798 0.212 0.232 0.158

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.279 
    ##  2 Monday   Unpopular    LDA_1      0.167 
    ##  3 Monday   Unpopular    LDA_2      0.176 
    ##  4 Monday   Unpopular    LDA_3      0.229 
    ##  5 Monday   Unpopular    LDA_4      0.148 
    ##  6 Monday   Popular      LDA_0      0.408 
    ##  7 Monday   Popular      LDA_1      0.0785
    ##  8 Monday   Popular      LDA_2      0.180 
    ##  9 Monday   Popular      LDA_3      0.195 
    ## 10 Monday   Popular      LDA_4      0.139 
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
    ## -6.9291 -0.6113 -0.1448  0.5145  4.0721 
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 0.041612   0.063535   0.655 0.512596    
    ## dayweek2                   -0.125622   0.083451  -1.505 0.132434    
    ## dayweek3                   -0.008039   0.084728  -0.095 0.924419    
    ## dayweek4                   -0.156892   0.083152  -1.887 0.059365 .  
    ## dayweek5                    0.044054   0.090246   0.488 0.625508    
    ## dayweek6                   -0.017382   0.107521  -0.162 0.871596    
    ## dayweek7                    0.193162   0.117851   1.639 0.101403    
    ## kw_avg_avg                  0.090421   0.024821   3.643 0.000278 ***
    ## LDA_02                     -0.091013   0.025394  -3.584 0.000348 ***
    ## self_reference_avg_sharess  0.089774   0.024574   3.653 0.000267 ***
    ## n_non_stop_unique_tokens   -0.200061   0.033766  -5.925 3.81e-09 ***
    ## average_token_length        0.033277   0.026244   1.268 0.204986    
    ## n_tokens_content            0.013137   0.033820   0.388 0.697749    
    ## n_tokens_title             -0.004844   0.024156  -0.201 0.841105    
    ## global_subjectivity        -0.024779   0.025841  -0.959 0.337758    
    ## num_imgs                   -0.135528   0.029898  -4.533 6.25e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9653 on 1612 degrees of freedom
    ## Multiple R-squared:  0.0767, Adjusted R-squared:  0.0681 
    ## F-statistic: 8.927 on 15 and 1612 DF,  p-value: < 2.2e-16

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
    ## -6.6519 -0.5959 -0.1234  0.4919  4.1775 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0311355  0.0645264   0.483 0.629499    
    ## dayweek2                                   -0.1308632  0.0820654  -1.595 0.110994    
    ## dayweek3                                    0.0004311  0.0833726   0.005 0.995875    
    ## dayweek4                                   -0.1401479  0.0818359  -1.713 0.086989 .  
    ## dayweek5                                    0.0518673  0.0887429   0.584 0.558989    
    ## dayweek6                                    0.0069376  0.1061725   0.065 0.947909    
    ## dayweek7                                    0.1899230  0.1160082   1.637 0.101795    
    ## kw_avg_avg                                  0.0821642  0.0270813   3.034 0.002452 ** 
    ## LDA_02                                     -0.0867274  0.0251266  -3.452 0.000572 ***
    ## self_reference_avg_sharess                  0.1263609  0.0614179   2.057 0.039810 *  
    ## n_non_stop_unique_tokens                   -0.2750174  0.0387158  -7.103 1.82e-12 ***
    ## average_token_length                       -0.1111250  0.0372809  -2.981 0.002919 ** 
    ## n_tokens_content                           -0.0650138  0.0520409  -1.249 0.211745    
    ## n_tokens_title                             -0.0066470  0.0238280  -0.279 0.780313    
    ## global_subjectivity                        -0.0449220  0.0257991  -1.741 0.081836 .  
    ## num_imgs                                   -0.1464051  0.0297594  -4.920 9.56e-07 ***
    ## `I(n_tokens_content^2)`                     0.0162710  0.0153882   1.057 0.290502    
    ## `kw_avg_avg:num_imgs`                       0.1070031  0.0290064   3.689 0.000233 ***
    ## `average_token_length:global_subjectivity` -0.0584646  0.0109070  -5.360 9.52e-08 ***
    ## `dayweek2:self_reference_avg_sharess`       0.0093627  0.0855858   0.109 0.912902    
    ## `dayweek3:self_reference_avg_sharess`       0.2187642  0.1285042   1.702 0.088876 .  
    ## `dayweek4:self_reference_avg_sharess`       0.2351740  0.1061740   2.215 0.026901 *  
    ## `dayweek5:self_reference_avg_sharess`      -0.0802568  0.0881442  -0.911 0.362687    
    ## `dayweek6:self_reference_avg_sharess`      -0.1316663  0.1609982  -0.818 0.413586    
    ## `dayweek7:self_reference_avg_sharess`      -0.1372001  0.0718601  -1.909 0.056406 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9484 on 1603 degrees of freedom
    ## Multiple R-squared:  0.1138, Adjusted R-squared:  0.1005 
    ## F-statistic: 8.573 on 24 and 1603 DF,  p-value: < 2.2e-16

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
    ## -6.6762 -0.5878 -0.1129  0.4909  4.2290 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.039781   0.064103   0.621 0.534970    
    ## dayweek2                                   -0.126485   0.081547  -1.551 0.121081    
    ## dayweek3                                   -0.006580   0.082767  -0.079 0.936648    
    ## dayweek4                                   -0.144110   0.081276  -1.773 0.076402 .  
    ## dayweek5                                    0.062522   0.088176   0.709 0.478391    
    ## dayweek6                                    0.012883   0.105472   0.122 0.902798    
    ## dayweek7                                    0.250641   0.115741   2.166 0.030494 *  
    ## kw_avg_avg                                  0.057767   0.024798   2.329 0.019958 *  
    ## LDA_02                                     -0.084626   0.024939  -3.393 0.000707 ***
    ## self_reference_avg_sharess                  0.316910   0.044713   7.088 2.03e-12 ***
    ## n_non_stop_unique_tokens                   -0.261638   0.038390  -6.815 1.33e-11 ***
    ## average_token_length                       -0.108783   0.037051  -2.936 0.003372 ** 
    ## n_tokens_content                           -0.055659   0.051576  -1.079 0.280678    
    ## n_tokens_title                             -0.008339   0.023644  -0.353 0.724363    
    ## global_subjectivity                        -0.045970   0.025636  -1.793 0.073132 .  
    ## num_imgs                                   -0.149091   0.029577  -5.041 5.16e-07 ***
    ## `I(n_tokens_content^2)`                     0.014625   0.015277   0.957 0.338544    
    ## `I(self_reference_avg_sharess^2)`          -0.015385   0.002482  -6.197 7.28e-10 ***
    ## `kw_avg_avg:num_imgs`                       0.092871   0.026998   3.440 0.000597 ***
    ## `average_token_length:global_subjectivity` -0.055468   0.010836  -5.119 3.44e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9428 on 1608 degrees of freedom
    ## Multiple R-squared:  0.1214, Adjusted R-squared:  0.111 
    ## F-statistic:  11.7 on 19 and 1608 DF,  p-value: < 2.2e-16

``` r
result_tab <- data.frame(t(cv_fit1$results),t(cv_fit2$results), t(cv_fit3$results))
colnames(result_tab) <- c("Model 1","Model 2", "Model 3")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 | Model 3 |
|:-----------|--------:|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |  1.0000 |
| RMSE       |  0.9790 |  1.0960 |  0.9924 |
| Rsquared   |  0.0623 |  0.0716 |  0.0902 |
| MAE        |  0.7372 |  0.7454 |  0.7231 |
| RMSESD     |  0.0640 |  0.3819 |  0.1261 |
| RsquaredSD |  0.0265 |  0.0477 |  0.0455 |
| MAESD      |  0.0299 |  0.0546 |  0.0263 |

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
| Model 1 | 1.0765 |   0.0619 | 0.7797 |
| Model 2 | 1.0802 |   0.0570 | 0.7880 |
| Model 3 | 1.0734 |   0.0697 | 0.7820 |

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

    ## # A tibble: 1,628 x 17
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
    ## # ... with 1,618 more rows, and 8 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
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
    ## 1628 samples
    ##   16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1466, 1466, 1466, 1464, 1465, 1466, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared   MAE      
    ##   1                   25      0.9563584  0.1039828  0.7292613
    ##   1                   50      0.9432782  0.1167046  0.7136912
    ##   1                   75      0.9371780  0.1239080  0.7064511
    ##   1                  100      0.9333179  0.1299314  0.7010428
    ##   2                   25      0.9459758  0.1120277  0.7154238
    ##   2                   50      0.9388429  0.1186364  0.7046044
    ##   2                   75      0.9393088  0.1182916  0.7048373
    ##   2                  100      0.9411249  0.1166611  0.7044972
    ##   3                   25      0.9426883  0.1144991  0.7128484
    ##   3                   50      0.9388824  0.1188852  0.7057286
    ##   3                   75      0.9416441  0.1163447  0.7075866
    ##   3                  100      0.9456942  0.1121671  0.7095331
    ##   4                   25      0.9410292  0.1164638  0.7098217
    ##   4                   50      0.9408549  0.1169649  0.7072676
    ##   4                   75      0.9444103  0.1133524  0.7101411
    ##   4                  100      0.9487296  0.1098466  0.7134655
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse3, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                |   RMSE | Rsquared |    MAE |
|:---------------|-------:|---------:|-------:|
| Linear Model 1 | 1.0765 |   0.0619 | 0.7797 |
| Linear Model 2 | 1.0734 |   0.0697 | 0.7820 |
| Boosted Model  | 1.0395 |   0.1268 | 0.7553 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the socmed
channel is “need to automate this part”.

The best model fit to predict the number of shares

# Automation

Automation is done with the modifications of the YAML header and the
render function.
