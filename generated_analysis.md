ST558 - Project 2 - Predictive Modeling
================
Jasmine Wang & Ilana Feldman
10/31/2021

-   [Introduction - done by Jasmine](#introduction---done-by-jasmine)
-   [Data](#data)
-   [Summarizations](#summarizations)
    -   [Numerical Summaries](#numerical-summaries)
    -   [Visualizations](#visualizations)
    -   [Linear Regression (2)](#linear-regression-2)
    -   [Random Forest](#random-forest)
    -   [Boosted Tree](#boosted-tree)
-   [Comparison](#comparison)
-   [Automation](#automation)

# Introduction - done by Jasmine

briefly describes the data briefly describes the variables you have to
work with (describe what you want to use)

purpose of the analysis methods you will use to model the response (more
details in modeling section)

61 variables (only 58 predictive variables, 2 non-predictive), target
response is “shares”.

# Data

I created a binary response variable, 0 if shares &lt; 1400, 1 if shares
&gt; 1400. “class\_shares” I created a categorical variable grouped all
binary variables, monday, tuesday, …, sunday, together. “dayweek” if
dayweek = 1, it’s Monday, 2 is tuesday, 3 is wednesday, …, 7 is sunday.

``` r
library(tidyverse)
library(knitr)
library(caret)
library(corrplot)
library(ggplot2)

allnews <- read_csv("C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Data/OnlineNewsPopularity.csv", 
                 col_names = TRUE)
dim(allnews)
```

    ## [1] 39644    61

``` r
all_news <- allnews %>% mutate(class_shares = if_else(shares < 1400, 0, 1))
news <- all_news %>% filter(data_channel_is_lifestyle == 1) %>% select(
  -data_channel_is_lifestyle, -data_channel_is_entertainment, -data_channel_is_bus, -data_channel_is_socmed, 
  -data_channel_is_tech, -data_channel_is_world, -url, -timedelta)
#news <- allnews %>% filter(data_channel_is_lifestyle == 1) %>% select(
#  n_tokens_title, n_tokens_content, num_hrefs, num_imgs, num_videos, kw_avg_min, kw_avg_max, avg_negative_polarity, 
#  avg_positive_polarity, title_subjectivity, title_sentiment_polarity, shares)
news
```

    ## # A tibble: 2,099 x 54
    ##    n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_tokens num_hrefs num_self_hrefs num_imgs num_videos
    ##             <dbl>            <dbl>           <dbl>            <dbl>                    <dbl>     <dbl>          <dbl>    <dbl>      <dbl>
    ##  1              8              960           0.418             1.00                    0.550        21             20       20          0
    ##  2             10              187           0.667             1.00                    0.800         7              0        1          0
    ##  3             11              103           0.689             1.00                    0.806         3              1        1          0
    ##  4             10              243           0.619             1.00                    0.824         1              1        0          0
    ##  5              8              204           0.586             1.00                    0.698         7              2        1          0
    ##  6             11              315           0.551             1.00                    0.702         4              4        1          0
    ##  7             10             1190           0.409             1.00                    0.561        25             24       20          0
    ##  8              6              374           0.641             1.00                    0.828         7              0        1          0
    ##  9             12              499           0.513             1.00                    0.662        14              1        1          0
    ## 10             11              223           0.662             1.00                    0.826         5              3        0          0
    ## # ... with 2,089 more rows, and 45 more variables: average_token_length <dbl>, num_keywords <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>,
    ## #   self_reference_min_shares <dbl>, self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>,
    ## #   weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
    ## #   weekday_is_sunday <dbl>, is_weekend <dbl>, LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>, LDA_03 <dbl>, LDA_04 <dbl>,
    ## #   global_subjectivity <dbl>, global_sentiment_polarity <dbl>, global_rate_positive_words <dbl>, global_rate_negative_words <dbl>,
    ## #   rate_positive_words <dbl>, rate_negative_words <dbl>, avg_positive_polarity <dbl>, min_positive_polarity <dbl>, ...

``` r
diffday <- news %>% mutate(tue = if_else(weekday_is_tuesday == 1, 2, 0), 
                           wed = if_else(weekday_is_wednesday == 1, 3, 0), 
                           thur = if_else(weekday_is_thursday == 1, 4, 0), 
                           fri = if_else(weekday_is_friday == 1, 5, 0),
                           sat = if_else(weekday_is_saturday == 1, 6, 0), 
                           sun = if_else(weekday_is_sunday == 1, 7, 0),
                           dayweek = as.factor(rowSums(data.frame(weekday_is_monday, tue, wed, thur, fri, sat, sun))))
diffday
```

    ## # A tibble: 2,099 x 61
    ##    n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_tokens num_hrefs num_self_hrefs num_imgs num_videos
    ##             <dbl>            <dbl>           <dbl>            <dbl>                    <dbl>     <dbl>          <dbl>    <dbl>      <dbl>
    ##  1              8              960           0.418             1.00                    0.550        21             20       20          0
    ##  2             10              187           0.667             1.00                    0.800         7              0        1          0
    ##  3             11              103           0.689             1.00                    0.806         3              1        1          0
    ##  4             10              243           0.619             1.00                    0.824         1              1        0          0
    ##  5              8              204           0.586             1.00                    0.698         7              2        1          0
    ##  6             11              315           0.551             1.00                    0.702         4              4        1          0
    ##  7             10             1190           0.409             1.00                    0.561        25             24       20          0
    ##  8              6              374           0.641             1.00                    0.828         7              0        1          0
    ##  9             12              499           0.513             1.00                    0.662        14              1        1          0
    ## 10             11              223           0.662             1.00                    0.826         5              3        0          0
    ## # ... with 2,089 more rows, and 52 more variables: average_token_length <dbl>, num_keywords <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>,
    ## #   self_reference_min_shares <dbl>, self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>,
    ## #   weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
    ## #   weekday_is_sunday <dbl>, is_weekend <dbl>, LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>, LDA_03 <dbl>, LDA_04 <dbl>,
    ## #   global_subjectivity <dbl>, global_sentiment_polarity <dbl>, global_rate_positive_words <dbl>, global_rate_negative_words <dbl>,
    ## #   rate_positive_words <dbl>, rate_negative_words <dbl>, avg_positive_polarity <dbl>, min_positive_polarity <dbl>, ...

``` r
sel_data <- diffday %>% select(class_shares, shares, dayweek, kw_avg_avg, kw_avg_max, kw_avg_min, kw_max_avg, 
                               LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
                               self_reference_min_shares, self_reference_avg_sharess, 
                               n_non_stop_unique_tokens, n_unique_tokens, average_token_length, 
                               n_tokens_content, n_tokens_title, global_subjectivity, 
                               num_imgs, num_videos)
sel_data
```

    ## # A tibble: 2,099 x 22
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 self_reference_~ self_reference_~
    ##           <dbl>  <dbl> <fct>        <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>            <dbl>            <dbl>
    ##  1            0    556 1               0          0          0          0  0.0201 0.115  0.0200 0.0200  0.825              545            3151.
    ##  2            1   1900 1               0          0          0          0  0.0286 0.0286 0.0286 0.0287  0.885                0               0 
    ##  3            1   5700 1               0          0          0          0  0.437  0.200  0.0335 0.0334  0.295             5000            5000 
    ##  4            0    462 1               0          0          0          0  0.0200 0.0200 0.0200 0.0200  0.920                0               0 
    ##  5            1   3600 1               0          0          0          0  0.211  0.0255 0.0251 0.0251  0.713                0               0 
    ##  6            0    343 1               0          0          0          0  0.0201 0.0206 0.0205 0.121   0.818             6200            6200 
    ##  7            0    507 1               0          0          0          0  0.0250 0.160  0.0250 0.0250  0.765              545            3151.
    ##  8            0    552 1               0          0          0          0  0.207  0.146  0.276  0.0251  0.346                0               0 
    ##  9            0   1200 2             885.      3460        581.      2193. 0.0202 0.133  0.120  0.0201  0.707             1300            1300 
    ## 10            1   1900 3            1207.      4517.       748.      1953. 0.0335 0.217  0.0334 0.0335  0.683             6700           11700 
    ## # ... with 2,089 more rows, and 8 more variables: n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>, average_token_length <dbl>,
    ## #   n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>, num_videos <dbl>

``` r
set.seed(388588)
sharesIndex <- createDataPartition(sel_data$class_shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]

train1 <- train %>% select(-class_shares)
test1 <- test %>% select(-class_shares)

# contingency table
table1 <- table(train$class_shares,train$dayweek)
colnames(table1) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
rownames(table1) <- c("Unpopular", "Popular")
table1 %>% kable(caption = "Table 1. Popularity on Day of the Week")
```

|           | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|:----------|-------:|--------:|----------:|---------:|-------:|---------:|-------:|
| Unpopular |     97 |      90 |       115 |      110 |    101 |       24 |     37 |
| Popular   |    135 |     131 |       144 |      150 |    120 |      110 |    106 |

Table 1. Popularity on Day of the Week

``` r
train %>% group_by(class_shares, dayweek) %>% summarise(
  Avg = mean(kw_avg_avg), Sd = sd(kw_avg_avg), Median = median(kw_avg_avg), IQR = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 2. Average keyword/Average shares on Day of the Week")
```

| class\_shares | dayweek |      Avg |        Sd |   Median |       IQR |
|--------------:|:--------|---------:|----------:|---------:|----------:|
|             0 | 1       | 3120.374 |  976.4837 | 3181.430 |  913.4314 |
|             0 | 2       | 3118.129 |  922.6064 | 2948.661 | 1058.3267 |
|             0 | 3       | 3330.738 | 1321.6701 | 3076.391 | 1180.2320 |
|             0 | 4       | 3237.475 | 1184.0388 | 3134.811 | 1179.1829 |
|             0 | 5       | 3083.561 |  992.0930 | 2903.560 | 1291.0737 |
|             0 | 6       | 3934.090 | 2084.0992 | 3476.715 | 1201.7208 |
|             0 | 7       | 3597.312 | 1292.7047 | 3299.847 | 1743.4938 |
|             1 | 1       | 3504.508 | 1655.8421 | 3259.992 | 1201.4987 |
|             1 | 2       | 3436.022 | 1073.6072 | 3305.181 | 1089.5585 |
|             1 | 3       | 3420.250 | 1632.9321 | 3179.649 | 1367.9715 |
|             1 | 4       | 3434.336 | 1312.0479 | 3267.808 | 1370.7682 |
|             1 | 5       | 3234.520 | 1159.7310 | 2998.755 | 1202.6734 |
|             1 | 6       | 3796.391 | 1286.9434 | 3588.583 | 1401.5313 |
|             1 | 7       | 3983.887 | 1348.3090 | 3950.496 | 1372.7139 |

Table 2. Average keyword/Average shares on Day of the Week

``` r
train %>% group_by(class_shares) %>% summarise(
  Avg = mean(kw_avg_avg), Sd = sd(kw_avg_avg), Median = median(kw_avg_avg), IQR = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 2. Average keyword/Average shares on Day of the Week")
```

| class\_shares |      Avg |       Sd |   Median |      IQR |
|--------------:|---------:|---------:|---------:|---------:|
|             0 | 3242.898 | 1181.705 | 3073.956 | 1204.615 |
|             1 | 3525.593 | 1390.698 | 3336.552 | 1346.552 |

Table 2. Average keyword/Average shares on Day of the Week

``` r
train %>% group_by(class_shares, dayweek) %>% summarise(
  Avg = mean(LDA_03), Sd = sd(LDA_03), Median = median(LDA_03), IQR = IQR(LDA_03)) %>% 
  kable(digits = 4, caption = "Table 3. Closeness to LDA topic 3 on Day of the Week")
```

| class\_shares | dayweek |    Avg |     Sd | Median |    IQR |
|--------------:|:--------|-------:|-------:|-------:|-------:|
|             0 | 1       | 0.1162 | 0.1618 | 0.0335 | 0.1248 |
|             0 | 2       | 0.1054 | 0.1744 | 0.0286 | 0.0687 |
|             0 | 3       | 0.1035 | 0.1541 | 0.0289 | 0.1044 |
|             0 | 4       | 0.1311 | 0.1900 | 0.0334 | 0.1809 |
|             0 | 5       | 0.1079 | 0.1647 | 0.0287 | 0.1132 |
|             0 | 6       | 0.2431 | 0.2863 | 0.0413 | 0.4007 |
|             0 | 7       | 0.1887 | 0.2121 | 0.0414 | 0.2818 |
|             1 | 1       | 0.1162 | 0.1604 | 0.0288 | 0.1212 |
|             1 | 2       | 0.1218 | 0.1864 | 0.0288 | 0.1199 |
|             1 | 3       | 0.1402 | 0.1940 | 0.0287 | 0.2012 |
|             1 | 4       | 0.1538 | 0.1926 | 0.0334 | 0.2289 |
|             1 | 5       | 0.1265 | 0.1896 | 0.0286 | 0.1351 |
|             1 | 6       | 0.2254 | 0.2582 | 0.0455 | 0.3443 |
|             1 | 7       | 0.2586 | 0.2601 | 0.1389 | 0.4644 |

Table 3. Closeness to LDA topic 3 on Day of the Week

``` r
#PLOTS
#correlation <- cor(train[, -3], method="spearman")

#corrplot(correlation, type = "upper", tl.pos = "lt")
#corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n")

scatter <- ggplot(data = train, aes(x = kw_avg_avg, y = shares, color = class_shares)) #y=kw_avg_max
scatter + geom_point(size = 3) + #aes(shape = class_shares)
  scale_shape_discrete(name = "Day of the Week") + 
  coord_cartesian() +
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "Average keyword", y = "Best Keyword", title = "Figure 1. Best vs Average keyword for shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/generated_analysis_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

``` r
train$class_shares <- as.factor(train$class_shares)
lineplot1 <- ggplot(data = train, aes(x = dayweek, y = LDA_03, color = class_shares))
lineplot1 + geom_line(aes(group=class_shares), lwd = 2) + geom_point() + #aes(group = State)
  labs(y = "Active Cases", title = "Figure 2. Closeness to LDA Topic 3 on Day of the Week") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/generated_analysis_files/figure-gfm/unnamed-chunk-32-2.png)<!-- -->

``` r
#g <- ggplot(data = barplot1, aes(x = Province, y = Avg_active, fill = vaccine))
#g + geom_bar(stat = "identity", position = "dodge") + 
#  labs(x = "State", y = "Average Active Cases", title = "Figure 3. ") + 
#  scale_fill_discrete(name = "Vaccine timeline") + 
#  theme(axis.text.x = element_text(angle = 45, size = 10), 
#        axis.text.y = element_text(size = 10), 
#        axis.title.x = element_text(size = 13), 
#        axis.title.y = element_text(size = 13), 
#        legend.key.size = unit(1, 'cm'), 
#        legend.text = element_text(size = 13), 
#        title = element_text(size = 13))

cv_fit1 <- train(shares ~ . , 
                 data=train1,
                 method = "lm",
                 preProcess = c("center", "scale"),
                 trControl = trainControl(method = "cv", number = 10))
cv_fit1
```

    ## Linear Regression 
    ## 
    ## 1470 samples
    ##   20 predictor
    ## 
    ## Pre-processing: centered (25), scaled (25) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1322, 1323, 1324, 1322, 1325, 1321, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   8603.215  0.02386289  3571.154
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
train2 <- train %>% select(-shares)
test2 <- test %>% select(-shares)
```

# Summarizations

## Numerical Summaries

## Visualizations

3 \# Modeling

## Linear Regression (2)

one each

## Random Forest

Ilana

## Boosted Tree

Jasmine

# Comparison

# Automation

You can also embed plots, for example:

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
