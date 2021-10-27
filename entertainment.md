ST558 - Project 2 - Predictive Modeling
================
Jasmine Wang & Ilana Feldman
10/31/2021

-   [Introduction - done by Jasmine](#introduction---done-by-jasmine)
-   [Data](#data)
-   [Summarizations](#summarizations)
    -   [Numerical Summaries](#numerical-summaries)
    -   [Visualizations](#visualizations)
-   [Modeling](#modeling)
    -   [Linear Regression](#linear-regression)
    -   [Random Forest](#random-forest)
    -   [Boosted Tree](#boosted-tree)
-   [Model Comparisons](#model-comparisons)
-   [Automation](#automation)
-   [Summarizations](#summarizations-1)
    -   [Numerical Summaries](#numerical-summaries-1)
    -   [Visualizations](#visualizations-1)
    -   [Linear Regression](#linear-regression-1)
    -   [Random Forest](#random-forest-1)
    -   [Boosted Tree](#boosted-tree-1)
-   [Comparison](#comparison)
-   [Automation](#automation-1)

# Introduction - done by Jasmine

briefly describes the data briefly describes the variables you have to
work with (describe what you want to use)

purpose of the analysis methods you will use to model the response (more
details in modeling section)

61 variables (only 58 predictive variables, 2 non-predictive), target
response is “shares”.

# Data

I created a binary response variable, 0 if shares &lt; 1400, 1 if shares
&gt; 1400. “class\_shares”

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

allnews <- read_csv("C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Data/OnlineNewsPopularity.csv", 
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
                           tue = if_else(weekday_is_tuesday == 1, 2, 0), 
                           wed = if_else(weekday_is_wednesday == 1, 3, 0), 
                           thur = if_else(weekday_is_thursday == 1, 4, 0), 
                           fri = if_else(weekday_is_friday == 1, 5, 0),
                           sat = if_else(weekday_is_saturday == 1, 6, 0), 
                           sun = if_else(weekday_is_sunday == 1, 7, 0),
                           dayweek = rowSums(data.frame(weekday_is_monday, tue, wed, thur, fri, sat, sun)))

sel_data <- diffday %>% select(class_shares, shares, dayweek, kw_avg_avg, kw_avg_max, kw_avg_min, kw_max_avg, 
                               log.shares, LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
                               weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday,
                               weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday,
                               self_reference_min_shares, self_reference_avg_sharess, 
                               n_non_stop_unique_tokens, n_unique_tokens, average_token_length, 
                               n_tokens_content, n_tokens_title, global_subjectivity, 
                               num_imgs, num_videos)
sel_data
```

    ## # A tibble: 7,057 x 30
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg log.shares LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday
    ##           <dbl>  <dbl>   <dbl>      <dbl>      <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>
    ##  1            0    593       1         0          0          0          0        6.39 0.500  0.378  0.0400 0.0413 0.0401                 1
    ##  2            0   1200       1         0          0          0          0        7.09 0.0286 0.419  0.495  0.0289 0.0286                 1
    ##  3            1   2100       1         0          0          0          0        7.65 0.0334 0.0345 0.215  0.684  0.0333                 1
    ##  4            0   1200       1         0          0          0          0        7.09 0.126  0.0203 0.0200 0.814  0.0200                 1
    ##  5            1   4600       1         0          0          0          0        8.43 0.200  0.340  0.0333 0.393  0.0333                 1
    ##  6            0   1200       1         0          0          0          0        7.09 0.0240 0.665  0.0225 0.266  0.0223                 1
    ##  7            0    631       1         0          0          0          0        6.45 0.456  0.482  0.0200 0.0213 0.0200                 1
    ##  8            0   1300       2      1114.      5725        461       2019.       7.17 0.0500 0.525  0.324  0.0510 0.0500                 0
    ##  9            1   1700       2       714.      4340        405       2019.       7.44 0.0400 0.840  0.0400 0.0401 0.0400                 0
    ## 10            0    455       3       707.      3833.       469.      1953.       6.12 0.0334 0.409  0.0333 0.491  0.0333                 0
    ## # ... with 7,047 more rows, and 16 more variables: weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
    ## #   weekday_is_friday <dbl>, weekday_is_saturday <dbl>, weekday_is_sunday <dbl>, self_reference_min_shares <dbl>,
    ## #   self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>, average_token_length <dbl>,
    ## #   n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>, num_videos <dbl>

``` r
set.seed(388588)
sharesIndex <- createDataPartition(sel_data$shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]

train1 <- train %>% select(-class_shares, -shares, 
                           -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                           -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday) #keep log.shares
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday) #keep log.shares
```

# Summarizations

## Numerical Summaries

``` r
# contingency table
table1 <- table(train$class_shares, train$dayweek)
colnames(table1) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
rownames(table1) <- c("Unpopular", "Popular")
table1 %>% kable(caption = "Table 1. Popularity on Day of the Week")
```

|           | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|:----------|-------:|--------:|----------:|---------:|-------:|---------:|-------:|
| Unpopular |    614 |     566 |       576 |      536 |    392 |       97 |    134 |
| Popular   |    355 |     345 |       325 |      323 |    269 |      172 |    237 |

Table 1. Popularity on Day of the Week

``` r
train %>% group_by(class_shares, dayweek) %>% summarise(
  Avg = mean(kw_avg_avg), Sd = sd(kw_avg_avg), Median = median(kw_avg_avg), IQR = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 2. Average keyword/Average shares on Day of the Week")
```

| class\_shares | dayweek |      Avg |        Sd |   Median |       IQR |
|--------------:|--------:|---------:|----------:|---------:|----------:|
|             0 |       1 | 3059.804 |  900.4426 | 2940.691 |  861.2788 |
|             0 |       2 | 3020.077 |  892.8909 | 2839.689 |  977.3843 |
|             0 |       3 | 3020.379 |  820.4659 | 2867.709 |  873.6100 |
|             0 |       4 | 3006.293 |  925.5407 | 2851.151 |  935.7669 |
|             0 |       5 | 3039.917 |  767.3336 | 2918.839 |  847.3184 |
|             0 |       6 | 3155.183 |  759.4183 | 3018.077 |  861.0783 |
|             0 |       7 | 2972.381 |  764.0736 | 2904.114 |  859.0615 |
|             1 |       1 | 3273.918 | 1142.8059 | 3136.203 | 1125.5530 |
|             1 |       2 | 3484.806 | 1208.7373 | 3231.094 | 1324.2004 |
|             1 |       3 | 3228.444 | 1061.4483 | 3090.059 | 1130.6604 |
|             1 |       4 | 3339.566 | 1029.9122 | 3145.588 | 1026.5716 |
|             1 |       5 | 3308.803 | 1021.3947 | 3151.540 | 1170.7428 |
|             1 |       6 | 3303.593 | 1212.5216 | 3023.576 | 1286.0784 |
|             1 |       7 | 3347.936 | 1161.3315 | 3137.677 |  972.2244 |

Table 2. Average keyword/Average shares on Day of the Week

``` r
train %>% group_by(class_shares) %>% summarise(
  Avg = mean(kw_avg_avg), Sd = sd(kw_avg_avg), Median = median(kw_avg_avg), IQR = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 2. Average keyword/Average shares on Day of the Week")
```

| class\_shares |      Avg |       Sd |   Median |       IQR |
|--------------:|---------:|---------:|---------:|----------:|
|             0 | 3030.941 |  860.763 | 2882.377 |  913.9985 |
|             1 | 3328.810 | 1118.616 | 3143.719 | 1167.7092 |

Table 2. Average keyword/Average shares on Day of the Week

``` r
train %>% group_by(class_shares, dayweek) %>% summarise(
  Avg = mean(LDA_03), Sd = sd(LDA_03), Median = median(LDA_03), IQR = IQR(LDA_03)) %>% 
  kable(digits = 4, caption = "Table 3. Closeness to LDA topic 3 on Day of the Week")
```

| class\_shares | dayweek |    Avg |     Sd | Median |    IQR |
|--------------:|--------:|-------:|-------:|-------:|-------:|
|             0 |       1 | 0.3480 | 0.3295 | 0.2627 | 0.6368 |
|             0 |       2 | 0.3327 | 0.3366 | 0.2033 | 0.6521 |
|             0 |       3 | 0.3494 | 0.3362 | 0.2262 | 0.6989 |
|             0 |       4 | 0.3181 | 0.3283 | 0.1459 | 0.6029 |
|             0 |       5 | 0.3396 | 0.3361 | 0.1881 | 0.6239 |
|             0 |       6 | 0.3540 | 0.3201 | 0.2351 | 0.6161 |
|             0 |       7 | 0.4403 | 0.3329 | 0.4563 | 0.6999 |
|             1 |       1 | 0.3658 | 0.3304 | 0.2720 | 0.6581 |
|             1 |       2 | 0.3886 | 0.3463 | 0.3209 | 0.7633 |
|             1 |       3 | 0.3518 | 0.3280 | 0.2451 | 0.6533 |
|             1 |       4 | 0.3743 | 0.3393 | 0.2789 | 0.6928 |
|             1 |       5 | 0.3806 | 0.3408 | 0.2844 | 0.7267 |
|             1 |       6 | 0.3467 | 0.3221 | 0.2680 | 0.6333 |
|             1 |       7 | 0.4132 | 0.3336 | 0.3898 | 0.6650 |

Table 3. Closeness to LDA topic 3 on Day of the Week

## Visualizations

``` r
#PLOTS
#train1 <- train %>% select(-class_shares) #keep log.shares
#test1 <- test %>% select(-class_shares) #keep log.shares

correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt")
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n", 
         title="Figure 1. Correlation Between the Variables")
```

![](../images/unnamed-chunk-3-1.png)<!-- -->

``` r
plotdata <- train
plotdata$class.shares <- cut(plotdata$class_shares, 2, c("Unpopular","Popular"))
plotdata$day.week <- cut(plotdata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
plotdata
```

    ## # A tibble: 4,941 x 32
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg log.shares LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday
    ##           <dbl>  <dbl>   <dbl>      <dbl>      <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>
    ##  1            0    593       1         0          0          0          0        6.39 0.500  0.378  0.0400 0.0413 0.0401                 1
    ##  2            0   1200       1         0          0          0          0        7.09 0.0286 0.419  0.495  0.0289 0.0286                 1
    ##  3            1   2100       1         0          0          0          0        7.65 0.0334 0.0345 0.215  0.684  0.0333                 1
    ##  4            0   1200       1         0          0          0          0        7.09 0.126  0.0203 0.0200 0.814  0.0200                 1
    ##  5            0   1200       1         0          0          0          0        7.09 0.0240 0.665  0.0225 0.266  0.0223                 1
    ##  6            0    631       1         0          0          0          0        6.45 0.456  0.482  0.0200 0.0213 0.0200                 1
    ##  7            0   1300       2      1114.      5725        461       2019.       7.17 0.0500 0.525  0.324  0.0510 0.0500                 0
    ##  8            1   6400       3       849.      4600        469.      1953.       8.76 0.0400 0.840  0.0400 0.0404 0.0400                 0
    ##  9            0   1100       3       935.      4760        461       1953.       7.00 0.0400 0.0418 0.258  0.620  0.0400                 0
    ## 10            1   1500       3       827.      4383.       480.      1953.       7.31 0.0333 0.677  0.0333 0.223  0.0335                 0
    ## # ... with 4,931 more rows, and 18 more variables: weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
    ## #   weekday_is_friday <dbl>, weekday_is_saturday <dbl>, weekday_is_sunday <dbl>, self_reference_min_shares <dbl>,
    ## #   self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>, average_token_length <dbl>,
    ## #   n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>, num_videos <dbl>, class.shares <fct>,
    ## #   day.week <fct>

``` r
plot1 <- plotdata %>% group_by(day.week, class.shares) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
plot1
```

    ## # A tibble: 14 x 8
    ##    day.week  class.shares Avg_keyword  LDA_0 LDA_1  LDA_2 LDA_3  LDA_4
    ##    <fct>     <fct>              <dbl>  <dbl> <dbl>  <dbl> <dbl>  <dbl>
    ##  1 Monday    Unpopular          3060. 0.0648 0.433 0.0890 0.348 0.0649
    ##  2 Monday    Popular            3274. 0.0603 0.437 0.0757 0.366 0.0610
    ##  3 Tuesday   Unpopular          3020. 0.0642 0.449 0.0922 0.333 0.0619
    ##  4 Tuesday   Popular            3485. 0.0706 0.400 0.0695 0.389 0.0681
    ##  5 Wednesday Unpopular          3020. 0.0636 0.431 0.0900 0.349 0.0660
    ##  6 Wednesday Popular            3228. 0.0696 0.428 0.0846 0.352 0.0659
    ##  7 Thursday  Unpopular          3006. 0.0713 0.450 0.0944 0.318 0.0659
    ##  8 Thursday  Popular            3340. 0.0609 0.419 0.0863 0.374 0.0592
    ##  9 Friday    Unpopular          3040. 0.0605 0.441 0.0990 0.340 0.0599
    ## 10 Friday    Popular            3309. 0.0690 0.400 0.0861 0.381 0.0646
    ## 11 Saturday  Unpopular          3155. 0.0733 0.367 0.154  0.354 0.0518
    ## 12 Saturday  Popular            3304. 0.0650 0.425 0.0919 0.347 0.0713
    ## 13 Sunday    Unpopular          2972. 0.0662 0.331 0.0919 0.440 0.0708
    ## 14 Sunday    Popular            3348. 0.0847 0.351 0.0918 0.413 0.0592

``` r
plot2 <- plot1 %>% pivot_longer(cols = 4:8, names_to = "LDA.Topic", values_to = "avg.LDA")
plot2
```

    ## # A tibble: 70 x 5
    ##    day.week class.shares Avg_keyword LDA.Topic avg.LDA
    ##    <fct>    <fct>              <dbl> <chr>       <dbl>
    ##  1 Monday   Unpopular          3060. LDA_0      0.0648
    ##  2 Monday   Unpopular          3060. LDA_1      0.433 
    ##  3 Monday   Unpopular          3060. LDA_2      0.0890
    ##  4 Monday   Unpopular          3060. LDA_3      0.348 
    ##  5 Monday   Unpopular          3060. LDA_4      0.0649
    ##  6 Monday   Popular            3274. LDA_0      0.0603
    ##  7 Monday   Popular            3274. LDA_1      0.437 
    ##  8 Monday   Popular            3274. LDA_2      0.0757
    ##  9 Monday   Popular            3274. LDA_3      0.366 
    ## 10 Monday   Popular            3274. LDA_4      0.0610
    ## # ... with 60 more rows

``` r
boxplot1 <- ggplot(data = plotdata, aes(x = class.shares, y = kw_avg_avg))
boxplot1 + geom_boxplot(fill = "white", outlier.shape = NA) + 
  coord_cartesian(ylim = c(0, 10000)) + 
  geom_jitter(aes(color = class.shares), size = 1) + 
  labs(x = "Vaccine Timeline", y = "Active cases", title = "Figure 1. Active cases at each timeline in US") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 14))
```

![](../images/unnamed-chunk-3-2.png)<!-- -->

``` r
scatter1 <- ggplot(data = plotdata, aes(x = kw_avg_avg, y = shares, color = class.shares)) #y=kw_avg_max
scatter1 + geom_point(size = 2) + #aes(shape = class.shares)
  scale_shape_discrete(name = "Day of the Week") + 
  coord_cartesian(ylim = c(0, 100000)) +
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "Average keyword", y = "Best Keyword", title = "Figure 2. Best vs Average keyword for shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/unnamed-chunk-3-3.png)<!-- -->

``` r
l.plot1 <- plotdata %>% group_by(day.week) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 7 x 7
    ##   day.week  Avg_keyword  LDA_0 LDA_1  LDA_2 LDA_3  LDA_4
    ##   <fct>           <dbl>  <dbl> <dbl>  <dbl> <dbl>  <dbl>
    ## 1 Monday          3138. 0.0632 0.435 0.0841 0.355 0.0634
    ## 2 Tuesday         3196. 0.0666 0.430 0.0836 0.354 0.0643
    ## 3 Wednesday       3095. 0.0658 0.430 0.0881 0.350 0.0659
    ## 4 Thursday        3132. 0.0674 0.439 0.0913 0.339 0.0634
    ## 5 Friday          3149. 0.0640 0.424 0.0938 0.356 0.0618
    ## 6 Saturday        3250. 0.0680 0.404 0.114  0.349 0.0643
    ## 7 Sunday          3212. 0.0780 0.344 0.0918 0.423 0.0634

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 35 x 4
    ##    day.week Avg_keyword LDA.Topic avg.LDA
    ##    <fct>          <dbl> <chr>       <dbl>
    ##  1 Monday         3138. LDA_0      0.0632
    ##  2 Monday         3138. LDA_1      0.435 
    ##  3 Monday         3138. LDA_2      0.0841
    ##  4 Monday         3138. LDA_3      0.355 
    ##  5 Monday         3138. LDA_4      0.0634
    ##  6 Tuesday        3196. LDA_0      0.0666
    ##  7 Tuesday        3196. LDA_1      0.430 
    ##  8 Tuesday        3196. LDA_2      0.0836
    ##  9 Tuesday        3196. LDA_3      0.354 
    ## 10 Tuesday        3196. LDA_4      0.0643
    ## # ... with 25 more rows

``` r
lineplot1 <- ggplot(data = l.plot2, aes(x = day.week, y = avg.LDA, color = LDA.Topic))
lineplot1 + geom_line(aes(group=LDA.Topic), lwd = 2) + geom_point() + 
  labs(x = "Day of the Week", y = "Closeness to LDA Topic", 
       title = "Figure 3. Closeness to LDA Topics on Day of the Week") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/unnamed-chunk-3-4.png)<!-- -->

``` r
b.plot1 <- plotdata %>% group_by(class.shares) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
b.plot1
```

    ## # A tibble: 2 x 7
    ##   class.shares Avg_keyword  LDA_0 LDA_1  LDA_2 LDA_3  LDA_4
    ##   <fct>              <dbl>  <dbl> <dbl>  <dbl> <dbl>  <dbl>
    ## 1 Unpopular          3031. 0.0654 0.433 0.0944 0.343 0.0639
    ## 2 Popular            3329. 0.0681 0.410 0.0824 0.375 0.0638

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 10 x 4
    ##    class.shares Avg_keyword LDA.Topic avg.LDA
    ##    <fct>              <dbl> <chr>       <dbl>
    ##  1 Unpopular          3031. LDA_0      0.0654
    ##  2 Unpopular          3031. LDA_1      0.433 
    ##  3 Unpopular          3031. LDA_2      0.0944
    ##  4 Unpopular          3031. LDA_3      0.343 
    ##  5 Unpopular          3031. LDA_4      0.0639
    ##  6 Popular            3329. LDA_0      0.0681
    ##  7 Popular            3329. LDA_1      0.410 
    ##  8 Popular            3329. LDA_2      0.0824
    ##  9 Popular            3329. LDA_3      0.375 
    ## 10 Popular            3329. LDA_4      0.0638

``` r
barplot1 <- ggplot(data = b.plot2, aes(x = class.shares, y = avg.LDA, fill = LDA.Topic))
barplot1 + geom_bar(stat = "identity", position = "dodge") + 
  labs(x = "Popularity", y = "Closeness to Top LDA Topic", title = "Figure 4. ") + 
  scale_fill_discrete(name = "LDA Topics") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 13), 
        axis.title.y = element_text(size = 13), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](../images/unnamed-chunk-3-5.png)<!-- -->

# Modeling

## Linear Regression

``` r
# using train1, dayweek is numeric, no class_shares
#train1 <- train %>% select(-class_shares, -shares) #keep log.shares
#test1 <- test %>% select(-class_shares, -shares) #keep log.shares

preProcValues <- preProcess(train1, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train1)
testTransformed <- predict(preProcValues, test1)

fit1 <- lm(log.shares ~ ., data = trainTransformed)
summary(fit1)
```

    ## 
    ## Call:
    ## lm(formula = log.shares ~ ., data = trainTransformed)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.8046 -0.5976 -0.2191  0.3581  4.4087 
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                -8.602e-15  1.366e-02   0.000  1.00000    
    ## dayweek                     1.093e-01  1.371e-02   7.972 1.92e-15 ***
    ## kw_avg_avg                  3.453e-01  2.758e-02  12.523  < 2e-16 ***
    ## kw_avg_max                 -1.236e-01  1.866e-02  -6.623 3.91e-11 ***
    ## kw_avg_min                 -3.618e-02  1.640e-02  -2.206  0.02744 *  
    ## kw_max_avg                 -1.270e-01  2.558e-02  -4.963 7.19e-07 ***
    ## LDA_00                      1.316e+01  1.306e+01   1.008  0.31372    
    ## LDA_01                      4.602e+01  4.560e+01   1.009  0.31292    
    ## LDA_02                      1.886e+01  1.871e+01   1.008  0.31352    
    ## LDA_03                      4.786e+01  4.744e+01   1.009  0.31304    
    ## LDA_04                      1.318e+01  1.307e+01   1.009  0.31308    
    ## self_reference_min_shares   2.015e-02  1.857e-02   1.085  0.27787    
    ## self_reference_avg_sharess  7.763e-02  1.878e-02   4.133 3.64e-05 ***
    ## n_non_stop_unique_tokens   -8.940e+00  3.151e+00  -2.837  0.00457 ** 
    ## n_unique_tokens             1.100e+01  4.196e+00   2.622  0.00877 ** 
    ## average_token_length       -3.223e-02  2.588e-02  -1.245  0.21304    
    ## n_tokens_content            1.629e-02  2.400e-02   0.679  0.49733    
    ## n_tokens_title              3.795e-03  1.387e-02   0.274  0.78442    
    ## global_subjectivity         5.435e-02  1.877e-02   2.895  0.00381 ** 
    ## num_imgs                    2.890e-02  1.864e-02   1.551  0.12099    
    ## num_videos                  2.068e-02  1.530e-02   1.352  0.17652    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9599 on 4920 degrees of freedom
    ## Multiple R-squared:  0.08229,    Adjusted R-squared:  0.07856 
    ## F-statistic: 22.06 on 20 and 4920 DF,  p-value: < 2.2e-16

``` r
fit2 <- lm(log.shares ~ . + I(n_tokens_content^2) + I(num_imgs^2) + I(num_videos^2), 
           data = trainTransformed)
summary(fit2)
```

    ## 
    ## Call:
    ## lm(formula = log.shares ~ . + I(n_tokens_content^2) + I(num_imgs^2) + 
    ##     I(num_videos^2), data = trainTransformed)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.7950 -0.5995 -0.2181  0.3596  4.4216 
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 0.011750   0.016318   0.720  0.47153    
    ## dayweek                     0.109988   0.013717   8.018 1.33e-15 ***
    ## kw_avg_avg                  0.342580   0.027625  12.401  < 2e-16 ***
    ## kw_avg_max                 -0.125799   0.018726  -6.718 2.05e-11 ***
    ## kw_avg_min                 -0.035951   0.016399  -2.192  0.02841 *  
    ## kw_max_avg                 -0.125699   0.025586  -4.913 9.27e-07 ***
    ## LDA_00                     11.170397  15.321722   0.729  0.46600    
    ## LDA_01                     39.067348  53.484152   0.730  0.46515    
    ## LDA_02                     16.002160  21.939116   0.729  0.46580    
    ## LDA_03                     40.622969  55.633265   0.730  0.46531    
    ## LDA_04                     11.188447  15.322715   0.730  0.46531    
    ## self_reference_min_shares   0.020883   0.018584   1.124  0.26120    
    ## self_reference_avg_sharess  0.076483   0.018822   4.064 4.91e-05 ***
    ## n_non_stop_unique_tokens   -8.443153   3.432062  -2.460  0.01392 *  
    ## n_unique_tokens            10.195835   4.908300   2.077  0.03783 *  
    ## average_token_length       -0.027605   0.028617  -0.965  0.33476    
    ## n_tokens_content            0.001462   0.037797   0.039  0.96915    
    ## n_tokens_title              0.002844   0.013891   0.205  0.83780    
    ## global_subjectivity         0.051685   0.018939   2.729  0.00637 ** 
    ## num_imgs                    0.050673   0.030435   1.665  0.09599 .  
    ## num_videos                  0.068854   0.025737   2.675  0.00749 ** 
    ## I(n_tokens_content^2)       0.004460   0.006997   0.637  0.52387    
    ## I(num_imgs^2)              -0.005788   0.006579  -0.880  0.37904    
    ## I(num_videos^2)            -0.010424   0.004655  -2.239  0.02517 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9597 on 4917 degrees of freedom
    ## Multiple R-squared:  0.08334,    Adjusted R-squared:  0.07905 
    ## F-statistic: 19.44 on 23 and 4917 DF,  p-value: < 2.2e-16

``` r
#I(dayweek^2) + I(kw_avg_avg^2) + I(kw_max_avg^2) + I(LDA_00^2) + 
#                   I(LDA_01^2) + I(LDA_02^2) + I(LDA_03^2) + I(self_reference_min_shares^2) + 
#               I(self_reference_avg_sharess^2) + I(n_non_stop_unique_tokens^2) + I(n_unique_tokens^2) + 
#               I(average_token_length^2) + I(n_tokens_content^2) + I(n_tokens_title^2) + 
#               I(global_subjectivity^2) + I(num_imgs^2) + I(num_videos^2)

cv_fit1 <- train(log.shares ~ . , 
                 data=trainTransformed,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
cv_fit1
```

    ## Linear Regression 
    ## 
    ## 4941 samples
    ##   20 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4447, 4447, 4447, 4447, 4447, 4447, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE      
    ##   1.450166  0.07360728  0.7334548
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
cv_fit2 <- train(log.shares ~ . + I(n_tokens_content^2) + I(num_imgs^2) + I(num_videos^2), 
                 data=trainTransformed,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
cv_fit2
```

    ## Linear Regression 
    ## 
    ## 4941 samples
    ##   20 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4448, 4447, 4447, 4447, 4447, 4447, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE      
    ##   1.227781  0.06767002  0.7233626
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
result_tab <- data.frame(t(cv_fit1$results), t(cv_fit2$results))
colnames(result_tab) <- c("Model 1", "Model 2")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 |
|:-----------|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |
| RMSE       |  1.4502 |  1.2278 |
| Rsquared   |  0.0736 |  0.0677 |
| MAE        |  0.7335 |  0.7234 |
| RMSESD     |  1.5764 |  0.8448 |
| RsquaredSD |  0.0427 |  0.0261 |
| MAESD      |  0.1036 |  0.0589 |

Cross Validation - Comparisons of the models in training set

``` r
pred1 <- predict(cv_fit1, newdata = testTransformed)
pred2 <- predict(cv_fit2, newdata = testTransformed)
cv_rmse1 <- postResample(pred1, obs = testTransformed$log.shares)
cv_rmse2 <- postResample(pred2, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse2)
row.names(result2) <- c("Model 1", "Model 2")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|         |   RMSE | Rsquared |    MAE |
|:--------|-------:|---------:|-------:|
| Model 1 | 0.9791 |   0.0517 | 0.7041 |
| Model 2 | 0.9777 |   0.0537 | 0.7033 |

Cross Validation - Comparisons of the models in test set

## Random Forest

Ilana

``` r
#                        select(class_shares, shares, dayweek, kw_avg_avg, kw_avg_max, kw_avg_min, kw_max_avg, 
#                               log.shares, LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
#                               weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday,
#                               weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday,
#                               self_reference_min_shares, self_reference_avg_sharess, 
#                               n_non_stop_unique_tokens, n_unique_tokens, average_token_length, 
#                               n_tokens_content, n_tokens_title, global_subjectivity, 
#                               num_imgs, num_videos)

train2 <- train %>% select(-class_shares, -shares, -dayweek)
test2 <- test %>% select(-class_shares, -shares, -dayweek)

preProcValues <- preProcess(train2, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train2)
testTransformed <- predict(preProcValues, test2)
```

## Boosted Tree

Jasmine

I need to import the weekdays as dummy variables

``` r
#expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10)
boosted_tree <- train(log.shares ~ . , data = trainTransformed,
      method = "gbm", 
      trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
      tuneGrid = expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 4941 samples
    ##   26 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 5 times) 
    ## Summary of sample sizes: 4448, 4447, 4446, 4448, 4447, 4446, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9642110  0.07937987  0.7125988
    ##   1                   50      0.9574057  0.08666278  0.7057773
    ##   1                  100      0.9541032  0.09039149  0.7028531
    ##   1                  150      0.9538050  0.09051497  0.7024915
    ##   1                  200      0.9539903  0.09029854  0.7027067
    ##   2                   25      0.9600154  0.08278163  0.7077151
    ##   2                   50      0.9555042  0.08792570  0.7031683
    ##   2                  100      0.9547898  0.08890650  0.7018734
    ##   2                  150      0.9556173  0.08807565  0.7024531
    ##   2                  200      0.9569904  0.08702360  0.7025462
    ##   3                   25      0.9586511  0.08374989  0.7061185
    ##   3                   50      0.9548992  0.08847361  0.7022688
    ##   3                  100      0.9565870  0.08657727  0.7027483
    ##   3                  150      0.9593738  0.08394216  0.7039960
    ##   3                  200      0.9624993  0.08099965  0.7057101
    ##   4                   25      0.9590353  0.08212496  0.7061147
    ##   4                   50      0.9571912  0.08460488  0.7032392
    ##   4                  100      0.9588061  0.08399934  0.7037882
    ##   4                  150      0.9626929  0.08044601  0.7064546
    ##   4                  200      0.9660760  0.07805044  0.7088149
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 150, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse2, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                |   RMSE | Rsquared |    MAE |
|:---------------|-------:|---------:|-------:|
| Linear Model 1 | 0.9791 |   0.0517 | 0.7041 |
| Linear Model 2 | 0.9777 |   0.0537 | 0.7033 |
| Boosted Model  | 0.9597 |   0.0788 | 0.6935 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

# Automation

# Summarizations

## Numerical Summaries

## Visualizations

3 \# Modeling

## Linear Regression

## Random Forest

Ilana

## Boosted Tree

Jasmine

# Comparison

# Automation
