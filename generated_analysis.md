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
&gt; 1400. “class\_shares” I created a categorical variable grouped all
binary variables, monday, tuesday, …, sunday, together. “dayweek” if
dayweek = 1, it’s Monday, 2 is tuesday, 3 is wednesday, …, 7 is sunday.

``` r
library(tidyverse)
library(knitr)
library(caret)
library(corrplot)
library(ggplot2)
library(gbm)

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

diffday <- news %>% mutate(log.shares = log(shares),
                           tue = if_else(weekday_is_tuesday == 1, 2, 0), 
                           wed = if_else(weekday_is_wednesday == 1, 3, 0), 
                           thur = if_else(weekday_is_thursday == 1, 4, 0), 
                           fri = if_else(weekday_is_friday == 1, 5, 0),
                           sat = if_else(weekday_is_saturday == 1, 6, 0), 
                           sun = if_else(weekday_is_sunday == 1, 7, 0),
                           dayweek = rowSums(data.frame(weekday_is_monday, tue, wed, thur, fri, sat, sun)))

sel_data <- diffday %>% select(class_shares, shares, dayweek, kw_avg_avg, kw_avg_max, kw_avg_min, kw_max_avg, 
                               log.shares, LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
                               self_reference_min_shares, self_reference_avg_sharess, 
                               n_non_stop_unique_tokens, n_unique_tokens, average_token_length, 
                               n_tokens_content, n_tokens_title, global_subjectivity, 
                               num_imgs, num_videos)
sel_data
```

    ## # A tibble: 2,099 x 23
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg log.shares LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 self_reference_min_s~
    ##           <dbl>  <dbl>   <dbl>      <dbl>      <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>                 <dbl>
    ##  1            0    556       1         0          0          0          0        6.32 0.0201 0.115  0.0200 0.0200  0.825                   545
    ##  2            1   1900       1         0          0          0          0        7.55 0.0286 0.0286 0.0286 0.0287  0.885                     0
    ##  3            1   5700       1         0          0          0          0        8.65 0.437  0.200  0.0335 0.0334  0.295                  5000
    ##  4            0    462       1         0          0          0          0        6.14 0.0200 0.0200 0.0200 0.0200  0.920                     0
    ##  5            1   3600       1         0          0          0          0        8.19 0.211  0.0255 0.0251 0.0251  0.713                     0
    ##  6            0    343       1         0          0          0          0        5.84 0.0201 0.0206 0.0205 0.121   0.818                  6200
    ##  7            0    507       1         0          0          0          0        6.23 0.0250 0.160  0.0250 0.0250  0.765                   545
    ##  8            0    552       1         0          0          0          0        6.31 0.207  0.146  0.276  0.0251  0.346                     0
    ##  9            0   1200       2       885.      3460        581.      2193.       7.09 0.0202 0.133  0.120  0.0201  0.707                  1300
    ## 10            1   1900       3      1207.      4517.       748.      1953.       7.55 0.0335 0.217  0.0334 0.0335  0.683                  6700
    ## # ... with 2,089 more rows, and 9 more variables: self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>,
    ## #   average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>, num_videos <dbl>

``` r
set.seed(388588)
sharesIndex <- createDataPartition(sel_data$shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]

train1 <- train %>% select(-class_shares, -shares) #keep log.shares
test1 <- test %>% select(-class_shares, -shares) #keep log.shares
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
| Unpopular |     91 |     107 |       121 |       99 |     91 |       19 |     44 |
| Popular   |    132 |     128 |       152 |      155 |    122 |       95 |    116 |

Table 1. Popularity on Day of the Week

``` r
train %>% group_by(class_shares, dayweek) %>% summarise(
  Avg = mean(kw_avg_avg), Sd = sd(kw_avg_avg), Median = median(kw_avg_avg), IQR = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 2. Average keyword/Average shares on Day of the Week")
```

| class\_shares | dayweek |      Avg |        Sd |   Median |      IQR |
|--------------:|--------:|---------:|----------:|---------:|---------:|
|             0 |       1 | 3098.700 | 1329.3232 | 3181.430 | 1077.273 |
|             0 |       2 | 3171.904 | 1028.8289 | 3040.416 | 1059.901 |
|             0 |       3 | 3324.717 | 1266.6794 | 3076.391 | 1186.611 |
|             0 |       4 | 3217.057 | 1211.0340 | 3087.453 | 1154.431 |
|             0 |       5 | 3063.612 |  892.1291 | 2946.626 | 1037.785 |
|             0 |       6 | 3484.992 |  921.3180 | 3452.297 | 1253.694 |
|             0 |       7 | 3358.302 | 1218.8528 | 3051.762 | 1233.854 |
|             1 |       1 | 3348.122 | 1357.5370 | 3193.961 | 1219.741 |
|             1 |       2 | 3340.996 | 1032.8417 | 3197.204 | 1075.079 |
|             1 |       3 | 3552.461 | 2130.6257 | 3274.212 | 1349.575 |
|             1 |       4 | 3371.186 | 1085.5532 | 3367.064 | 1316.527 |
|             1 |       5 | 3296.074 | 1173.4322 | 3076.777 | 1370.557 |
|             1 |       6 | 3787.659 | 1102.4676 | 3578.370 | 1451.018 |
|             1 |       7 | 3992.231 | 1338.4788 | 3977.696 | 1469.784 |

Table 2. Average keyword/Average shares on Day of the Week

``` r
train %>% group_by(class_shares) %>% summarise(
  Avg = mean(kw_avg_avg), Sd = sd(kw_avg_avg), Median = median(kw_avg_avg), IQR = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 2. Average keyword/Average shares on Day of the Week")
```

| class\_shares |      Avg |       Sd |   Median |      IQR |
|--------------:|---------:|---------:|---------:|---------:|
|             0 | 3207.909 | 1157.875 | 3071.825 | 1168.771 |
|             1 | 3507.950 | 1407.509 | 3346.071 | 1400.033 |

Table 2. Average keyword/Average shares on Day of the Week

``` r
train %>% group_by(class_shares, dayweek) %>% summarise(
  Avg = mean(LDA_03), Sd = sd(LDA_03), Median = median(LDA_03), IQR = IQR(LDA_03)) %>% 
  kable(digits = 4, caption = "Table 3. Closeness to LDA topic 3 on Day of the Week")
```

| class\_shares | dayweek |    Avg |     Sd | Median |    IQR |
|--------------:|--------:|-------:|-------:|-------:|-------:|
|             0 |       1 | 0.1060 | 0.1511 | 0.0334 | 0.1145 |
|             0 |       2 | 0.1058 | 0.1782 | 0.0286 | 0.0230 |
|             0 |       3 | 0.1017 | 0.1580 | 0.0287 | 0.0996 |
|             0 |       4 | 0.1394 | 0.2054 | 0.0305 | 0.1819 |
|             0 |       5 | 0.1115 | 0.1563 | 0.0287 | 0.1293 |
|             0 |       6 | 0.2422 | 0.2508 | 0.2392 | 0.3361 |
|             0 |       7 | 0.1734 | 0.1946 | 0.0724 | 0.2728 |
|             1 |       1 | 0.1151 | 0.1532 | 0.0291 | 0.1263 |
|             1 |       2 | 0.1300 | 0.1785 | 0.0320 | 0.1291 |
|             1 |       3 | 0.1217 | 0.1703 | 0.0286 | 0.1488 |
|             1 |       4 | 0.1274 | 0.1721 | 0.0286 | 0.1738 |
|             1 |       5 | 0.1343 | 0.1926 | 0.0287 | 0.1536 |
|             1 |       6 | 0.2588 | 0.2754 | 0.1488 | 0.4742 |
|             1 |       7 | 0.2694 | 0.2581 | 0.2074 | 0.4467 |

Table 3. Closeness to LDA topic 3 on Day of the Week

## Visualizations

``` r
#PLOTS
train1 <- train %>% select(-class_shares) #keep log.shares
test1 <- test %>% select(-class_shares) #keep log.shares

correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt")
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n", 
         title="Figure 1. Correlation Between the Variables")
```

![](/images/unnamed-chunk-22-1.png)<!-- -->

``` r
plotdata <- train
plotdata$class.shares <- cut(plotdata$class_shares, 2, c("Unpopular","Popular"))
plotdata$day.week <- cut(plotdata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
plotdata
```

    ## # A tibble: 1,472 x 25
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg log.shares LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 self_reference_min_s~
    ##           <dbl>  <dbl>   <dbl>      <dbl>      <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>                 <dbl>
    ##  1            0    556       1         0          0          0          0        6.32 0.0201 0.115  0.0200 0.0200  0.825                   545
    ##  2            1   1900       1         0          0          0          0        7.55 0.0286 0.0286 0.0286 0.0287  0.885                     0
    ##  3            1   5700       1         0          0          0          0        8.65 0.437  0.200  0.0335 0.0334  0.295                  5000
    ##  4            0    462       1         0          0          0          0        6.14 0.0200 0.0200 0.0200 0.0200  0.920                     0
    ##  5            1   3600       1         0          0          0          0        8.19 0.211  0.0255 0.0251 0.0251  0.713                     0
    ##  6            0    343       1         0          0          0          0        5.84 0.0201 0.0206 0.0205 0.121   0.818                  6200
    ##  7            0    507       1         0          0          0          0        6.23 0.0250 0.160  0.0250 0.0250  0.765                   545
    ##  8            0    552       1         0          0          0          0        6.31 0.207  0.146  0.276  0.0251  0.346                     0
    ##  9            0   1200       2       885.      3460        581.      2193.       7.09 0.0202 0.133  0.120  0.0201  0.707                  1300
    ## 10            1   1900       3      1207.      4517.       748.      1953.       7.55 0.0335 0.217  0.0334 0.0335  0.683                  6700
    ## # ... with 1,462 more rows, and 11 more variables: self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>,
    ## #   average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>, num_videos <dbl>,
    ## #   class.shares <fct>, day.week <fct>

``` r
plot1 <- plotdata %>% group_by(day.week, class.shares) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
plot1
```

    ## # A tibble: 14 x 8
    ##    day.week  class.shares Avg_keyword  LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##    <fct>     <fct>              <dbl>  <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ##  1 Monday    Unpopular          3099. 0.204  0.0600 0.0772 0.106 0.553
    ##  2 Monday    Popular            3348. 0.174  0.0718 0.0584 0.115 0.580
    ##  3 Tuesday   Unpopular          3172. 0.183  0.0736 0.0742 0.106 0.564
    ##  4 Tuesday   Popular            3341. 0.215  0.0724 0.0745 0.130 0.508
    ##  5 Wednesday Unpopular          3325. 0.207  0.0692 0.0695 0.102 0.553
    ##  6 Wednesday Popular            3552. 0.158  0.0684 0.0886 0.122 0.563
    ##  7 Thursday  Unpopular          3217. 0.226  0.0796 0.0917 0.139 0.463
    ##  8 Thursday  Popular            3371. 0.210  0.0770 0.0885 0.127 0.497
    ##  9 Friday    Unpopular          3064. 0.161  0.0606 0.0917 0.111 0.575
    ## 10 Friday    Popular            3296. 0.187  0.0551 0.0878 0.134 0.536
    ## 11 Saturday  Unpopular          3485. 0.0518 0.0667 0.0761 0.242 0.563
    ## 12 Saturday  Popular            3788. 0.162  0.0769 0.0550 0.259 0.447
    ## 13 Sunday    Unpopular          3358. 0.127  0.0672 0.0638 0.173 0.569
    ## 14 Sunday    Popular            3992. 0.116  0.0478 0.0500 0.269 0.517

``` r
plot2 <- plot1 %>% pivot_longer(cols = 4:8, names_to = "LDA.Topic", values_to = "avg.LDA")
plot2
```

    ## # A tibble: 70 x 5
    ##    day.week class.shares Avg_keyword LDA.Topic avg.LDA
    ##    <fct>    <fct>              <dbl> <chr>       <dbl>
    ##  1 Monday   Unpopular          3099. LDA_0      0.204 
    ##  2 Monday   Unpopular          3099. LDA_1      0.0600
    ##  3 Monday   Unpopular          3099. LDA_2      0.0772
    ##  4 Monday   Unpopular          3099. LDA_3      0.106 
    ##  5 Monday   Unpopular          3099. LDA_4      0.553 
    ##  6 Monday   Popular            3348. LDA_0      0.174 
    ##  7 Monday   Popular            3348. LDA_1      0.0718
    ##  8 Monday   Popular            3348. LDA_2      0.0584
    ##  9 Monday   Popular            3348. LDA_3      0.115 
    ## 10 Monday   Popular            3348. LDA_4      0.580 
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

![](/images/unnamed-chunk-22-2.png)<!-- -->

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

![](/images/unnamed-chunk-22-3.png)<!-- -->

``` r
l.plot1 <- plotdata %>% group_by(day.week) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 7 x 7
    ##   day.week  Avg_keyword LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##   <fct>           <dbl> <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ## 1 Monday          3246. 0.186 0.0669 0.0661 0.111 0.569
    ## 2 Tuesday         3264. 0.200 0.0729 0.0744 0.119 0.533
    ## 3 Wednesday       3452. 0.180 0.0687 0.0801 0.113 0.559
    ## 4 Thursday        3311. 0.216 0.0780 0.0898 0.132 0.484
    ## 5 Friday          3197. 0.176 0.0575 0.0894 0.125 0.553
    ## 6 Saturday        3737. 0.144 0.0752 0.0585 0.256 0.466
    ## 7 Sunday          3818. 0.119 0.0532 0.0538 0.243 0.531

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 35 x 4
    ##    day.week Avg_keyword LDA.Topic avg.LDA
    ##    <fct>          <dbl> <chr>       <dbl>
    ##  1 Monday         3246. LDA_0      0.186 
    ##  2 Monday         3246. LDA_1      0.0669
    ##  3 Monday         3246. LDA_2      0.0661
    ##  4 Monday         3246. LDA_3      0.111 
    ##  5 Monday         3246. LDA_4      0.569 
    ##  6 Tuesday        3264. LDA_0      0.200 
    ##  7 Tuesday        3264. LDA_1      0.0729
    ##  8 Tuesday        3264. LDA_2      0.0744
    ##  9 Tuesday        3264. LDA_3      0.119 
    ## 10 Tuesday        3264. LDA_4      0.533 
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

![](/images/unnamed-chunk-22-4.png)<!-- -->

``` r
b.plot1 <- plotdata %>% group_by(class.shares) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
b.plot1
```

    ## # A tibble: 2 x 7
    ##   class.shares Avg_keyword LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##   <fct>              <dbl> <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ## 1 Unpopular          3208. 0.186 0.0687 0.0788 0.121 0.545
    ## 2 Popular            3508. 0.177 0.0674 0.0735 0.158 0.524

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 10 x 4
    ##    class.shares Avg_keyword LDA.Topic avg.LDA
    ##    <fct>              <dbl> <chr>       <dbl>
    ##  1 Unpopular          3208. LDA_0      0.186 
    ##  2 Unpopular          3208. LDA_1      0.0687
    ##  3 Unpopular          3208. LDA_2      0.0788
    ##  4 Unpopular          3208. LDA_3      0.121 
    ##  5 Unpopular          3208. LDA_4      0.545 
    ##  6 Popular            3508. LDA_0      0.177 
    ##  7 Popular            3508. LDA_1      0.0674
    ##  8 Popular            3508. LDA_2      0.0735
    ##  9 Popular            3508. LDA_3      0.158 
    ## 10 Popular            3508. LDA_4      0.524

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

![](/images/unnamed-chunk-22-5.png)<!-- -->

# Modeling

## Linear Regression

``` r
# using train1, dayweek is numeric, no class_shares
train1 <- train %>% select(-class_shares, -shares) #keep log.shares
test1 <- test %>% select(-class_shares, -shares) #keep log.shares

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
    ## -3.1941 -0.6430 -0.1646  0.5092  4.4792 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 1.974e-16  2.540e-02   0.000  1.00000    
    ## dayweek                     5.045e-02  2.626e-02   1.921  0.05491 .  
    ## kw_avg_avg                  2.933e-01  6.215e-02   4.719  2.6e-06 ***
    ## kw_avg_max                 -6.782e-02  3.763e-02  -1.802  0.07171 .  
    ## kw_avg_min                 -2.108e-02  3.205e-02  -0.658  0.51081    
    ## kw_max_avg                 -1.359e-01  5.162e-02  -2.632  0.00859 ** 
    ## LDA_00                      5.189e-03  2.690e-02   0.193  0.84706    
    ## LDA_01                     -1.041e-02  2.613e-02  -0.398  0.69050    
    ## LDA_02                     -1.910e-02  2.746e-02  -0.696  0.48684    
    ## LDA_03                     -1.835e-02  3.426e-02  -0.536  0.59223    
    ## LDA_04                             NA         NA      NA       NA    
    ## self_reference_min_shares   1.141e-01  4.021e-02   2.838  0.00460 ** 
    ## self_reference_avg_sharess -3.771e-02  3.938e-02  -0.958  0.33834    
    ## n_non_stop_unique_tokens    1.688e-02  7.433e-02   0.227  0.82036    
    ## n_unique_tokens            -1.991e-02  8.359e-02  -0.238  0.81173    
    ## average_token_length       -3.143e-02  3.510e-02  -0.895  0.37078    
    ## n_tokens_content            9.119e-03  4.023e-02   0.227  0.82073    
    ## n_tokens_title              2.529e-02  2.614e-02   0.967  0.33347    
    ## global_subjectivity        -2.206e-02  2.949e-02  -0.748  0.45462    
    ## num_imgs                    8.162e-02  3.584e-02   2.277  0.02293 *  
    ## num_videos                  7.633e-03  2.653e-02   0.288  0.77363    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9746 on 1452 degrees of freedom
    ## Multiple R-squared:  0.06241,    Adjusted R-squared:  0.05014 
    ## F-statistic: 5.087 on 19 and 1452 DF,  p-value: 6.141e-12

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
    ## -3.2280 -0.6360 -0.1646  0.5139  4.5013 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 0.020306   0.026397   0.769  0.44187    
    ## dayweek                     0.048223   0.026236   1.838  0.06627 .  
    ## kw_avg_avg                  0.278251   0.062489   4.453 9.13e-06 ***
    ## kw_avg_max                 -0.067302   0.037665  -1.787  0.07417 .  
    ## kw_avg_min                 -0.015297   0.032242  -0.474  0.63525    
    ## kw_max_avg                 -0.128941   0.051603  -2.499  0.01257 *  
    ## LDA_00                      0.003095   0.026894   0.115  0.90840    
    ## LDA_01                     -0.011517   0.026091  -0.441  0.65899    
    ## LDA_02                     -0.016464   0.027449  -0.600  0.54872    
    ## LDA_03                     -0.025283   0.034543  -0.732  0.46432    
    ## LDA_04                            NA         NA      NA       NA    
    ## self_reference_min_shares   0.116636   0.040314   2.893  0.00387 ** 
    ## self_reference_avg_sharess -0.037357   0.039398  -0.948  0.34319    
    ## n_non_stop_unique_tokens   -0.056294   0.082920  -0.679  0.49732    
    ## n_unique_tokens             0.124439   0.104309   1.193  0.23307    
    ## average_token_length       -0.069307   0.038949  -1.779  0.07538 .  
    ## n_tokens_content            0.135940   0.069393   1.959  0.05031 .  
    ## n_tokens_title              0.029835   0.026196   1.139  0.25494    
    ## global_subjectivity        -0.031213   0.029777  -1.048  0.29471    
    ## num_imgs                    0.108203   0.049581   2.182  0.02924 *  
    ## num_videos                  0.064735   0.047951   1.350  0.17723    
    ## I(n_tokens_content^2)      -0.013537   0.007743  -1.748  0.08063 .  
    ## I(num_imgs^2)              -0.002317   0.007106  -0.326  0.74447    
    ## I(num_videos^2)            -0.004466   0.003037  -1.471  0.14155    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9729 on 1449 degrees of freedom
    ## Multiple R-squared:  0.06762,    Adjusted R-squared:  0.05347 
    ## F-statistic: 4.777 on 22 and 1449 DF,  p-value: 2.709e-12

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
    ## 1472 samples
    ##   20 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1325, 1325, 1325, 1326, 1325, 1324, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared    MAE     
    ##   0.9813569  0.04456415  0.752103
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
    ## 1472 samples
    ##   20 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1324, 1325, 1325, 1326, 1324, 1324, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared    MAE      
    ##   0.9838479  0.04826519  0.7508475
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
| RMSE       |  0.9814 |  0.9838 |
| Rsquared   |  0.0446 |  0.0483 |
| MAE        |  0.7521 |  0.7508 |
| RMSESD     |  0.0501 |  0.0739 |
| RsquaredSD |  0.0353 |  0.0490 |
| MAESD      |  0.0330 |  0.0458 |

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
| Model 1 | 0.9841 |   0.0542 | 0.7338 |
| Model 2 | 0.9846 |   0.0533 | 0.7326 |

Cross Validation - Comparisons of the models in test set

## Random Forest

Ilana

## Boosted Tree

Jasmine

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
    ## 1472 samples
    ##   20 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 5 times) 
    ## Summary of sample sizes: 1325, 1325, 1324, 1326, 1325, 1324, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9800154  0.04220822  0.7504828
    ##   1                   50      0.9783305  0.04473271  0.7475627
    ##   1                  100      0.9804705  0.04415813  0.7475341
    ##   1                  150      0.9842315  0.04051909  0.7498831
    ##   1                  200      0.9870658  0.03903579  0.7520712
    ##   2                   25      0.9787739  0.04432395  0.7482515
    ##   2                   50      0.9821395  0.04115217  0.7495890
    ##   2                  100      0.9879828  0.03940829  0.7533882
    ##   2                  150      0.9941154  0.03685747  0.7573299
    ##   2                  200      1.0007602  0.03455854  0.7620241
    ##   3                   25      0.9787759  0.04431048  0.7472368
    ##   3                   50      0.9832517  0.04075586  0.7494483
    ##   3                  100      0.9923272  0.03730622  0.7557478
    ##   3                  150      1.0007873  0.03367677  0.7620960
    ##   3                  200      1.0085492  0.03135485  0.7696132
    ##   4                   25      0.9803496  0.04143731  0.7473339
    ##   4                   50      0.9852857  0.03958637  0.7513406
    ##   4                  100      0.9985902  0.03457621  0.7594116
    ##   4                  150      1.0097772  0.03222937  0.7678542
    ##   4                  200      1.0201455  0.03005231  0.7764080
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = testTransformed)

boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed$log.shares)

result2 <- rbind(cv_rmse1, cv_rmse2, boost_rmse)
row.names(result2) <- c("Linear Model 1", "Linear Model 2", "Boosted Model")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|                |   RMSE | Rsquared |    MAE |
|:---------------|-------:|---------:|-------:|
| Linear Model 1 | 0.9841 |   0.0542 | 0.7338 |
| Linear Model 2 | 0.9846 |   0.0533 | 0.7326 |
| Boosted Model  | 0.9854 |   0.0516 | 0.7407 |

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
