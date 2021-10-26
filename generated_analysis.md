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

diffday <- news %>% mutate(tue = if_else(weekday_is_tuesday == 1, 2, 0), 
                           wed = if_else(weekday_is_wednesday == 1, 3, 0), 
                           thur = if_else(weekday_is_thursday == 1, 4, 0), 
                           fri = if_else(weekday_is_friday == 1, 5, 0),
                           sat = if_else(weekday_is_saturday == 1, 6, 0), 
                           sun = if_else(weekday_is_sunday == 1, 7, 0),
                           dayweek = rowSums(data.frame(weekday_is_monday, tue, wed, thur, fri, sat, sun)))

sel_data <- diffday %>% select(class_shares, shares, dayweek, kw_avg_avg, kw_avg_max, kw_avg_min, kw_max_avg, 
                               LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
                               self_reference_min_shares, self_reference_avg_sharess, 
                               n_non_stop_unique_tokens, n_unique_tokens, average_token_length, 
                               n_tokens_content, n_tokens_title, global_subjectivity, 
                               num_imgs, num_videos)
sel_data
```

    ## # A tibble: 2,099 x 22
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 self_reference_min_shares
    ##           <dbl>  <dbl>   <dbl>      <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>                     <dbl>
    ##  1            0    556       1         0          0          0          0  0.0201 0.115  0.0200 0.0200  0.825                       545
    ##  2            1   1900       1         0          0          0          0  0.0286 0.0286 0.0286 0.0287  0.885                         0
    ##  3            1   5700       1         0          0          0          0  0.437  0.200  0.0335 0.0334  0.295                      5000
    ##  4            0    462       1         0          0          0          0  0.0200 0.0200 0.0200 0.0200  0.920                         0
    ##  5            1   3600       1         0          0          0          0  0.211  0.0255 0.0251 0.0251  0.713                         0
    ##  6            0    343       1         0          0          0          0  0.0201 0.0206 0.0205 0.121   0.818                      6200
    ##  7            0    507       1         0          0          0          0  0.0250 0.160  0.0250 0.0250  0.765                       545
    ##  8            0    552       1         0          0          0          0  0.207  0.146  0.276  0.0251  0.346                         0
    ##  9            0   1200       2       885.      3460        581.      2193. 0.0202 0.133  0.120  0.0201  0.707                      1300
    ## 10            1   1900       3      1207.      4517.       748.      1953. 0.0335 0.217  0.0334 0.0335  0.683                      6700
    ## # ... with 2,089 more rows, and 9 more variables: self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>,
    ## #   average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>, num_videos <dbl>

``` r
set.seed(388588)
sharesIndex <- createDataPartition(sel_data$class_shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]

train1 <- train %>% select(-class_shares)
test1 <- test %>% select(-class_shares)

# contingency table
table1 <- table(train$class_shares, train$dayweek)
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
|--------------:|--------:|---------:|----------:|---------:|----------:|
|             0 |       1 | 3120.374 |  976.4837 | 3181.430 |  913.4314 |
|             0 |       2 | 3118.129 |  922.6064 | 2948.661 | 1058.3267 |
|             0 |       3 | 3330.738 | 1321.6701 | 3076.391 | 1180.2320 |
|             0 |       4 | 3237.475 | 1184.0388 | 3134.811 | 1179.1829 |
|             0 |       5 | 3083.561 |  992.0930 | 2903.560 | 1291.0737 |
|             0 |       6 | 3934.090 | 2084.0992 | 3476.715 | 1201.7208 |
|             0 |       7 | 3597.312 | 1292.7047 | 3299.847 | 1743.4938 |
|             1 |       1 | 3504.508 | 1655.8421 | 3259.992 | 1201.4987 |
|             1 |       2 | 3436.022 | 1073.6072 | 3305.181 | 1089.5585 |
|             1 |       3 | 3420.250 | 1632.9321 | 3179.649 | 1367.9715 |
|             1 |       4 | 3434.336 | 1312.0479 | 3267.808 | 1370.7682 |
|             1 |       5 | 3234.520 | 1159.7310 | 2998.755 | 1202.6734 |
|             1 |       6 | 3796.391 | 1286.9434 | 3588.583 | 1401.5313 |
|             1 |       7 | 3983.887 | 1348.3090 | 3950.496 | 1372.7139 |

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
|--------------:|--------:|-------:|-------:|-------:|-------:|
|             0 |       1 | 0.1162 | 0.1618 | 0.0335 | 0.1248 |
|             0 |       2 | 0.1054 | 0.1744 | 0.0286 | 0.0687 |
|             0 |       3 | 0.1035 | 0.1541 | 0.0289 | 0.1044 |
|             0 |       4 | 0.1311 | 0.1900 | 0.0334 | 0.1809 |
|             0 |       5 | 0.1079 | 0.1647 | 0.0287 | 0.1132 |
|             0 |       6 | 0.2431 | 0.2863 | 0.0413 | 0.4007 |
|             0 |       7 | 0.1887 | 0.2121 | 0.0414 | 0.2818 |
|             1 |       1 | 0.1162 | 0.1604 | 0.0288 | 0.1212 |
|             1 |       2 | 0.1218 | 0.1864 | 0.0288 | 0.1199 |
|             1 |       3 | 0.1402 | 0.1940 | 0.0287 | 0.2012 |
|             1 |       4 | 0.1538 | 0.1926 | 0.0334 | 0.2289 |
|             1 |       5 | 0.1265 | 0.1896 | 0.0286 | 0.1351 |
|             1 |       6 | 0.2254 | 0.2582 | 0.0455 | 0.3443 |
|             1 |       7 | 0.2586 | 0.2601 | 0.1389 | 0.4644 |

Table 3. Closeness to LDA topic 3 on Day of the Week

``` r
#PLOTS
correlation <- cor(train[ , -1], method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt")
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n", 
         title="Figure 1. Correlation Between the Variables")
```

![](../images/unnamed-chunk-1-1.png)<!-- -->

``` r
plotdata <- train
plotdata$class.shares <- cut(plotdata$class_shares, 2, c("Unpopular","Popular"))
plotdata$day.week <- cut(plotdata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
plotdata
```

    ## # A tibble: 1,470 x 24
    ##    class_shares shares dayweek kw_avg_avg kw_avg_max kw_avg_min kw_max_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 self_reference_min_shares
    ##           <dbl>  <dbl>   <dbl>      <dbl>      <dbl>      <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>                     <dbl>
    ##  1            0    556       1         0          0          0          0  0.0201 0.115  0.0200 0.0200  0.825                       545
    ##  2            0    462       1         0          0          0          0  0.0200 0.0200 0.0200 0.0200  0.920                         0
    ##  3            1   3600       1         0          0          0          0  0.211  0.0255 0.0251 0.0251  0.713                         0
    ##  4            0    343       1         0          0          0          0  0.0201 0.0206 0.0205 0.121   0.818                      6200
    ##  5            0   1200       2       885.      3460        581.      2193. 0.0202 0.133  0.120  0.0201  0.707                      1300
    ##  6            1   1900       3      1207.      4517.       748.      1953. 0.0335 0.217  0.0334 0.0335  0.683                      6700
    ##  7            0   1200       3      1160.      8262.       340.      2011. 0.0200 0.115  0.0201 0.0200  0.825                       545
    ##  8            1   2300       3      1367.      3880.      1096.      3600  0.155  0.0223 0.159  0.0225  0.641                      2000
    ##  9            0    752       3      1223.      6267.       373.      2239. 0.0223 0.381  0.0223 0.0222  0.552                         0
    ## 10            0    866       3      1007.      6110        460.      2011. 0.0201 0.156  0.0200 0.0217  0.782                       545
    ## # ... with 1,460 more rows, and 11 more variables: self_reference_avg_sharess <dbl>, n_non_stop_unique_tokens <dbl>, n_unique_tokens <dbl>,
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
    ##  1 Monday    Unpopular          3120. 0.199  0.0645 0.0814 0.116 0.539
    ##  2 Monday    Popular            3505. 0.191  0.0663 0.0624 0.116 0.564
    ##  3 Tuesday   Unpopular          3118. 0.216  0.0681 0.0644 0.105 0.546
    ##  4 Tuesday   Popular            3436. 0.234  0.0770 0.0693 0.122 0.498
    ##  5 Wednesday Unpopular          3331. 0.213  0.0693 0.0908 0.103 0.524
    ##  6 Wednesday Popular            3420. 0.169  0.0634 0.0897 0.140 0.538
    ##  7 Thursday  Unpopular          3237. 0.198  0.0840 0.106  0.131 0.480
    ##  8 Thursday  Popular            3434. 0.193  0.0760 0.0892 0.154 0.488
    ##  9 Friday    Unpopular          3084. 0.160  0.0654 0.0980 0.108 0.569
    ## 10 Friday    Popular            3235. 0.159  0.0563 0.0988 0.127 0.559
    ## 11 Saturday  Unpopular          3934. 0.0442 0.0571 0.0905 0.243 0.565
    ## 12 Saturday  Popular            3796. 0.197  0.0740 0.0657 0.225 0.438
    ## 13 Sunday    Unpopular          3597. 0.110  0.0623 0.0471 0.189 0.592
    ## 14 Sunday    Popular            3984. 0.109  0.0416 0.0552 0.259 0.535

``` r
plot2 <- plot1 %>% pivot_longer(cols = 4:8, names_to = "LDA.Topic", values_to = "avg.LDA")
plot2
```

    ## # A tibble: 70 x 5
    ##    day.week class.shares Avg_keyword LDA.Topic avg.LDA
    ##    <fct>    <fct>              <dbl> <chr>       <dbl>
    ##  1 Monday   Unpopular          3120. LDA_0      0.199 
    ##  2 Monday   Unpopular          3120. LDA_1      0.0645
    ##  3 Monday   Unpopular          3120. LDA_2      0.0814
    ##  4 Monday   Unpopular          3120. LDA_3      0.116 
    ##  5 Monday   Unpopular          3120. LDA_4      0.539 
    ##  6 Monday   Popular            3505. LDA_0      0.191 
    ##  7 Monday   Popular            3505. LDA_1      0.0663
    ##  8 Monday   Popular            3505. LDA_2      0.0624
    ##  9 Monday   Popular            3505. LDA_3      0.116 
    ## 10 Monday   Popular            3505. LDA_4      0.564 
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

![](../images/unnamed-chunk-1-2.png)<!-- -->

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

![](../images/unnamed-chunk-1-3.png)<!-- -->

``` r
l.plot1 <- plotdata %>% group_by(day.week) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 7 x 7
    ##   day.week  Avg_keyword LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##   <fct>           <dbl> <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ## 1 Monday          3344. 0.195 0.0656 0.0703 0.116 0.553
    ## 2 Tuesday         3307. 0.227 0.0734 0.0673 0.115 0.517
    ## 3 Wednesday       3381. 0.188 0.0660 0.0902 0.124 0.532
    ## 4 Thursday        3351. 0.195 0.0794 0.0964 0.144 0.485
    ## 5 Friday          3166. 0.159 0.0605 0.0984 0.118 0.564
    ## 6 Saturday        3821. 0.169 0.0710 0.0702 0.229 0.461
    ## 7 Sunday          3884. 0.109 0.0469 0.0531 0.241 0.550

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 35 x 4
    ##    day.week Avg_keyword LDA.Topic avg.LDA
    ##    <fct>          <dbl> <chr>       <dbl>
    ##  1 Monday         3344. LDA_0      0.195 
    ##  2 Monday         3344. LDA_1      0.0656
    ##  3 Monday         3344. LDA_2      0.0703
    ##  4 Monday         3344. LDA_3      0.116 
    ##  5 Monday         3344. LDA_4      0.553 
    ##  6 Tuesday        3307. LDA_0      0.227 
    ##  7 Tuesday        3307. LDA_1      0.0734
    ##  8 Tuesday        3307. LDA_2      0.0673
    ##  9 Tuesday        3307. LDA_3      0.115 
    ## 10 Tuesday        3307. LDA_4      0.517 
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

![](../images/unnamed-chunk-1-4.png)<!-- -->

``` r
b.plot1 <- plotdata %>% group_by(class.shares) %>% 
  summarise(Avg_keyword = mean(kw_avg_avg), 
            LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
b.plot1
```

    ## # A tibble: 2 x 7
    ##   class.shares Avg_keyword LDA_0  LDA_1  LDA_2 LDA_3 LDA_4
    ##   <fct>              <dbl> <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ## 1 Unpopular          3243. 0.185 0.0695 0.0864 0.123 0.535
    ## 2 Popular            3526. 0.181 0.0657 0.0767 0.159 0.518

``` r
b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
b.plot2
```

    ## # A tibble: 10 x 4
    ##    class.shares Avg_keyword LDA.Topic avg.LDA
    ##    <fct>              <dbl> <chr>       <dbl>
    ##  1 Unpopular          3243. LDA_0      0.185 
    ##  2 Unpopular          3243. LDA_1      0.0695
    ##  3 Unpopular          3243. LDA_2      0.0864
    ##  4 Unpopular          3243. LDA_3      0.123 
    ##  5 Unpopular          3243. LDA_4      0.535 
    ##  6 Popular            3526. LDA_0      0.181 
    ##  7 Popular            3526. LDA_1      0.0657
    ##  8 Popular            3526. LDA_2      0.0767
    ##  9 Popular            3526. LDA_3      0.159 
    ## 10 Popular            3526. LDA_4      0.518

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

![](../images/unnamed-chunk-1-5.png)<!-- -->

``` r
# using train1, dayweek is numeric, no class_shares
preProcValues <- preProcess(train1, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train1)
testTransformed <- predict(preProcValues, test1)

fit1 <- lm(shares ~ ., data = trainTransformed)
summary(fit1)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ ., data = trainTransformed)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.2129 -0.2844 -0.1457  0.0286 20.9984 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)   
    ## (Intercept)                 6.516e-17  2.565e-02   0.000  1.00000   
    ## dayweek                    -4.816e-02  2.639e-02  -1.825  0.06822 . 
    ## kw_avg_avg                  1.742e-01  6.480e-02   2.688  0.00727 **
    ## kw_avg_max                 -9.804e-02  3.824e-02  -2.564  0.01045 * 
    ## kw_avg_min                 -6.287e-02  3.205e-02  -1.962  0.04998 * 
    ## kw_max_avg                 -4.588e-02  5.311e-02  -0.864  0.38779   
    ## LDA_00                     -7.056e-03  2.717e-02  -0.260  0.79510   
    ## LDA_01                     -2.396e-02  2.644e-02  -0.906  0.36491   
    ## LDA_02                     -1.428e-02  2.801e-02  -0.510  0.61028   
    ## LDA_03                      4.854e-02  3.459e-02   1.403  0.16074   
    ## LDA_04                             NA         NA      NA       NA   
    ## self_reference_min_shares   1.013e-01  4.342e-02   2.333  0.01977 * 
    ## self_reference_avg_sharess -4.837e-02  4.319e-02  -1.120  0.26295   
    ## n_non_stop_unique_tokens    5.908e-02  8.079e-02   0.731  0.46475   
    ## n_unique_tokens             9.548e-03  8.983e-02   0.106  0.91536   
    ## average_token_length       -5.458e-02  3.736e-02  -1.461  0.14431   
    ## n_tokens_content            1.455e-01  4.422e-02   3.291  0.00102 **
    ## n_tokens_title              2.044e-03  2.641e-02   0.077  0.93833   
    ## global_subjectivity        -9.188e-03  3.011e-02  -0.305  0.76027   
    ## num_imgs                   -3.577e-02  3.971e-02  -0.901  0.36779   
    ## num_videos                  8.677e-02  2.667e-02   3.254  0.00117 **
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9834 on 1450 degrees of freedom
    ## Multiple R-squared:  0.04543,    Adjusted R-squared:  0.03292 
    ## F-statistic: 3.632 on 19 and 1450 DF,  p-value: 2.125e-07

``` r
fit2 <- lm(shares ~ . + I(n_tokens_content^2) + I(num_imgs^2) + I(num_videos^2), 
           data = trainTransformed)
summary(fit2)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ . + I(n_tokens_content^2) + I(num_imgs^2) + 
    ##     I(num_videos^2), data = trainTransformed)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -1.4876 -0.2762 -0.1381  0.0321 20.9310 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                 0.023482   0.026446   0.888 0.374727    
    ## dayweek                    -0.053058   0.026204  -2.025 0.043072 *  
    ## kw_avg_avg                  0.147176   0.064812   2.271 0.023304 *  
    ## kw_avg_max                 -0.092029   0.038005  -2.422 0.015578 *  
    ## kw_avg_min                 -0.045288   0.032100  -1.411 0.158517    
    ## kw_max_avg                 -0.033915   0.052783  -0.643 0.520620    
    ## LDA_00                     -0.005796   0.026958  -0.215 0.829796    
    ## LDA_01                     -0.023232   0.026225  -0.886 0.375835    
    ## LDA_02                     -0.010131   0.027840  -0.364 0.715999    
    ## LDA_03                      0.026821   0.034649   0.774 0.439003    
    ## LDA_04                            NA         NA      NA       NA    
    ## self_reference_min_shares   0.078470   0.043319   1.811 0.070281 .  
    ## self_reference_avg_sharess -0.035412   0.042966  -0.824 0.409969    
    ## n_non_stop_unique_tokens    0.136903   0.087540   1.564 0.118063    
    ## n_unique_tokens            -0.074300   0.108466  -0.685 0.493448    
    ## average_token_length       -0.037013   0.041706  -0.887 0.374976    
    ## n_tokens_content            0.043504   0.074089   0.587 0.557168    
    ## n_tokens_title              0.004462   0.026200   0.170 0.864788    
    ## global_subjectivity        -0.022318   0.030203  -0.739 0.460055    
    ## num_imgs                    0.151857   0.058223   2.608 0.009195 ** 
    ## num_videos                  0.214687   0.047130   4.555 5.67e-06 ***
    ## I(n_tokens_content^2)       0.044260   0.012832   3.449 0.000578 ***
    ## I(num_imgs^2)              -0.058965   0.013872  -4.251 2.27e-05 ***
    ## I(num_videos^2)            -0.008793   0.002960  -2.971 0.003018 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9755 on 1447 degrees of freedom
    ## Multiple R-squared:  0.06272,    Adjusted R-squared:  0.04847 
    ## F-statistic: 4.401 on 22 and 1447 DF,  p-value: 6.112e-11

``` r
#I(dayweek^2) + I(kw_avg_avg^2) + I(kw_max_avg^2) + I(LDA_00^2) + 
#                   I(LDA_01^2) + I(LDA_02^2) + I(LDA_03^2) + I(self_reference_min_shares^2) + 
#               I(self_reference_avg_sharess^2) + I(n_non_stop_unique_tokens^2) + I(n_unique_tokens^2) + 
#               I(average_token_length^2) + I(n_tokens_content^2) + I(n_tokens_title^2) + 
#               I(global_subjectivity^2) + I(num_imgs^2) + I(num_videos^2)

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
    ## Pre-processing: centered (20), scaled (20) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1324, 1324, 1322, 1323, 1322, 1321, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   8338.634  0.02819194  3560.248
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
cv_fit2 <- train(shares ~ . + I(n_tokens_content^2) + I(num_imgs^2) + I(num_videos^2), 
                 data=train1,
                 method = "lm",
                 preProcess = c("center", "scale"),
                 trControl = trainControl(method = "cv", number = 10))
cv_fit2
```

    ## Linear Regression 
    ## 
    ## 1470 samples
    ##   20 predictor
    ## 
    ## Pre-processing: centered (23), scaled (23) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1323, 1323, 1321, 1321, 1324, 1323, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   8577.334  0.02363931  3628.087
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
result_tab <- data.frame(t(cv_fit1$results), t(cv_fit2$results))
colnames(result_tab) <- c("Model 1", "Model 2")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            |   Model 1 |   Model 2 |
|:-----------|----------:|----------:|
| intercept  |    1.0000 |    1.0000 |
| RMSE       | 8338.6341 | 8577.3339 |
| Rsquared   |    0.0282 |    0.0236 |
| MAE        | 3560.2478 | 3628.0869 |
| RMSESD     | 5102.2314 | 4855.7113 |
| RsquaredSD |    0.0267 |    0.0341 |
| MAESD      |  598.5546 |  651.6356 |

Cross Validation - Comparisons of the models in training set

``` r
pred1 <- predict(cv_fit1, newdata = test1)
pred2 <- predict(cv_fit2, newdata = test1)
cv_rmse1 <- postResample(pred1, obs = test1$shares)
cv_rmse2 <- postResample(pred2, obs = test1$shares)

result2 <- rbind(cv_rmse1, cv_rmse2)
row.names(result2) <- c("Model 1", "Model 2")
kable(result2, digits = 4, caption = "Cross Validation - Comparisons of the models in test set")
```

|         |     RMSE | Rsquared |      MAE |
|:--------|---------:|---------:|---------:|
| Model 1 | 7343.478 |    0.004 | 3332.207 |
| Model 2 | 8206.548 |    0.001 | 3484.968 |

Cross Validation - Comparisons of the models in test set

``` r
train2 <- train %>% select(-shares)
test2 <- test %>% select(-shares)
train2$class_shares <- as.factor(train2$class_shares)
test2$class_shares <- as.factor(test2$class_shares)
#expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10)
boosted_tree <- train(class_shares ~ . , data = train2,
      method = "gbm", 
      trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
      preProcess = c("center", "scale"),
      tuneGrid = expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 1470 samples
    ##   20 predictor
    ##    2 classes: '0', '1' 
    ## 
    ## Pre-processing: centered (20), scaled (20) 
    ## Resampling: Cross-Validated (10 fold, repeated 5 times) 
    ## Summary of sample sizes: 1324, 1323, 1324, 1323, 1322, 1322, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa     
    ##   1                   25      0.6085801  0.01119758
    ##   1                   50      0.6127877  0.05521320
    ##   1                  100      0.6159225  0.09716441
    ##   1                  150      0.6168620  0.11263500
    ##   1                  200      0.6159104  0.11896515
    ##   2                   25      0.6140029  0.05414795
    ##   2                   50      0.6150795  0.09011348
    ##   2                  100      0.6160522  0.11842818
    ##   2                  150      0.6121148  0.11878070
    ##   2                  200      0.6094028  0.12002088
    ##   3                   25      0.6100611  0.06288190
    ##   3                   50      0.6171221  0.11059656
    ##   3                  100      0.6119612  0.11779505
    ##   3                  150      0.6074758  0.11859744
    ##   3                  200      0.6035210  0.11707133
    ##   4                   25      0.6195924  0.09686241
    ##   4                   50      0.6159180  0.11701880
    ##   4                  100      0.6112901  0.12424049
    ##   4                  150      0.6055832  0.12256033
    ##   4                  200      0.6005406  0.11610351
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 4, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boosted_tree_predict <- predict(boosted_tree, newdata = test2)
confusionMatrix(boosted_tree_predict, test2$class_shares)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0  49  34
    ##          1 193 353
    ##                                           
    ##                Accuracy : 0.6391          
    ##                  95% CI : (0.6002, 0.6767)
    ##     No Information Rate : 0.6153          
    ##     P-Value [Acc > NIR] : 0.1171          
    ##                                           
    ##                   Kappa : 0.1307          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.2025          
    ##             Specificity : 0.9121          
    ##          Pos Pred Value : 0.5904          
    ##          Neg Pred Value : 0.6465          
    ##              Prevalence : 0.3847          
    ##          Detection Rate : 0.0779          
    ##    Detection Prevalence : 0.1320          
    ##       Balanced Accuracy : 0.5573          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

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
