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

This analysis is based on the world channel popularity.

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

    ## [1] 8427   53

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

    ## # A tibble: 5,900 x 11
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_a~ n_non_stop_uniq~ average_token_l~ n_tokens_content n_tokens_title global_subjecti~
    ##         <dbl>   <dbl>      <dbl>  <dbl>             <dbl>            <dbl>            <dbl>            <dbl>          <dbl>            <dbl>
    ##  1       6.57       1         0   0.840                0             0.797             5.09              231             10            0.314
    ##  2       7.70       1         0   0.401                0             0.732             4.62             1248              9            0.482
    ##  3       7.38       1         0   0.867                0             0.635             4.62              682             12            0.473
    ##  4       7.31       1         0   0.700            16100             0.797             4.82              125             11            0.396
    ##  5       7.50       1         0   0.840             1560.            0.729             5.24              317             11            0.375
    ##  6       7.09       1         0   0.485                0             0.806             4.58              399             11            0.565
    ##  7       6.20       1         0   0.702                0             0.589             5.01              443              9            0.420
    ##  8       6.63       2       804.  0.862             3100             0.726             4.38              288             12            0.450
    ##  9       6.15       2       728.  0.700                0             0.777             4.98              414             10            0.343
    ## 10       7.24       3      1047.  0.602                0             0.505             4.15              540             12            0.387
    ## # ... with 5,890 more rows, and 1 more variable: num_imgs <dbl>

``` r
test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04) #keep log.shares
```

# Exploratory Data Analysis

The world channel has 5900 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## Numerical Summaries

Table 1 shows the popularity of the news articles on different days of
the week. I classified number of shares greater than 1400 in a day as
“popular” and number of shares less than 1400 in a day as “unpopular”.
We can see the total number of articles from world channel falls into
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

Table 3 shows the numerical summaries of average keywords from world
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
| Unpopular |    601 |     723 |       726 |      695 |    538 |      157 |    189 |
| Popular   |    372 |     378 |       366 |      376 |    352 |      216 |    211 |

Table 1. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 2. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   2337.043 |  6543.355 |          1100 |        7.1990 |       0.8232 |           7.0031 |
| Tuesday   |   2426.688 |  6565.391 |          1100 |        7.1584 |       0.8682 |           7.0031 |
| Wednesday |   1925.386 |  3406.646 |          1100 |        7.1317 |       0.7749 |           7.0031 |
| Thursday  |   2133.721 |  4561.754 |          1100 |        7.1459 |       0.8352 |           7.0031 |
| Friday    |   2296.185 |  6716.310 |          1100 |        7.2045 |       0.8052 |           7.0031 |
| Saturday  |   2557.188 |  3558.843 |          1600 |        7.4388 |       0.8664 |           7.3778 |
| Sunday    |   2463.262 |  3821.872 |          1400 |        7.4483 |       0.7235 |           7.2442 |

Table 2. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 3. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    2473.181 |   751.0025 |       2385.787 |    745.9407 |
| Tuesday   |    2498.915 |   912.8466 |       2377.593 |    709.0243 |
| Wednesday |    2524.790 |  1086.8147 |       2394.104 |    666.2618 |
| Thursday  |    2553.265 |   866.3338 |       2432.320 |    744.0002 |
| Friday    |    2540.271 |   809.9252 |       2412.769 |    721.8967 |
| Saturday  |    2543.164 |   789.7911 |       2404.729 |    669.8686 |
| Sunday    |    2551.390 |   741.6412 |       2436.715 |    649.0045 |

Table 3. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      3916.641 |     9109.592 |         1500.000 |      2471.000 |
| Tuesday   |      3650.512 |    12051.108 |         1400.000 |      3100.000 |
| Wednesday |      3860.214 |    15032.993 |         1400.000 |      2458.000 |
| Thursday  |      3638.208 |     9786.237 |         1392.571 |      2377.000 |
| Friday    |      5484.440 |    29458.710 |         1489.196 |      2756.583 |
| Saturday  |      3208.286 |     6186.295 |         1300.000 |      2162.000 |
| Sunday    |      3337.209 |     8687.275 |         1500.000 |      2050.125 |

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
dev.off()
```

    ## png 
    ##   2

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
dev.off()
```

    ## png 
    ##   2

### Barplot

Figure 3 shows the popularity of the closeness to a top LDA topic for
the world channel on mashable.com on any day of the week. The
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
file.name <- paste0("../images/", params$channel, 3, ".png")
png(filename = file.name)

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
dev.off()
```

    ## png 
    ##   2

### Line Plot

Here, Figure 4 shows the same measurements as in Figure 3 but in line
plot which we can see how the patterns of the mean ratios of a LDA topic
vary or not vary across time in different popularity groups more
clearly. Again, some mean ratios do not seem to vary across time and
across popularity groups while some other mean ratios vary across time
and popularity groups for articles in the world channel.

``` r
file.name <- paste0("../images/", params$channel, 4, ".png")
png(filename = file.name)

l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))
l.plot1
```

    ## # A tibble: 14 x 7
    ##    day.week  class.shares  LDA_0  LDA_1 LDA_2  LDA_3 LDA_4
    ##    <fct>     <fct>         <dbl>  <dbl> <dbl>  <dbl> <dbl>
    ##  1 Monday    Unpopular    0.0552 0.0526 0.692 0.0665 0.134
    ##  2 Monday    Popular      0.0788 0.0591 0.607 0.0883 0.166
    ##  3 Tuesday   Unpopular    0.0635 0.0497 0.694 0.0543 0.138
    ##  4 Tuesday   Popular      0.0826 0.0555 0.615 0.0888 0.158
    ##  5 Wednesday Unpopular    0.0628 0.0530 0.692 0.0685 0.124
    ##  6 Wednesday Popular      0.0740 0.0516 0.632 0.0776 0.165
    ##  7 Thursday  Unpopular    0.0633 0.0509 0.691 0.0615 0.133
    ##  8 Thursday  Popular      0.0766 0.0575 0.621 0.0884 0.156
    ##  9 Friday    Unpopular    0.0594 0.0549 0.706 0.0679 0.112
    ## 10 Friday    Popular      0.0771 0.0485 0.648 0.0795 0.147
    ## 11 Saturday  Unpopular    0.0526 0.0606 0.714 0.0635 0.109
    ## 12 Saturday  Popular      0.0892 0.0559 0.649 0.0768 0.129
    ## 13 Sunday    Unpopular    0.0575 0.0579 0.695 0.0644 0.125
    ## 14 Sunday    Popular      0.0757 0.0658 0.625 0.0796 0.154

``` r
l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")
l.plot2
```

    ## # A tibble: 70 x 4
    ##    day.week class.shares LDA.Topic avg.LDA
    ##    <fct>    <fct>        <chr>       <dbl>
    ##  1 Monday   Unpopular    LDA_0      0.0552
    ##  2 Monday   Unpopular    LDA_1      0.0526
    ##  3 Monday   Unpopular    LDA_2      0.692 
    ##  4 Monday   Unpopular    LDA_3      0.0665
    ##  5 Monday   Unpopular    LDA_4      0.134 
    ##  6 Monday   Popular      LDA_0      0.0788
    ##  7 Monday   Popular      LDA_1      0.0591
    ##  8 Monday   Popular      LDA_2      0.607 
    ##  9 Monday   Popular      LDA_3      0.0883
    ## 10 Monday   Popular      LDA_4      0.166 
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
dev.off()
```

    ## png 
    ##   2

### Scatterplot

Figure 5 shows the relationship between average keyword and
log-transformed number of shares for articles in the world channel
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
dev.off()
```

    ## png 
    ##   2

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
    ## -4.4554 -0.5457 -0.1652  0.3702  5.5236 
    ## 
    ## Coefficients:
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                -0.002751   0.030770  -0.089   0.9288    
    ## dayweek2                   -0.046973   0.042228  -1.112   0.2660    
    ## dayweek3                   -0.081259   0.042300  -1.921   0.0548 .  
    ## dayweek4                   -0.066960   0.042492  -1.576   0.1151    
    ## dayweek5                   -0.008333   0.044579  -0.187   0.8517    
    ## dayweek6                    0.305409   0.058488   5.222 1.83e-07 ***
    ## dayweek7                    0.304747   0.057147   5.333 1.00e-07 ***
    ## kw_avg_avg                  0.091773   0.012877   7.127 1.15e-12 ***
    ## LDA_02                     -0.123324   0.013046  -9.453  < 2e-16 ***
    ## self_reference_avg_sharess  0.056802   0.012602   4.507 6.69e-06 ***
    ## n_non_stop_unique_tokens    0.046487   0.025219   1.843   0.0653 .  
    ## average_token_length       -0.147636   0.024126  -6.119 1.00e-09 ***
    ## n_tokens_content            0.005144   0.015755   0.326   0.7441    
    ## n_tokens_title              0.028264   0.012579   2.247   0.0247 *  
    ## global_subjectivity         0.106297   0.016556   6.420 1.47e-10 ***
    ## num_imgs                    0.117106   0.014140   8.282  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9588 on 5884 degrees of freedom
    ## Multiple R-squared:  0.08296,    Adjusted R-squared:  0.08062 
    ## F-statistic: 35.49 on 15 and 5884 DF,  p-value: < 2.2e-16

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
    ## -4.4454 -0.5373 -0.1627  0.3689  5.3513 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.029871   0.031987   0.934 0.350430    
    ## dayweek2                                   -0.051402   0.042146  -1.220 0.222667    
    ## dayweek3                                   -0.082557   0.042187  -1.957 0.050406 .  
    ## dayweek4                                   -0.065039   0.042398  -1.534 0.125079    
    ## dayweek5                                   -0.006443   0.044471  -0.145 0.884816    
    ## dayweek6                                    0.319985   0.058665   5.454 5.11e-08 ***
    ## dayweek7                                    0.301551   0.057108   5.280 1.34e-07 ***
    ## kw_avg_avg                                  0.087563   0.012969   6.752 1.60e-11 ***
    ## LDA_02                                     -0.119596   0.013032  -9.177  < 2e-16 ***
    ## self_reference_avg_sharess                  0.132455   0.052368   2.529 0.011455 *  
    ## n_non_stop_unique_tokens                   -0.054433   0.030693  -1.773 0.076202 .  
    ## average_token_length                       -0.304731   0.044298  -6.879 6.65e-12 ***
    ## n_tokens_content                           -0.074987   0.023078  -3.249 0.001163 ** 
    ## n_tokens_title                              0.020020   0.012671   1.580 0.114158    
    ## global_subjectivity                         0.104450   0.016676   6.263 4.03e-10 ***
    ## num_imgs                                    0.110089   0.014953   7.362 2.06e-13 ***
    ## `I(n_tokens_content^2)`                     0.012808   0.003719   3.444 0.000578 ***
    ## `kw_avg_avg:num_imgs`                       0.005288   0.011168   0.473 0.635904    
    ## `average_token_length:global_subjectivity` -0.074859   0.015168  -4.935 8.22e-07 ***
    ## `dayweek2:self_reference_avg_sharess`      -0.003436   0.063894  -0.054 0.957112    
    ## `dayweek3:self_reference_avg_sharess`      -0.090528   0.060133  -1.505 0.132260    
    ## `dayweek4:self_reference_avg_sharess`      -0.065537   0.069694  -0.940 0.347073    
    ## `dayweek5:self_reference_avg_sharess`      -0.099533   0.054950  -1.811 0.070137 .  
    ## `dayweek6:self_reference_avg_sharess`       0.043667   0.134343   0.325 0.745161    
    ## `dayweek7:self_reference_avg_sharess`      -0.041787   0.099903  -0.418 0.675764    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9562 on 5875 degrees of freedom
    ## Multiple R-squared:  0.08944,    Adjusted R-squared:  0.08572 
    ## F-statistic: 24.04 on 24 and 5875 DF,  p-value: < 2.2e-16

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
    ## -4.4489 -0.5407 -0.1628  0.3677  5.3424 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0304397  0.0319105   0.954  0.34017    
    ## dayweek2                                   -0.0494542  0.0420452  -1.176  0.23956    
    ## dayweek3                                   -0.0790586  0.0420920  -1.878  0.06040 .  
    ## dayweek4                                   -0.0625729  0.0422904  -1.480  0.13903    
    ## dayweek5                                   -0.0044289  0.0443591  -0.100  0.92047    
    ## dayweek6                                    0.3183649  0.0582230   5.468 4.74e-08 ***
    ## dayweek7                                    0.3045810  0.0568780   5.355 8.88e-08 ***
    ## kw_avg_avg                                  0.0828304  0.0129750   6.384 1.86e-10 ***
    ## LDA_02                                     -0.1173924  0.0130039  -9.027  < 2e-16 ***
    ## self_reference_avg_sharess                  0.1662317  0.0232132   7.161 8.98e-13 ***
    ## n_non_stop_unique_tokens                   -0.0515534  0.0306075  -1.684  0.09217 .  
    ## average_token_length                       -0.3031317  0.0441563  -6.865 7.33e-12 ***
    ## n_tokens_content                           -0.0703070  0.0230303  -3.053  0.00228 ** 
    ## n_tokens_title                              0.0183542  0.0126362   1.453  0.14641    
    ## global_subjectivity                         0.1019267  0.0166406   6.125 9.65e-10 ***
    ## num_imgs                                    0.1061784  0.0149365   7.109 1.31e-12 ***
    ## `I(n_tokens_content^2)`                     0.0123470  0.0037102   3.328  0.00088 ***
    ## `I(self_reference_avg_sharess^2)`          -0.0044047  0.0007788  -5.656 1.63e-08 ***
    ## `kw_avg_avg:num_imgs`                       0.0051250  0.0111290   0.461  0.64517    
    ## `average_token_length:global_subjectivity` -0.0726858  0.0151340  -4.803 1.60e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9539 on 5880 degrees of freedom
    ## Multiple R-squared:  0.09295,    Adjusted R-squared:  0.09002 
    ## F-statistic: 31.71 on 19 and 5880 DF,  p-value: < 2.2e-16

``` r
result_tab <- data.frame(t(cv_fit1$results),t(cv_fit2$results), t(cv_fit3$results))
colnames(result_tab) <- c("Model 1","Model 2", "Model 3")
rownames(result_tab) <- c("intercept", "RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD")

kable(result_tab, digits = 4, caption = "Cross Validation - Comparisons of the models in training set")
```

|            | Model 1 | Model 2 | Model 3 |
|:-----------|--------:|--------:|--------:|
| intercept  |  1.0000 |  1.0000 |  1.0000 |
| RMSE       |  0.9601 |  0.9631 |  0.9599 |
| Rsquared   |  0.0778 |  0.0807 |  0.0848 |
| MAE        |  0.6775 |  0.6784 |  0.6759 |
| RMSESD     |  0.0502 |  0.0540 |  0.0332 |
| RsquaredSD |  0.0193 |  0.0324 |  0.0306 |
| MAESD      |  0.0282 |  0.0273 |  0.0201 |

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
| Model 1 | 0.9845 |   0.0754 | 0.6977 |
| Model 2 | 0.9970 |   0.0607 | 0.6996 |
| Model 3 | 0.9792 |   0.0848 | 0.6940 |

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

    ## # A tibble: 5,900 x 17
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thur~ weekday_is_frid~ weekday_is_satu~
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>            <dbl>            <dbl>            <dbl>
    ##  1       6.57         0   0.840                 1                  0                    0                0                0                0
    ##  2       7.70         0   0.401                 1                  0                    0                0                0                0
    ##  3       7.38         0   0.867                 1                  0                    0                0                0                0
    ##  4       7.31         0   0.700                 1                  0                    0                0                0                0
    ##  5       7.50         0   0.840                 1                  0                    0                0                0                0
    ##  6       7.09         0   0.485                 1                  0                    0                0                0                0
    ##  7       6.20         0   0.702                 1                  0                    0                0                0                0
    ##  8       6.63       804.  0.862                 0                  1                    0                0                0                0
    ##  9       6.15       728.  0.700                 0                  1                    0                0                0                0
    ## 10       7.24      1047.  0.602                 0                  0                    1                0                0                0
    ## # ... with 5,890 more rows, and 8 more variables: weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
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
    ## 5900 samples
    ##   16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 5309, 5310, 5310, 5311, 5310, 5309, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9635030  0.08648992  0.6844799
    ##   1                   50      0.9520744  0.10102467  0.6744123
    ##   1                   75      0.9464554  0.10742177  0.6693347
    ##   1                  100      0.9441004  0.11032794  0.6669838
    ##   2                   25      0.9547316  0.09940789  0.6759982
    ##   2                   50      0.9459073  0.10835889  0.6678525
    ##   2                   75      0.9428522  0.11256333  0.6653418
    ##   2                  100      0.9420267  0.11422755  0.6654233
    ##   3                   25      0.9516396  0.10230574  0.6723454
    ##   3                   50      0.9445689  0.10988299  0.6662520
    ##   3                   75      0.9428174  0.11298656  0.6657519
    ##   3                  100      0.9423184  0.11386594  0.6651884
    ##   4                   25      0.9489781  0.10520807  0.6714238
    ##   4                   50      0.9447569  0.10943031  0.6681485
    ##   4                   75      0.9422777  0.11409467  0.6672219
    ##   4                  100      0.9430697  0.11310669  0.6680246
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
| Linear Model 1      | 0.9845 |   0.0754 | 0.6977 |
| Linear Model 2      | 0.9792 |   0.0848 | 0.6940 |
| Random Forest Model | 0.9714 |   0.0995 | 0.6858 |
| Boosted Model       | 0.9681 |   0.1055 | 0.6854 |

Cross Validation - Comparisons of the models in test set

# Model Comparisons

The best model fit to predict the number of shares for the world channel
is “need to automate this part”.

The best model fit to predict the number of shares

# Automation

Automation is done with the modifications of the YAML header and the
render function.
