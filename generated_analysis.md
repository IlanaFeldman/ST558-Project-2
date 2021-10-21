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

``` r
library(tidyverse)
library(knitr)
library(caret)

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
set.seed(388588)
sharesIndex <- createDataPartition(news$shares, p = 0.7, list = FALSE)
train <- news[sharesIndex, ]
test <- news[-sharesIndex, ]

train1 <- train %>% select(-class_shares)
test1 <- test %>% select(-class_shares)
fit1 <- lm(shares ~ . , data = train1)
summary(fit1)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ ., data = train1)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -10784  -2600  -1288    417 195628 
    ## 
    ## Coefficients: (4 not defined because of singularities)
    ##                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   9.732e+02  3.263e+03   0.298  0.76556    
    ## n_tokens_title                2.169e+00  1.111e+02   0.020  0.98443    
    ## n_tokens_content              7.726e-01  6.283e-01   1.230  0.21900    
    ## n_unique_tokens              -3.170e+03  7.051e+03  -0.450  0.65311    
    ## n_non_stop_words             -2.711e+03  6.544e+03  -0.414  0.67870    
    ## n_non_stop_unique_tokens      9.303e+03  6.004e+03   1.549  0.12151    
    ## num_hrefs                     3.554e+01  2.287e+01   1.554  0.12041    
    ## num_self_hrefs               -5.459e+01  8.141e+01  -0.671  0.50260    
    ## num_imgs                      3.067e+01  3.881e+01   0.790  0.42952    
    ## num_videos                    5.523e+01  9.959e+01   0.555  0.57925    
    ## average_token_length         -2.749e+02  8.766e+02  -0.314  0.75385    
    ## num_keywords                 -7.263e+01  1.479e+02  -0.491  0.62349    
    ## kw_min_min                   -2.145e+00  4.594e+00  -0.467  0.64063    
    ## kw_max_min                    1.182e-01  1.888e-01   0.626  0.53129    
    ## kw_avg_min                   -1.608e+00  1.390e+00  -1.157  0.24737    
    ## kw_min_max                    1.423e-03  1.499e-02   0.095  0.92434    
    ## kw_max_max                   -9.848e-04  1.753e-03  -0.562  0.57428    
    ## kw_avg_max                   -8.402e-03  4.704e-03  -1.786  0.07428 .  
    ## kw_min_avg                   -7.982e-01  2.600e-01  -3.070  0.00218 ** 
    ## kw_max_avg                   -1.869e-01  7.639e-02  -2.447  0.01451 *  
    ## kw_avg_avg                    2.218e+00  4.830e-01   4.591 4.79e-06 ***
    ## self_reference_min_shares     6.963e-02  5.710e-02   1.219  0.22291    
    ## self_reference_max_shares     1.967e-02  3.371e-02   0.583  0.55971    
    ## self_reference_avg_sharess   -5.941e-02  8.190e-02  -0.725  0.46831    
    ## weekday_is_monday             4.006e+02  8.282e+02   0.484  0.62869    
    ## weekday_is_tuesday            9.832e+02  8.171e+02   1.203  0.22909    
    ## weekday_is_wednesday         -2.318e+01  7.973e+02  -0.029  0.97681    
    ## weekday_is_thursday           4.438e+02  8.087e+02   0.549  0.58330    
    ## weekday_is_friday            -3.062e+02  8.335e+02  -0.367  0.71343    
    ## weekday_is_saturday           4.551e+02  9.534e+02   0.477  0.63320    
    ## weekday_is_sunday                    NA         NA      NA       NA    
    ## is_weekend                           NA         NA      NA       NA    
    ## LDA_00                        2.410e+02  8.558e+02   0.282  0.77831    
    ## LDA_01                       -2.797e+03  2.227e+03  -1.256  0.20929    
    ## LDA_02                       -2.830e+02  2.094e+03  -0.135  0.89250    
    ## LDA_03                       -4.474e+01  1.445e+03  -0.031  0.97531    
    ## LDA_04                               NA         NA      NA       NA    
    ## global_subjectivity          -3.265e+02  3.267e+03  -0.100  0.92040    
    ## global_sentiment_polarity    -4.460e+03  6.422e+03  -0.694  0.48751    
    ## global_rate_positive_words    8.765e+03  2.688e+04   0.326  0.74437    
    ## global_rate_negative_words   -1.613e+04  6.277e+04  -0.257  0.79728    
    ## rate_positive_words          -1.132e+03  5.081e+03  -0.223  0.82379    
    ## rate_negative_words                  NA         NA      NA       NA    
    ## avg_positive_polarity         2.243e+03  5.229e+03   0.429  0.66800    
    ## min_positive_polarity         3.077e+03  4.145e+03   0.742  0.45808    
    ## max_positive_polarity        -2.398e+03  1.611e+03  -1.488  0.13691    
    ## avg_negative_polarity         3.429e+03  4.431e+03   0.774  0.43914    
    ## min_negative_polarity        -1.646e+03  1.533e+03  -1.073  0.28326    
    ## max_negative_polarity        -1.670e+03  3.915e+03  -0.427  0.66974    
    ## title_subjectivity            1.186e+03  9.172e+02   1.293  0.19637    
    ## title_sentiment_polarity     -2.720e+02  9.113e+02  -0.298  0.76538    
    ## abs_title_subjectivity        1.456e+03  1.277e+03   1.141  0.25423    
    ## abs_title_sentiment_polarity -5.782e+02  1.311e+03  -0.441  0.65925    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7660 on 1423 degrees of freedom
    ## Multiple R-squared:  0.05299,    Adjusted R-squared:  0.02105 
    ## F-statistic: 1.659 on 48 and 1423 DF,  p-value: 0.003403

``` r
cv_fit1 <- train(shares ~ . , 
                 data=train1,
                 method = "lm",
                 preProcess = c("center", "scale"),
                 trControl = trainControl(method = "cv", number = 10))
cv_fit1
```

    ## Linear Regression 
    ## 
    ## 1472 samples
    ##   52 predictor
    ## 
    ## Pre-processing: centered (52), scaled (52) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1326, 1325, 1324, 1324, 1324, 1325, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   6716.527  0.02629579  3296.362
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
train2 <- train %>% select(-shares)
test2 <- test %>% select(-shares)
```

# Summarizations

## Numerical Summaries

## Visualizations

# Modeling

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
