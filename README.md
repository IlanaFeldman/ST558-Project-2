# Predicting the Popularity of Online News
Collaborative project between Ilana Feldman and Jasmine Wang.

## Purpose of the repo




## A list of R packages used to generate the analyses

   * tidyverse
   * knitr
   * caret
   * corrplot
   * ggplot2
   * gbm

## Links to each of the generated analyses using automation

   * The analysis for [Lifestyle articles is available here](lifestyle.html)
   * The analysis for [Entertainment articles is available here](entertainment.html)
   * The analysis for [Business articles is available here](bus.html)
   * The analysis for [Social Media articles is available here](socmed.html)
   * The analysis for [Technology articles is available here](tech.html)
   * The analysis for [World articles is available here](world.html)

## The render function used to create six analyses from a single .Rmd file for different channels

```{r eval=FALSE}  
  library(rmarkdown)  
  library(tidyverse)  
  type <- c("lifestyle", "entertainment") #, "bus", "socmed", "tech", "world"  
  output_file <- paste0(type, ".md")  
  params <- lapply(type, FUN = function(x){list(channel = x)})  
  reports <- tibble(output_file, params)  
  
  apply(reports, MARGIN = 1,  
        FUN = function(x){  
          render(input = "C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Rmd/ST558_project2_auto.Rmd",  
                 output_format = "github_document",  
                 output_file = paste0("C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/", x[[1]]),  
                 params = x[[2]],  
                 output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble"))  
        })  
```  
