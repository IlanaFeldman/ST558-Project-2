# ST558-Project-2
Collaborative project between Ilana Feldman and Jasmine Wang.

1. Brief description of the purpose of the repo




2. List of R packages were used to generate the analysis

   * tidyverse
   * knitr
   * caret
   * corrplot
   * ggplot2
   * gbm

3. A link to the generated analysis

   * The analysis for [Lifestyle articles is available here](lifestyle.html)
   * The analysis for [Entertainment articles is available here](entertainment.html)
   * The analysis for [Business articles is available here](bus.html)
   * The analysis for [Social Media articles is available here](socmed.html)
   * The analysis for [Technology articles is available here](tech.html)
   * The analysis for [World articles is available here](world.html)

4. A render function to generate the six analysese for different articles

```{r eval=FALSE}
library(rmarkdown)
library(tidyverse)
type <- c("lifestyle", "entertainment") #, "bus", "socmed", "tech", "world"
output_file <- paste0(type, ".md")
params <- lapply(type, FUN = function(x){list(channel = x)})
reports <- tibble(output_file, params)
reports

apply(reports, MARGIN = 1, 
      FUN = function(x){
        render(input = "C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Rmd/ST558_project2_auto.Rmd",
               output_format = "github_document", 
               output_file = paste0("C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/", x[[1]]),
               params = x[[2]],
               output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble"))
      })
```
