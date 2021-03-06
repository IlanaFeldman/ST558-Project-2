###########################################################
# ST558 - Automating Project 2
# Jasmine Wang
# Due 10/31/2021
# 
# Render function to output .Rmd file for github
##########################################################3

##### To run 6 reports !!!!!!!!!######### Knit with parameters, automating reports render function
library(rmarkdown)
library(tidyverse)
type <- c("lifestyle", "entertainment", "bus", "socmed", "tech", "world") #, "bus", "socmed", "tech", "world"
output_file <- paste0(type, ".md")
params <- lapply(type, FUN = function(x){list(channel = x)})
reports <- tibble(output_file, params)

apply(reports, MARGIN = 1, 
      FUN = function(x){
        render(input = "ST558_project2_auto.Rmd",
               output_format = "github_document", 
               output_file = x[[1]],
               params = x[[2]],
               output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble"))
      })


