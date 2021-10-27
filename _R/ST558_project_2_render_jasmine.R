###########################################################
# ST558 - Automating Project 2
# Jasmine Wang
# Due 10/31/2021
# 
# Render function to output .Rmd file for github
##########################################################3

rmarkdown::render("C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Rmd/ST558_project_2.Rmd", 
                  output_format = "github_document", 
                  output_file = "C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/generated_analysis.md", 
                  output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble")
)



# Knit with parameters, automating reports render function
library(rmarkdown)
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


#rmarkdown::render("C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Rmd/ST558_project2_auto.Rmd", 
#                  output_format = "github_document", 
#                  output_file = "C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/generated_analysis.md", 
#                  params = list(channel = ""),
#                  output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble")
#)
