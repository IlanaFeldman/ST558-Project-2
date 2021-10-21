###########################################################
# ST558 - Automating Project 2
# Jasmine Wang
# Due 10/31/2021
# 
# Render function to output .Rmd file for github
##########################################################3


rmarkdown::render("C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Rmd/ST558_project_2.Rmd", 
                  output_format = "github_document", 
                  output_file = "C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/analysis.md", 
                  output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble")
)

