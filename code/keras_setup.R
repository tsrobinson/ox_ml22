################################################################################
##                                                                            ##
##                              SICSS 2021 -- Oxford                          ##
##                                Machine Learning                            ##
##                                                                            ##
##                              Neural Networks in R                          ##
##                                    R Setup                                 ##
##                                                                            ##
################################################################################

## The below commands should work on R >= 4.0, hopefully even R >= 3.6
## If you have any troubles following these steps, let me know via Slack!

## Set up commands -- follow this to set up on your own machine
install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
install.packages("reticulate")
tensorflow::install_tensorflow()

## If you get the message "No non-system installation of Python could be found.
# "Would you like to download and install Miniconda?"
# Type 'Y' and press enter

install.packages("keras")
keras::install_keras()
install.packages("recipes")

# If you do not already have tidyverse installed:
install.packages("tidyverse")

## Double check everything's working:
library(tensorflow)
library(keras)
library(tidyverse)
library(recipes)
