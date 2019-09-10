##########################
# Load packages and data #
##########################

# Load packages
library("tidyverse")

# Load data (takes a few seconds)
load('scrapings.Rdata')


# How long is it?
n_athletes <- length(info_table) # 135584

# filter only the ones that I was able to scrape

info_table = info_table[1:50937]
results_table = results_table[1:50937]

save(ind_links, info_table, results_table, file="scrapings_filtered.Rdata")


load('scrapings.Rdata')
# names
infobox %>% lapply(function(x) grepl("Full name", x)) %>% unlist %>% sum # 100% complete
# The first column of the tibble contains the full, unedited name of each athlete
info <- lapply(infobox, function(x) strsplit(x[[1]], ": ")[[1]][2]) %>% unlist %>% tibble(Name = .)
info$Name %>% is.null %>% sum # check: no NULL entries
