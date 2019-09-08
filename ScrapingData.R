
library("RCurl")
library("XML")
library("tidyverse")

# getting the directories
rawHtml = getURL("https://www.sports-reference.com/olympics/athletes/")
parsedHtml = htmlParse(rawHtml, asText=TRUE)

# <td align="center" valign="middle"><a href="/olympics/athletes/tq/">Tq</a></td>
# this is how the links are structured
links = xpathSApply(parsedHtml, "//td//a", xmlGetAttr, 'href')

athlete_directory = paste('https://www.sports-reference.com', links, sep="")


# how many are there? 
#> head(athlete_directory)
#[1] "https://www.sportsreference.com/olympics/athletes/a/"  "https://www.sportsreference.com/olympics/athletes/aa/"
#[3] "https://www.sportsreference.com/olympics/athletes/ab/" "https://www.sportsreference.com/olympics/athletes/ac/"
#[5] "https://www.sportsreference.com/olympics/athletes/ad/" "https://www.sportsreference.com/olympics/athletes/ae/"
# > length(athlete_directory)
#[1] 453

# now the next step is to go through the links and then visit the individual athlete pages
ind_links = c()

system.time(
  for (i in 1:length(athlete_directory)) {
    new <- getURL(athlete_directory[i]) %>%
      htmlParse(asText=TRUE) %>%
      xpathSApply('//*[(@id = "page_content")]//a', xmlGetAttr, 'href') %>%
      paste('http://www.sports-reference.com/', ., sep="")
    # update vector of athlete pages
    ind_links <- c(ind_links, new) 
    # track progress in console
    print(i) 
    flush.console() # avoid output buffering
  }
) 

# user  system elapsed 
# 44.58    5.64  204.07 


# ind_links