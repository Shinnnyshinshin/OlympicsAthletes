
library("RCurl")
library("XML")
library("tidyverse")
library('curl')

# getting the directories
rawHtml = getURL("https://www.sports-reference.com/olympics/athletes/")
parsedHtml = htmlParse(rawHtml, asText=TRUE)

# how does  //td//a work? 
# The text in <td> elements are regular and left-aligned by default.
# <a href="https://www.w3schools.com">Visit W3Schools.com!</a> just the hyper-link 

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

pool = new_pool(total_con = 5)
ind_links = c()

# curl fetch multi : 
# CURL : which you can pass it a PTML
for( i in 1:length(athlete_directory)){
  curl_fetch_multi(athlete_directory[i], function(x){
      res = rawToChar(x$content)
      temp = htmlParse(res,asText=TRUE)
      temp = xpathSApply(temp, '//*[(@id = "page_content")]//a', xmlGetAttr, 'href')
      temp = paste('http://www.sports-reference.com/', temp, sep="")
      ind_links <<- c(ind_links, temp)
    }, pool = pool)
}


system.time(
  out <- multi_run(pool = pool)
)

# these are all the individual links

# user  system elapsed 
# 44.58    5.64  204.07

# after parallelization
# user  system elapsed 
# 18.28    0.53   21.97 

ind_links = ind_links[1:4]

is.not.null <- function(x) !is.null(x)

# initializing the results variable
info_table = vector("list", length(ind_links))
results_table = vector("list", length(ind_links))
global_counter = 1
pool = new_pool(total_con = 2)

done <- function(res){
  html = rawToChar(res$content)
  html <- htmlParse(html, asText=TRUE, encoding="utf-8")
  # parse first
  initial_table = xpathSApply(html, '//*[@id="info_box"]/p', xmlValue)
  if (is.not.null(initial_table)){
    info_table[[global_counter]] <<- strsplit(initial_table, "\n") %>% .[[1]]
    results_table[[global_counter]] <<- readHTMLTable(html) %>% .$results
  }
  print(global_counter)
  global_counter <<- global_counter + 1
}
for (i in 1:length(ind_links)){
  hello = new_handle(url = ind_links[i])
  multi_add(hello, pool = pool, done=done)
}
system.time(multi_run(pool = pool))

save(ind_links, info_table, results_table, file="multi_scrapings.Rdata")

