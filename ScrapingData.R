
library("RCurl")
library("XML")
library("tidyverse")
library('curl')

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



pool = new_pool()
ind_links = c()


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



# user  system elapsed 
# 44.58    5.64  204.07 

# initializing the results variable
info_table = vector("list", length(ind_links))
results_table = vector("list", length(ind_links))


pool = new_pool()

for( i in 1:1){
  curl_fetch_multi(ind_links[i], function(x){
    html <- try(getURL(x, .opts=curlOptions(followlocation=TRUE), .encoding="UTF-8"), silent=TRUE)
    if(class(html) == "try-error") {
      Sys.sleep(5)
      html <- getURL(res, .opts=curlOptions(followlocation=TRUE))
    }
    #res = rawToChar(x$content)
    
    html <- htmlParse(html, asText=TRUE, encoding="utf-8")
    
    res = xpathSApply(html, '//*[@id="info_box"]/p', xmlValue) %>%
      strsplit('\n') %>% .[[1]]
    
    assign("info_table[[i]]", res, envir = .GlobalEnv) 
    print(res)
    print(i)
    table_res = readHTMLTable(html) %>% .$results
    print(table_res)
    # save 'infobox'
      # save 'results table'
    results_table[[i]] <<- readHTMLTable(html) %>% .$results
  }, pool = pool)
}


system.time(
  out <- multi_run(pool = pool)
)



save(ind_links, info_table, results_table, file="scrapings.Rdata")

#uris = c("http://www.omegahat.org/index.html", "http://www.omegahat.org/RecentActivities.html")

#uris = c("http://www.omegahat.net/RCurl/index.html",
#         "http://www.omegahat.net/RCurl/philosophy.xml")
#txt = getURIAsynchronous(uris)
