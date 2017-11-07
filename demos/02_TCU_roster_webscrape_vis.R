


library(rvest)     # for web scraping
library(ggplot2)   # for visualizations
library(rbokeh)    # interactive javascript visualizations
library(plotly)    # more interactive javascript visualizations


# don't worry about this line of code for now
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()



# assign the full url to this variable
target_url <- "http://www.gofrogs.com/sports/m-footbl/mtt/tcu-m-footbl-mtt.html"

# read the html of the website (some sites use javascript/AJAX to make this more difficult)
site_html  <- read_html(target_url)

# isolate the "sortable_roster" id element -- right click and inspect element in chrome to view HTML
table_html <- html_node(site_html, xpath='//*[@id="sortable_roster"]')

# conert this html table into a data.frame in R
roster_df  <- html_table(table_html)




# If the above code fails, then uncomment this line of code and run it
# roster_df <- readRDS('data/roster_df.rds') 


# Now let's inspect the data and build some visualizations

# print the first few rows of the roster data frame to make sure it's correct
print(head(roster_df))


# inspect column names
names(roster_df)


# fix names
names(roster_df) <- gsub(pattern = "\\.", replacement = "", x = tolower(names(roster_df)))
names(roster_df)



# color palette -- google "hex color picker":
col1 <- "#4542f4"




# histogram of weight -- 30 bins
ggplot(data=roster_df, aes(x=wt)) +
    geom_histogram(fill=col1, alpha=0.7, bins=30) +
    ggtitle("Histogram of Player Weight -- bins=30") +
    xlab("Weight") + ylab("Count of Players in This Bin") +
    theme_bw(base_size=16)




# histogram of weight -- 20 bins
ggplot(data=roster_df, aes(x=wt)) +
    geom_histogram(fill=col1, alpha=0.7, bins=20) +
    ggtitle("Histogram of Player Weight -- bins=20") +
    xlab("Weight") + ylab("Count of Players in This Bin") +
    theme_bw(base_size=16)





# scatter plot of height and weight of players
ggplot(data=roster_df, aes(x=height, y=wt)) +
    geom_point(alpha=0.4, color=col1, size=2) +
    geom_smooth() +
    ggtitle("Height vs Weight of TCU 2017 Roster") +
    geom_rug() +
    ylab("Weight") + xlab("Height") +
    theme_bw(base_size=16)




# box plot of player weight by classification
table(roster_df$class)
roster_df$class <- factor(roster_df$class, levels=c("RS FR", "FR", "SO", "JR", "SR"))

ggplot(data=roster_df, aes(x=class, y=wt)) +
    geom_boxplot(fill=col1, alpha=0.6) +
    theme_bw(base_size=16) +
    xlab("Year/Class") + 
    ylab("Weight") +
    ggtitle("Weight by Year/Class of Player")




# rbokeh -- interactive height / weight javascript visualizations
figure(width=600, height=600) %>%
    ly_points(data=roster_df, x=height, y=wt,
              color = class, glyph = class, hover = list(name, hometown))



# dplyr to group/summarise data
# plotly to build 3d interactive histogram (technically 2d... this is a contrived example)
roster_grp <- roster_df %>%
    group_by(height, wt) %>%
    summarise(count = n()) %>% ungroup()

all_area <- expand.grid(63:83, 145:350)
names(all_area) <- c("height", "wt")
all_area <- merge(all_area, roster_grp, by=c("height", "wt"), all.x=T, all.y=F)
all_area[is.na(all_area)] <- 0

all_area_spread <- tidyr::spread(all_area, wt, count)
row.names(all_area_spread) <- all_area_spread$height
all_area_spread$height <- NULL

plot_ly(z = as.matrix(all_area_spread)) %>% add_surface()




# a more proper way to use the 3d surface plot
dim(volcano)
plot_ly(z = ~volcano) %>% add_surface()


