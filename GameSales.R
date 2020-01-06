################################
# Create training set, validation set
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

# Set path to the directory that contains the dataset
path <- "."
filename <- "Video_Games_Sales_as_at_22_Dec_2016.csv"
fullpath <- file.path(path, filename)

sales <- read.csv(fullpath)

# The data set contains rows missing information; I decided to drop the corresponding rows 
#
# Note that this has the potential to introduce some bias in the analysis e.g. if there's a correlation between sales and missing data. 
# I leave it for future work to further explore the link between missing data and sales and potentially use other methods to account for missing data (e.g. imputation).
sales <- sales[complete.cases(sales), ]

# Convert the user scores to numeric values (from factor) to be able to perform a regression
sales$User_Score <- as.numeric(as.character(sales$User_Score))

# Publisher and Developer information are encoded as factors with many different levels. 
# This frequently causes problems with the regression, as the validation set can contain publishers or developers for which no coefficients have been calculated. 
# One solution would be to replace the prediction in this cases with the naive base prediction (mean sales of the training set). 
# I decided to instead assign all small publishers / developers for which calculated coefficients are unreliable anyways to a residual "Other" category.
sales <- sales %>% group_by(Publisher) %>% mutate(n_published = n())
levels(sales$Publisher) <- union(levels(sales$Publisher), "Other")
sales$Publisher[sales$n_published <= 5] = "Other"
sales <- ungroup(sales)

sales <- sales %>% group_by(Developer) %>% mutate(n_developed = n())
levels(sales$Developer) <- union(levels(sales$Developer), "Other")
sales$Developer[sales$n_developed <= 5] = "Other"
sales <- ungroup(sales)

# Columns for region specifc sales are not needed for this analysis, I am only interested in the overall sales
sales <- subset(sales, select = -c(NA_Sales, EU_Sales, JP_Sales, Other_Sales, n_published, n_developed))

#set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1)

# Assign the data randomly to a training and a validation set
test_index <- createDataPartition(y = sales$Global_Sales, times = 1, p = 0.1, list = FALSE)
train_set <- sales[-test_index,]
test_set <- sales[test_index,]


################################
# Naive prediction
################################


# Calculate the mean sales per video game, i.e. a naive prediction 
m <- mean(train_set$Global_Sales)

# Calculate the RMSE for the naive prediction using the mean sales as a benchmark
rmse_naive <- RMSE(test_set$Global_Sales, m)


################################
# Prediction using score $ count variables
################################

# Fit a linear regression using the critic scores only
fit_critic_score <- lm(Global_Sales ~ Critic_Score, data = train_set)
y_hat_critic_score <- predict(fit_critic_score, test_set)
rmse_critic_score <- RMSE(test_set$Global_Sales, y_hat_critic_score)

# Fit a exponential model using the critic scores only (i.e. transform Global_Sales and perform the regression)
fit_critic_score_exp <- lm(log(Global_Sales) ~ Critic_Score, data = train_set)
y_hat_critic_score_exp <- predict(fit_critic_score_exp, test_set)
rmse_critic_score_exp <- RMSE(test_set$Global_Sales, y_hat_critic_score_exp)

# Fit a linear regression using the user scores only 
# Note that this step required cleaning the data 
# (dropping rows that contain no or "tbd" values and convert the result to numeric values)
fit_user_score <- lm(Global_Sales ~ User_Score, data = train_set)
y_hat_user_score <- predict(fit_user_score, test_set)
rmse_user_score <- RMSE(test_set$Global_Sales, y_hat_user_score)

# Fit a linear regression using number of critics 
fit_critic_count <- lm(Global_Sales ~ Critic_Count, data = train_set)
y_hat_critic_count <- predict(fit_critic_count, test_set)
rmse_critic_count <- RMSE(test_set$Global_Sales, y_hat_critic_count)

# Fit a linear regression using number of users 
fit_user_count <- lm(Global_Sales ~ User_Count, data = train_set)
y_hat_user_count <- predict(fit_user_count, test_set)
rmse_user_count <- RMSE(test_set$Global_Sales, y_hat_user_count)

# Fit a linear regression using the platform only
fit_platform <- lm(Global_Sales ~ Platform, data = train_set)
y_hat_platform <- predict(fit_platform, test_set)
rmse_platform <- RMSE(test_set$Global_Sales, y_hat_platform)

# Fit a linear regression using the publisher only
fit_publisher <- lm(Global_Sales ~ Publisher, data = train_set)
y_hat_publisher <- predict(fit_publisher, test_set)
rmse_publisher <- RMSE(test_set$Global_Sales, y_hat_publisher)

# Fit a linear regression using the developer only
fit_developer <- lm(Global_Sales ~ Developer, data = train_set)
y_hat_developer <- predict(fit_developer, test_set)
rmse_developer <- RMSE(test_set$Global_Sales, y_hat_developer)

# Fit a multivariate linear regression combining the variables that seem to have the most explanatory power: critic score, critic count, publisher & platform
fit_mult<- lm(Global_Sales ~ Critic_Score + Critic_Count + Publisher + Platform, data = train_set)
y_hat_mult <- predict(fit_mult, test_set)
rmse_mult <- RMSE(test_set$Global_Sales, y_hat_mult)