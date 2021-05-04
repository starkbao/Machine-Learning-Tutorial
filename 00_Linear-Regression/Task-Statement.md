*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Linear Regression, please watch it on [YouTube](https://www.youtube.com/watch?v=fegAeph9UaA&ab_channel=Hung-yiLee).*
# Linear Regression
In statistics, linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.\
Linear regression plays an important role in the subfield of artificial intelligence known as machine learning. The linear regression algorithm is one of the fundamental supervised machine-learning algorithms due to its relative simplicity and well-known properties.
([from Wikipedia](https://en.wikipedia.org/wiki/Linear_regression#Machine_learning))

## Task Description
In the tutorial, we will build the linear regression **from scratch** (Yes, we are going to code the model on our own, not from any open source library!) to predict the amount of PM2.5.\
**In a nutshell, we will leverage 18 features (including PM2.5) in the first 9 hours to predict PM2.5 in the 10th hour.**

## Dataset
The dataset for training the linear regression model was collected from [Taiwan Air Quality Monitoring Network](https://airtw.epa.gov.tw/ENG/default.aspx).\
The data is the observation records of Fengyuan Station, which are divided into a train set and a test set. The train set is all the data in the first 20 days of each month at Fengyuan Station. The test set is sampled from the remaining days at Fengyuan Station. You can find them in separate `.csv` files.
- `train.csv`: Complete data from the first 20 days of each month.
- `test.csv`: Sample from the remaining data for 10 consecutive hours as one piece. All observation data of the first nine hours are regarded as features, and PM2.5 at the tenth hour is regarded as the target. A total of 240 unique test data are retrieved. Please predict the PM2.5 of these 240 test data based on features.\
The dataset contains 18 features: *AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR*.\

A snapshot of the data is as follows.

## Result
