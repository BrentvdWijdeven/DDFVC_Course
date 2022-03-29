### test with sample dataset that is available in base R
# https://nwfsc-timeseries.github.io/atsa-labs/sec-tslab-correlation-within-and-among-time-series.html

lynx
## get the matching years of sunspot data
suns <- ts.intersect(lynx, sunspot.year)[, "sunspot.year"]
## get the matching lynx data
lynx <- ts.intersect(lynx, sunspot.year)[, "lynx"]

## plot time series
plot(cbind(suns, lynx), yax.flip = TRUE)

## CCF of sunspots and lynx
ccf(suns, log(lynx), ylab = "Cross-correlation")


######################
### with sample of dataset of course
merged_df = read.csv('data/merged_df.csv')

?ts
(zero <- ts(merged_df$sensor_0, start=1, end=length(merged_df$X), frequency=1))
(one <- ts(merged_df$sensor_1, start=1, end=length(merged_df$X), frequency=1))

?ccf
res <- ccf(zero, one, lag.max=100, type="correlation", ci=.95)
