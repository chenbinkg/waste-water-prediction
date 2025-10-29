# Load necessary libraries
library(data.table)
library(ggplot2)
library(lubridate)
library(forecast)
library(xgboost)
library(caret)
library(Metrics)

# --- Step 1: Generate Example Data ---
set.seed(42)
hours <- seq.POSIXt(from=as.POSIXct("2023-01-01 00:00"), by="hour", length.out=24*90) # 90 days of hourly data
rainfall <- rgamma(length(hours), shape=1.5, scale=2) * (runif(length(hours)) < 0.1) # Sparse rainfall
temperature <- 10 + 10*sin(2*pi*yday(hours)/365) + rnorm(length(hours), 0, 2)
wastewater_inflow <- 500 + 50*sin(2*pi*hour(hours)/24) + 5*rainfall + 2*temperature + rnorm(length(hours), 0, 10)

df <- data.table(Datetime=hours, Rainfall=rainfall, Temperature=temperature, Inflow=wastewater_inflow)

path.in<-"O:\\WWL25503\\Working\\InflowModelling\\input\\"
path.out<-"O:\\WWL25503\\Working\\InflowModelling\\output\\"

wwf<-read.csv(file=paste0(path.in,"WWF_hourly_flow_CUR_2023.csv"))
rain<-read.csv(file=paste0(path.in,"Hutt Model Rainfall PinehavenBirch2008-2017.csv"))


wwf$Time <- as.POSIXct(wwf$Time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
rain$P_DATETIME <- as.POSIXct(rain$P_DATETIME, format = "%d/%m/%Y %H:%M:%S", tz = "UTC")

# Create an "hourly" timestamp
rain$Hour <- format(rain$P_DATETIME, "%Y-%m-%d %H:00:00")
rain$Hour <- as.POSIXct(rain$Hour, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

# Aggregate by hour, summing rainfall values
rain_hourly <- aggregate(. ~ Hour, data = rain[, c("Hour", "Birch.Lane..mm.hr.", "Pinehaven..mm.hr.")], sum, na.rm = TRUE)


wastewater_inflow<-wwf$SilverstreamAllInflow
#plot(wastewater_inflow)
rainfall<- rain_hourly$Pinehaven..mm.hr.  
hours<-rain_hourly$Hour
df <- data.table(Datetime=hours, Rainfall=rainfall, Inflow=wastewater_inflow)

# --- Step 2: Feature Engineering ---
df[, `:=`(
  Rainfall_Lag1 = shift(Rainfall, 1, type="lag"),
  Rainfall_Lag3 = shift(Rainfall, 3, type="lag"),
 # Temp_Lag1 = shift(Temperature, 1, type="lag"),
  Hour = hour(Datetime),
  DayofWeek = wday(Datetime),
  RollingRainfall_6h = frollsum(Rainfall, 6, align="right"),
  RollingRainfall_24h = frollsum(Rainfall, 24, align="right"),
  sin_hour = sin(2 * pi * hour(Datetime) / 24),
  cos_hour = cos(2 * pi * hour(Datetime) / 24)
)]

df <- na.omit(df)  # Remove NA rows due to lags

# --- Step 3: Train-Test Split ---
train_size <- floor(0.8 * nrow(df))
train <- df[1:train_size]
test <- df[(train_size+1):.N]

X_train <- as.matrix(train[, !c("Datetime", "Inflow"), with=FALSE])
y_train <- train$Inflow
X_test <- as.matrix(test[, !c("Datetime", "Inflow"), with=FALSE])
y_test <- test$Inflow

# --- Step 4: Statistical Model - ARIMAX ---
arimax_model <- auto.arima(y_train, xreg=X_train)
arimax_pred <- forecast(arimax_model, xreg=X_test)$mean

# --- Step 5: Machine Learning Model - XGBoost ---
xgb_model <- xgboost(data=X_train, label=y_train, nrounds=100, objective="reg:squarederror", max_depth=5, eta=0.1, verbose=0)
xgb_pred <- predict(xgb_model, X_test)

# --- Step 6: Hybrid Model (ARIMAX Residuals + XGBoost) ---
residuals_train <- y_train - fitted(arimax_model)
xgb_model_residuals <- xgboost(data=X_train, label=residuals_train, nrounds=100, objective="reg:squarederror", max_depth=5, eta=0.1, verbose=0)
hybrid_pred <- arimax_pred + predict(xgb_model_residuals, X_test)

# --- Step 7: Model Evaluation ---
eval_metrics <- function(y_true, y_pred, model_name) {
  rmse_val <- rmse(y_true, y_pred)
  mae_val <- mae(y_true, y_pred)
  r2_val <- cor(y_true, y_pred)^2
  return(data.table(Model=model_name, RMSE=rmse_val, MAE=mae_val, R2=r2_val))
}

results <- rbind(
  eval_metrics(y_test, arimax_pred, "ARIMAX"),
  eval_metrics(y_test, xgb_pred, "XGBoost"),
  eval_metrics(y_test, hybrid_pred, "Hybrid (ARIMAX + XGBoost)")
)

print(results)

# --- Step 8: Visualization ---
df_pred <- data.table(Datetime=test$Datetime, Actual=y_test, ARIMAX=arimax_pred, XGBoost=xgb_pred, Hybrid=hybrid_pred)

ggplot(df_pred, aes(x=Datetime)) +
  geom_line(aes(y=Actual, color="Actual"), size=1) +
  geom_line(aes(y=ARIMAX, color="ARIMAX"), linetype="dashed") +
  geom_line(aes(y=XGBoost, color="XGBoost"), linetype="dotted") +
  geom_line(aes(y=Hybrid, color="Hybrid"), linetype="twodash") +
  labs(title="Hourly Wastewater Inflow Predictions", y="Inflow", x="Time") +
  scale_color_manual(values=c("blue", "red", "green", "purple")) +
  theme_minimal()


################

# Load necessary libraries
library(data.table)
library(mgcv)           # For GAM model
library(ggplot2)
library(lubridate)
library(forecast)       # ARIMAX
library(xgboost)        # XGBoost
library(caret)          # Data preprocessing
#library(SWMMR)          # SWMM Hydrological Model
library(Metrics)        # RMSE, MAE

# --- Step 1: Generate Example Data ---
set.seed(42)
# hours <- seq.POSIXt(from=as.POSIXct("2023-01-01 00:00"), by="hour", length.out=24*90) # 90 days
# rainfall <- rgamma(length(hours), shape=1.5, scale=2) * (runif(length(hours)) < 0.1) # Sparse rain
# temperature <- 10 + 10*sin(2*pi*yday(hours)/365) + rnorm(length(hours), 0, 2)
# wastewater_inflow <- 500 + 50*sin(2*pi*hour(hours)/24) + 5*rainfall + 2*temperature + rnorm(length(hours), 0, 10)
# 
# df <- data.table(Datetime=hours, Rainfall=rainfall, Temperature=temperature, Inflow=wastewater_inflow)
# 
# # --- Step 2: Feature Engineering ---
# df[, `:=`(
#   Rainfall_Lag1 = shift(Rainfall, 1, type="lag"),
#   Rainfall_Lag3 = shift(Rainfall, 3, type="lag"),
#   Temp_Lag1 = shift(Temperature, 1, type="lag"),
#   Hour = hour(Datetime),
#   DayofWeek = wday(Datetime),
#   RollingRainfall_6h = frollsum(Rainfall, 6, align="right"),
#   RollingRainfall_24h = frollsum(Rainfall, 24, align="right"),
#   sin_hour = sin(2 * pi * hour(Datetime) / 24),
#   cos_hour = cos(2 * pi * hour(Datetime) / 24)
# )]
# df <- na.omit(df)  # Remove NA rows due to lags

# --- Step 3: Train-Test Split ---
train_size <- floor(0.9 * nrow(df))
train <- df[1:train_size]
test <- df[(train_size+1):.N]

X_train <- as.matrix(train[, !c("Datetime", "Inflow"), with=FALSE])
y_train <- train$Inflow
X_test <- as.matrix(test[, !c("Datetime", "Inflow"), with=FALSE])
y_test <- test$Inflow

# --- Step 4: Statistical Model - ARIMAX ---
arimax_model <- auto.arima(y_train, xreg=X_train)
arimax_pred <- forecast(arimax_model, xreg=X_test)$mean

# --- Step 5: Machine Learning Model - XGBoost ---
xgb_model <- xgboost(data=X_train, label=y_train, nrounds=100, objective="reg:squarederror", max_depth=5, eta=0.1, verbose=0)
xgb_pred <- predict(xgb_model, X_test)

# --- Step 6: Generalized Additive Model (GAM) ---
gam_model <- gam(Inflow ~ s(Rainfall)  + s(Hour, bs="cc") + s(DayofWeek, bs="re"), data=train)
gam_pred <- predict(gam_model, test)

# --- Step 7: Hydrological Model (SWMM) - Simulated Inflow ---
# Load SWMM model and run simulation (modify paths accordingly)
# swmm_file <- "your_swmm_model.inp" # Replace with actual SWMM model file
# swmm_run(swmm_file)
# swmm_results <- swmm_out_read("your_swmm_model.out") # Extract results
swmm_pred <- gam_pred # - swmm_results$subcatchment[, "TotalInflow"]

# Ensure SWMM output aligns with test data timestamps
swmm_pred <- swmm_pred[1:length(y_test)]

# --- Step 8: Hybrid Model (ARIMAX Residuals + XGBoost) ---
residuals_train <- y_train - fitted(arimax_model)
xgb_model_residuals <- xgboost(data=X_train, label=residuals_train, nrounds=100, objective="reg:squarederror", max_depth=5, eta=0.1, verbose=0)
hybrid_pred <- arimax_pred + predict(xgb_model_residuals, X_test)

# --- Step 9: Model Evaluation ---
eval_metrics <- function(y_true, y_pred, model_name) {
  rmse_val <- rmse(y_true, y_pred)
  mae_val <- mae(y_true, y_pred)
  r2_val <- cor(y_true, y_pred)^2
  return(data.table(Model=model_name, RMSE=rmse_val, MAE=mae_val, R2=r2_val))
}

results <- rbind(
  eval_metrics(y_test, arimax_pred, "ARIMAX"),
  eval_metrics(y_test, xgb_pred, "XGBoost"),
  eval_metrics(y_test, gam_pred, "GAM"),
  eval_metrics(y_test, swmm_pred, "SWMM"),
  eval_metrics(y_test, hybrid_pred, "Hybrid (ARIMAX + XGBoost)")
)

print(results)

# --- Step 10: Visualization ---
df_pred <- data.table(Datetime=test$Datetime, Actual=y_test, ARIMAX=arimax_pred, XGBoost=xgb_pred, GAM=gam_pred, SWMM=swmm_pred, Hybrid=hybrid_pred)

ggplot(df_pred, aes(x=Datetime)) +
  geom_line(aes(y=Actual, color="Actual"), size=1) +
  geom_line(aes(y=ARIMAX, color="ARIMAX"), linetype="dashed") +
  geom_line(aes(y=XGBoost, color="XGBoost"), linetype="dotted") +
  geom_line(aes(y=GAM, color="GAM"), linetype="twodash") +
  geom_line(aes(y=SWMM, color="SWMM"), linetype="dotdash") +
  geom_line(aes(y=Hybrid, color="Hybrid"), linetype="longdash") +
  labs(title="Hourly Wastewater Inflow Predictions", y="Inflow", x="Time") +
  scale_color_manual(values=c("blue", "red", "green", "orange", "purple", "black")) +
  theme_minimal()


library(data.table)
library(ggplot2)
library(lubridate)

# Create year column
df_pred$Year <- year(df_pred$Datetime)

# Plot with one subplot per year
ggplot(df_pred, aes(x = Datetime)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = ARIMAX, color = "ARIMAX"), linetype = "dashed") +
  geom_line(aes(y = XGBoost, color = "XGBoost"), linetype = "dotted") +
  geom_line(aes(y = GAM, color = "GAM"), linetype = "twodash") +
  geom_line(aes(y = SWMM, color = "SWMM"), linetype = "dotdash") +
  geom_line(aes(y = Hybrid, color = "Hybrid"), linetype = "longdash") +
  labs(
    title = "Hourly Wastewater Inflow Predictions (Faceted by Year)",
    y = "Inflow",
    x = "Time"
  ) +
  scale_color_manual(values = c("Actual" = "blue", "ARIMAX" = "red", "XGBoost" = "green", 
                                "GAM" = "orange", "SWMM" = "purple", "Hybrid" = "black")) +
  theme_minimal() +
  facet_wrap(~Year, scales = "free_x", ncol = 1)


library(data.table)
library(ggplot2)
library(lubridate)

# Define NZ season classification
get_season <- function(date) {
  m <- month(date)
  ifelse(m %in% c(12, 1, 2), "Summer",
         ifelse(m %in% c(3, 4, 5), "Autumn",
                ifelse(m %in% c(6, 7, 8), "Winter", "Spring")))
}

# Add Year and Season
df_pred$Year <- year(df_pred$Datetime)
df_pred$Season <- factor(get_season(df_pred$Datetime), levels = c("Summer", "Autumn", "Winter", "Spring"))

# Plot by year and season
ggplot(df_pred, aes(x = Datetime)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = ARIMAX, color = "ARIMAX"), linetype = "dashed") +
  geom_line(aes(y = XGBoost, color = "XGBoost"), linetype = "dotted") +
  geom_line(aes(y = GAM, color = "GAM"), linetype = "twodash") +
  geom_line(aes(y = SWMM, color = "SWMM"), linetype = "dotdash") +
  geom_line(aes(y = Hybrid, color = "Hybrid"), linetype = "longdash") +
  labs(
    title = "Seasonal Hourly Wastewater Inflow Predictions by Year",
    y = "Inflow",
    x = "Time"
  ) +
  scale_color_manual(values = c("Actual" = "blue", "ARIMAX" = "red", "XGBoost" = "green", 
                                "GAM" = "orange", "SWMM" = "purple", "Hybrid" = "black")) +
  theme_minimal() +
  facet_grid(Year ~ Season, scales = "free_x")





library(data.table)
library(ggplot2)
library(lubridate)
library(dplyr)

# --- Make sure df_pred exists with required columns ---
# Add Season and Year columns
df_pred <- df_pred %>%
  mutate(
    Year = year(Datetime),
    Month = month(Datetime),
    Season = case_when(
      Month %in% c(12, 1, 2) ~ "Summer",
      Month %in% c(3, 4, 5)  ~ "Autumn",
      Month %in% c(6, 7, 8)  ~ "Winter",
      Month %in% c(9, 10, 11) ~ "Spring"
    ),
    Season = factor(Season, levels = c("Summer", "Autumn", "Winter", "Spring"))
  )

# Unique years to loop through
unique_years <- unique(df_pred$Year)

# --- Plot for each year ---
for (yr in unique_years[2]) {
  df_year <- df_pred %>% filter(Year == yr)
  
  p <- ggplot(df_year, aes(x = Datetime)) +
    geom_line(aes(y = Actual, color = "Actual"), size = 1) +
    geom_line(aes(y = ARIMAX, color = "ARIMAX"), linetype = "dashed") +
    geom_line(aes(y = XGBoost, color = "XGBoost"), linetype = "dotted") +
    geom_line(aes(y = GAM, color = "GAM"), linetype = "twodash") +
    geom_line(aes(y = SWMM, color = "SWMM"), linetype = "dotdash") +
    geom_line(aes(y = Hybrid, color = "Hybrid"), linetype = "longdash") +
    labs(
      title = paste("Seasonal Inflow for Year", yr),
      x = "Datetime",
      y = "Inflow"
    ) +
    scale_color_manual(values = c(
      "Actual" = "blue",
      "ARIMAX" = "red",
      "XGBoost" = "green",
      "GAM" = "orange",
      "SWMM" = "purple",
      "Hybrid" = "black"
    )) +
    theme_minimal() +
    facet_wrap(~ Season, nrow = 2, ncol = 2, scales = "free_x")
  
  print(p)
}


library(dplyr)
library(ggplot2)

df_summary <- df_pred %>%
  mutate(
    Year = year(Datetime),
    Month = month(Datetime),
    Season = case_when(
      Month %in% c(12, 1, 2) ~ "Summer",
      Month %in% c(3, 4, 5)  ~ "Autumn",
      Month %in% c(6, 7, 8)  ~ "Winter",
      Month %in% c(9, 10, 11) ~ "Spring"
    )
  ) %>%
  group_by(Year, Season) %>%
  summarise(across(c(Actual, ARIMAX, XGBoost, GAM, SWMM, Hybrid), mean, na.rm = TRUE), .groups = "drop") %>%
  tidyr::pivot_longer(-c(Year, Season), names_to = "Model", values_to = "AvgInflow")

ggplot(df_summary, aes(x = Season, y = factor(Year), fill = AvgInflow)) +
  geom_tile(color = "white") +
  facet_wrap(~ Model, ncol = 3) +
  scale_fill_viridis_c() +
  labs(
    title = "Average Inflow by Season and Year per Model",
    x = "Season", y = "Year"
  ) +
  theme_minimal()
####################################
ggplot(df_pred, aes(x = Datetime)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Hybrid, color = "Hybrid"), linetype = "longdash") +
  geom_line(aes(y = ARIMAX, color = "ARIMAX"), linetype = "dashed") +
  geom_line(aes(y = GAM, color = "GAM"), linetype = "twodash") +
  geom_line(aes(y = XGBoost, color = "XGBoost"), linetype = "dotted") +
  geom_line(aes(y = SWMM, color = "SWMM"), linetype = "dotdash") +
  facet_wrap(~Year, scales = "free_x", ncol = 2) +
  scale_color_manual(values = c(
    "Actual" = "black", "Hybrid" = "purple", "ARIMAX" = "red",
    "GAM" = "orange", "XGBoost" = "green", "SWMM" = "blue"
  )) +
  labs(
    title = "Model Predictions vs Actual per Year",
    y = "Inflow", x = "Time"
  ) +
  theme_minimal()

#####################

df_pred$Month <- month(df_pred$Datetime, label = TRUE)

df_melt <- df_pred %>%
  group_by(Month) %>%
  summarise(across(c(Actual, Hybrid, ARIMAX, XGBoost, GAM, SWMM), mean, na.rm = TRUE)) %>%
  tidyr::pivot_longer(-Month, names_to = "Model", values_to = "MeanInflow")

ggplot(df_melt, aes(x = Month, y = MeanInflow, group = Model, color = Model)) +
  geom_line(size = 1) +
  coord_polar() +
  theme_minimal() +
  labs(title = "Average Monthly Inflow by Model", y = "Inflow")



###################

library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)

# Step 1: Define summer months and inflow threshold
summer_data <- df_pred %>%
  filter(month(Datetime) %in% c( 7)) %>%
  mutate(Year = year(Datetime))

# Optional: Define threshold as top 10% inflow
inflow_threshold <- quantile(summer_data$Actual, 0.90, na.rm = TRUE)

# Step 2: Create a long-format data frame for ggplot
summer_long <- summer_data %>%
  select(Datetime, Year, Actual, ARIMAX, XGBoost, GAM, SWMM, Hybrid) %>%
  pivot_longer(-c(Datetime, Year), names_to = "Model", values_to = "Inflow") %>%
  mutate(HighInflow = Inflow > inflow_threshold)

# Step 3: Plot
ggplot(summer_long, aes(x = Datetime, y = Inflow, color = Model)) +
  geom_line(alpha = 0.6) +
  #geom_point(data = filter(summer_long, HighInflow), aes(x = Datetime, y = Inflow), size = 1.5, shape = 21, fill = "yellow", color = "black", stroke = 0.3) +
  facet_wrap(~Year, scales = "free_x", ncol = 2) +
  labs(
    title = "Summer Season: High Inflow Events Highlighted",
    subtitle = paste("Inflow > ", round(inflow_threshold, 2), "highlighted in yellow"),
    x = "Date", y = "Inflow"
  ) +
  scale_color_manual(values = c(
    "Actual" = "black", "ARIMAX" = "red", "XGBoost" = "green",
    "GAM" = "orange", "SWMM" = "blue", "Hybrid" = "purple"
  )) +
  theme_minimal() +
  theme(legend.position = "bottom")


#*****************************
#*
evaluate_predictions <- function(df_pred,
                                 actual_col = "Actual",
                                 model_cols = "XGBoost",
                                 time_col = "Datetime",
                                 output_csv = "model_metrics.csv",
                                 export = TRUE,
                                 show_taylor = TRUE,
                                 show_radar = TRUE) {
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(tidyr)
  library(Metrics)
  library(openair)  # For Taylor Diagram
  library(fmsb)     # For Radar Chart
  
  if (is.null(model_cols)) {
    model_cols <- setdiff(colnames(df_pred), c(actual_col, time_col))
  }
  
  # --- 1. Compute Metrics ---
  get_metrics <- function(actual, predicted) {
    data.frame(
      RMSE = rmse(actual, predicted),
      MAE = mae(actual, predicted),
      R2 = cor(actual, predicted, use = "complete.obs")^2,
      NSE = 1 - sum((actual - predicted)^2, na.rm = TRUE) / sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE),
      Bias = mean(predicted - actual, na.rm = TRUE),
      Correlation = cor(actual, predicted, use = "complete.obs")
    )
  }
  
  metrics_table <- do.call(rbind, lapply(model_cols, function(model) {
    get_metrics(df_pred[[actual_col]], df_pred[[model]])
  }))
  rownames(metrics_table) <- model_cols
  
  if (export) {
    write.csv(metrics_table, output_csv, row.names = TRUE)
    message(paste("Metrics exported to:", output_csv))
  }
  
  # --- 2. Scatter Plots ---
  scatter_plots <- lapply(model_cols, function(model) {
    ggplot(df_pred, aes_string(x = actual_col, y = model)) +
      geom_point(alpha = 0.5, color = "steelblue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      labs(title = paste(model, "vs Actual"), x = "Observed", y = "Predicted") +
      theme_minimal()
  })
  scatter_combined <- wrap_plots(scatter_plots, ncol = 2)
  
  # --- 3. Error Distribution ---
  error_data <- df_pred %>%
    select(all_of(c(actual_col, model_cols))) %>%
    pivot_longer(cols = all_of(model_cols), names_to = "Model", values_to = "Predicted") %>%
    mutate(Error = Predicted - .data[[actual_col]])
  
  error_plot <- ggplot(error_data, aes(x = Error, fill = Model)) +
    geom_histogram(bins = 60, alpha = 0.6, position = "identity") +
    facet_wrap(~Model, scales = "free") +
    theme_minimal() +
    labs(title = "Prediction Error Distribution", x = "Error", y = "Frequency")
  
  # --- 4. Time Series Overlay ---
  ts_long <- df_pred %>%
    select(all_of(c(time_col, actual_col, model_cols))) %>%
    pivot_longer(cols = all_of(c(actual_col, model_cols)), names_to = "Type", values_to = "Value")
  
  ts_plot <- ggplot(ts_long, aes_string(x = time_col, y = "Value", color = "Type")) +
    geom_line(alpha = 0.7) +
    labs(title = "Time Series: Observed vs Predicted", y = "Inflow", x = "Time") +
    theme_minimal()
  
  # --- 5. Taylor Diagram (optional) ---
  if (show_taylor) {
    taylor_df <- df_pred %>%
      select(all_of(c(actual_col, model_cols))) %>%
      rename(obs = !!actual_col)
    
    taylor_long <- taylor_df %>%
      pivot_longer(-obs, names_to = "model", values_to = "sim")
    
    taylor_plot <- TaylorDiagram(taylor_long, obs = "obs", mod = "sim", group = "model", main = "Taylor Diagram")
  } else {
    taylor_plot <- NULL
  }
  
  # --- 6. Radar Plot (optional) ---
  if (show_radar) {
    metrics_scaled <- as.data.frame(scale(metrics_table))
    radar_data <- rbind(
      rep(1.5, ncol(metrics_scaled)),  # Max scale
      rep(-1.5, ncol(metrics_scaled)), # Min scale
      metrics_scaled
    )
    radar_plot <- radarchart(radar_data,
                             axistype = 1,
                             pcol = rainbow(nrow(metrics_table)),
                             plty = 1,
                             plwd = 2,
                             cglcol = "grey", cglty = 1,
                             axislabcol = "grey",
                             title = "Scaled Model Performance")
    legend("topright", legend = rownames(metrics_table), col = rainbow(nrow(metrics_table)), lty = 1, lwd = 2, bty = "n")
  } else {
    radar_plot <- NULL
  }
  
  return(list(
    metrics = metrics_table,
    scatter_plots = scatter_combined,
    error_plot = error_plot,
    ts_plot = ts_plot,
    taylor_plot = taylor_plot,
    radar_plot = radar_plot
  ))
}

  


results <- evaluate_predictions(df_pred)

# View
results$metrics
results$scatter_plots
results$ts_plot
results$error_plot
results$taylor_plot
# radar_plot is drawn in the base plotting window



