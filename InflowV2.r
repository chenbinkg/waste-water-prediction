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


wastewater_inflow<-wwf$SilverstreamAllInflow    ###  y 
#plot(wastewater_inflow)
rainfall<- rain_hourly$Pinehaven..mm.hr.  
hours<-rain_hourly$Hour
df <- data.table(Datetime=hours, Rainfall=rainfall, Inflow=wastewater_inflow)

# # --- Step 2: Feature Engineering ---
# df[, `:=`(
#   Rainfall_Lag1 = shift(Rainfall, 1, type="lag"),
#   Rainfall_Lag3 = shift(Rainfall, 3, type="lag"),
#   # Temp_Lag1 = shift(Temperature, 1, type="lag"),
#   Hour = hour(Datetime),
#   DayofWeek = wday(Datetime),
#   RollingRainfall_3h = frollsum(Rainfall, 3, align="right"),
#   RollingRainfall_6h = frollsum(Rainfall, 6, align="right"),
#   RollingRainfall_6h = frollsum(Rainfall, 12, align="right"),
#   RollingRainfall_24h = frollsum(Rainfall, 24, align="right"),
#   RollingRainfall_6h = frollsum(Rainfall, 48, align="right"),
#   sin_hour = sin(2 * pi * hour(Datetime) / 24),
#   cos_hour = cos(2 * pi * hour(Datetime) / 24)
# )]

# Set decay factor for API (Antecedent Precipitation Index)
k_api <- 0.85  # Typical value between 0.8 and 0.95

# Initialize API and Soil Moisture as vectors
df[, API := 0]
df[, SoilMoisture := 0]

# Compute rolling sums and lags
df[, `:=`(
  Rainfall_Lag1 = shift(Rainfall, 1, type="lag"),
  Rainfall_Lag3 = shift(Rainfall, 3, type="lag"),
  Hour = hour(Datetime),
  DayofWeek = wday(Datetime),
  RollingRainfall_3h = frollsum(Rainfall, 3, align="right"),
  RollingRainfall_6h = frollsum(Rainfall, 6, align="right"),
  RollingRainfall_12h = frollsum(Rainfall, 12, align="right"),
  RollingRainfall_24h = frollsum(Rainfall, 24, align="right"),
  RollingRainfall_48h = frollsum(Rainfall, 48, align="right"),
  sin_hour = sin(2 * pi * hour(Datetime) / 24),
  cos_hour = cos(2 * pi * hour(Datetime) / 24)
)]

# Calculate Antecedent Precipitation Index (API)
for (i in 2:nrow(df)) {
  df$API[i] <- k_api * df$API[i - 1] + df$Rainfall[i - 1]
}

# Soil Moisture Index (very simplified: saturates at some threshold)
max_storage <- 100  # Arbitrary max bucket size (mm)
soil_decay <- 0.95  # Moisture decay rate
df$SoilMoisture[1] <- min(max_storage, df$Rainfall[1])  # Initial condition

for (i in 2:nrow(df)) {
  df$SoilMoisture[i] <- min(
    max_storage,
    df$SoilMoisture[i - 1] * soil_decay + df$Rainfall[i - 1]
  )
}



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



# Step 1: Define summer months and inflow threshold
df_pred_SL<- df_pred %>%
  filter(year(Datetime) %in% c(2017))

ggplot(df_pred_SL, aes(x=Datetime)) +
  geom_line(aes(y=Actual, color="Actual"), size=1) +
  geom_line(aes(y=ARIMAX, color="ARIMAX"), linetype="dashed") +
  geom_line(aes(y=XGBoost, color="XGBoost"), linetype="dotted") +
  geom_line(aes(y=Hybrid, color="Hybrid"), linetype="twodash") +
  labs(title="Hourly Wastewater Inflow Predictions", y="Inflow", x="Time") +
  scale_color_manual(values=c("blue", "red", "green", "purple")) +
  theme_minimal()
################################################################
# Step 1: Define summer months and inflow threshold
df_pred_SL<- df_pred %>%
  filter(month(Datetime) %in% c(12)) %>%
  mutate(Year = year(Datetime))
# Step 1: Define summer months and inflow threshold
df_pred_SL<- df_pred_SL %>%
  filter(year(Datetime) %in% c(2017))

ggplot(df_pred_SL, aes(x=Datetime)) +
  geom_line(aes(y=Actual, color="Actual"), size=1) +
  geom_line(aes(y=ARIMAX, color="ARIMAX"), linetype="dashed") +
  geom_line(aes(y=XGBoost, color="XGBoost"), linetype="dotted", size=1.2) +
  geom_line(aes(y=Hybrid, color="Hybrid"), linetype="twodash") +
  labs(title="Hourly Wastewater Inflow Predictions", y="Inflow", x="Time") +
  scale_color_manual(values=c("blue", "red", "green", "purple")) +
  theme_minimal()


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




