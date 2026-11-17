# LEGACY: used for the first Option A runs. gr_run.R reproduces the same
# protocol for all models and both forcing options and should be used instead.
# =============================================================================
# gr_calibrate.R - Calibrate GR4J or GR6J on the eligible catchments (Option A)
# Warm-up 1985-87 | calibration 1988-97 (NSE, Calibration_Michel) | test 1998-2008
# Periods follow Lees et al. (2021). Run with:  Rscript gr_calibrate.R GR6J
# =============================================================================
suppressPackageStartupMessages(library(airGR))

model <- commandArgs(trailingOnly = TRUE)[1]               # "GR4J" or "GR6J"
stopifnot(model %in% c("GR4J", "GR6J"))
FUN <- if (model == "GR4J") RunModel_GR4J else RunModel_GR6J

# --- Paths -------------------------------------------------------------------
root      <- "/content/drive/MyDrive/rainfall_runoff_benchmark"
daily_dir <- file.path(root, "data/camels_gb_v2/Catchment_Timeseries/hydro-meteorological/daily")
out_dir   <- file.path(root, "outputs/gr_models")
dir.create(file.path(out_dir, "sim"),    recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "params"), recursive = TRUE, showWarnings = FALSE)

gauges <- as.character(read.csv(file.path(root, "outputs/catchment_selection/eligible_catchments.csv"))$gauge_id)
files  <- list.files(daily_dir, pattern = "\\.csv$", full.names = TRUE)
names(files) <- sub(".*timeseries_([0-9]+)_.*", "\\1", basename(files))

# --- Periods -------------------------------------------------------------------
P_WARM  <- c("1985-01-01", "1987-12-31")
P_TRAIN <- c("1988-01-01", "1997-12-31")
P_TEST  <- c("1998-01-01", "2008-12-31")

# --- Metrics (missing observed days ignored) ---------------------------------
nse <- function(s, o) { k <- !is.na(s) & !is.na(o)
  1 - sum((s[k] - o[k])^2) / sum((o[k] - mean(o[k]))^2) }
kge <- function(s, o) { k <- !is.na(s) & !is.na(o)            # Gupta et al. (2009)
  r <- cor(s[k], o[k]); a <- sd(s[k]) / sd(o[k]); b <- mean(s[k]) / mean(o[k])
  1 - sqrt((r - 1)^2 + (a - 1)^2 + (b - 1)^2) }

# --- Loop over catchments, one at a time, printing progress ------------------
t_all <- Sys.time()
for (i in seq_along(gauges)) {
  gid <- gauges[i]
  par_file <- file.path(out_dir, "params", paste0(model, "_", gid, ".csv"))
  if (file.exists(par_file)) { cat(sprintf("[%3d/%d] %s %-7s already done - skipped\n",
                                           i, length(gauges), model, gid)); next }
  t0 <- Sys.time()
  res <- tryCatch({
    # read daily data from warm-up start to end of test period
    d <- read.csv(files[[gid]])
    d$date <- as.POSIXct(substr(d$date, 1, 10), tz = "UTC")
    d <- d[d$date >= as.POSIXct(P_WARM[1], tz = "UTC") &
           d$date <= as.POSIXct(P_TEST[2], tz = "UTC"), ]
    idx <- function(p) which(d$date >= as.POSIXct(p[1], tz = "UTC") &
                             d$date <= as.POSIXct(p[2], tz = "UTC"))
    i_warm <- idx(P_WARM); i_train <- idx(P_TRAIN); i_test <- idx(P_TEST)
    i_all  <- c(i_train, i_test)

    # inputs: CEH-GEAR rainfall and CHESS-PE PET (mm/day)
    inputs <- CreateInputsModel(FUN_MOD = FUN, DatesR = d$date,
                                Precip = d$precipitation_cehgear, PotEvap = d$pet_chess)

    # calibrate on 1988-97, maximising NSE
    run_cal  <- CreateRunOptions(FUN_MOD = FUN, InputsModel = inputs,
                                 IndPeriod_WarmUp = i_warm, IndPeriod_Run = i_train,
                                 verbose = FALSE)
    crit_cal <- CreateInputsCrit(FUN_CRIT = ErrorCrit_NSE, InputsModel = inputs,
                                 RunOptions = run_cal, Obs = d$discharge_spec[i_train])
    cal <- Calibration_Michel(InputsModel = inputs, RunOptions = run_cal,
                              InputsCrit = crit_cal,
                              CalibOptions = CreateCalibOptions(FUN_MOD = FUN),
                              FUN_MOD = FUN, verbose = FALSE)
    par <- cal$ParamFinalR

    # continuous run 1988-2008 with calibrated parameters
    run_all <- CreateRunOptions(FUN_MOD = FUN, InputsModel = inputs,
                                IndPeriod_WarmUp = i_warm, IndPeriod_Run = i_all,
                                verbose = FALSE)
    sim <- FUN(InputsModel = inputs, RunOptions = run_all, Param = par)$Qsim
    obs <- d$discharge_spec[i_all]
    tr  <- seq_along(i_train); te <- (length(i_train) + 1):length(i_all)

    # save simulated vs observed series
    write.csv(data.frame(date = format(d$date[i_all], "%Y-%m-%d"), q_obs = obs, q_sim = sim),
              file.path(out_dir, "sim", paste0(model, "_", gid, ".csv")), row.names = FALSE)

    out <- data.frame(gauge_id = gid, model = model,
                      nse_train = nse(sim[tr], obs[tr]), kge_train = kge(sim[tr], obs[tr]),
                      nse_test  = nse(sim[te], obs[te]), kge_test  = kge(sim[te], obs[te]))
    out[paste0("X", seq_along(par))] <- as.list(par)
    write.csv(out, par_file, row.names = FALSE)
    sprintf("test NSE = %.3f, KGE = %.3f", out$nse_test, out$kge_test)
  }, error = function(e) paste("ERROR:", conditionMessage(e)))

  cat(sprintf("[%3d/%d] %s %-7s %5.1f s  %s\n", i, length(gauges), model, gid,
              as.numeric(difftime(Sys.time(), t0, units = "secs")), res))
  flush.console()
}

# --- Combine all saved results (both models) into one table ------------------
all_res <- do.call(rbind, lapply(list.files(file.path(out_dir, "params"), full.names = TRUE),
                                 function(f) { x <- read.csv(f)
                                   x[setdiff(paste0("X", 1:6), names(x))] <- NA; x }))
write.csv(all_res, file.path(out_dir, "gr_results_optionA.csv"), row.names = FALSE)
for (m in unique(all_res$model)) {
  s <- all_res[all_res$model == m, ]
  cat(sprintf("\n%s: %d catchments | median test NSE = %.3f, KGE = %.3f",
              m, nrow(s), median(s$nse_test, na.rm = TRUE), median(s$kge_test, na.rm = TRUE)))
}
cat(sprintf("\nTotal time: %.1f min\n", as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
