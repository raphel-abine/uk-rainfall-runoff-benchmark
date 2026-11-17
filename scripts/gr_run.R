# =============================================================================
# gr_run.R - Calibrate GR4J / GR6J / CemaNeigeGR4J / CemaNeigeGR6J with forcing A or B
# Protocol: warm-up 1985-87 | calibration 1988-97 (NSE, Calibration_Michel)
#           | test 1998-2008 (periods follow Lees et al., 2021)
# Forcing A: CEH-GEAR P, CHESS-PE PET, CHESS-met T
# Forcing B: HadUK-Grid P, Hydro-PE PET, HadUK-Grid T
#
# Usage:  Rscript gr_run.R <model> <A|B> [project_root]
#   project_root must contain data/camels_gb_v2/ and
#   outputs/catchment_selection/eligible_catchments.csv
# =============================================================================
suppressPackageStartupMessages(library(airGR))

args    <- commandArgs(trailingOnly = TRUE)
model   <- args[1]
forcing <- args[2]
root    <- if (length(args) >= 3) args[3] else "/content/drive/MyDrive/rainfall_runoff_benchmark"
stopifnot(model %in% c("GR4J", "GR6J", "CemaNeigeGR4J", "CemaNeigeGR6J"), forcing %in% c("A", "B"))

FUN  <- switch(model, GR4J = RunModel_GR4J, GR6J = RunModel_GR6J,
               CemaNeigeGR4J = RunModel_CemaNeigeGR4J, CemaNeigeGR6J = RunModel_CemaNeigeGR6J)
snow <- grepl("CemaNeige", model)
FORCING <- list(A = c(P = "precipitation_cehgear", E = "pet_chess",   T = "temperature_chess"),
                B = c(P = "precipitation_haduk",   E = "pet_hydrope", T = "temperature_haduk"))
fv <- FORCING[[forcing]]

# --- Paths (Option B results kept in their own folder) -----------------------
daily_dir <- file.path(root, "data/camels_gb_v2/Catchment_Timeseries/hydro-meteorological/daily")
out_dir   <- file.path(root, if (forcing == "A") "outputs/gr_models" else "outputs/gr_models_optionB")
dir.create(file.path(out_dir, "sim"),    recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "params"), recursive = TRUE, showWarnings = FALSE)

gauges <- as.character(read.csv(file.path(root, "outputs/catchment_selection/eligible_catchments.csv"))$gauge_id)
files  <- list.files(daily_dir, pattern = "\\.csv$", full.names = TRUE)
names(files) <- sub(".*timeseries_([0-9]+)_.*", "\\1", basename(files))

# --- Hypsometry for CemaNeige: 101 quantiles interpolated from the five
#     elevation values CAMELS-GB provides (min, 10th, 50th, 90th, max) --------
topo <- read.csv(file.path(root, "data/camels_gb_v2/Catchment_Attributes/camels_gb_v2_topographic_attributes.csv"))
topo$gauge_id <- as.character(topo$gauge_id)
hypso_for <- function(gid) {
  r <- topo[topo$gauge_id == gid, ]
  approx(x = c(0, 10, 50, 90, 100),
         y = c(r$elev_min, r$elev_10, r$elev_50, r$elev_90, r$elev_max), xout = 0:100)$y
}

# --- Periods and metrics -----------------------------------------------------
P_WARM  <- c("1985-01-01", "1987-12-31")
P_TRAIN <- c("1988-01-01", "1997-12-31")
P_TEST  <- c("1998-01-01", "2008-12-31")
nse <- function(s, o) { k <- !is.na(s) & !is.na(o)
  1 - sum((s[k] - o[k])^2) / sum((o[k] - mean(o[k]))^2) }
kge <- function(s, o) { k <- !is.na(s) & !is.na(o)            # Gupta et al. (2009)
  r <- cor(s[k], o[k]); a <- sd(s[k]) / sd(o[k]); b <- mean(s[k]) / mean(o[k])
  1 - sqrt((r - 1)^2 + (a - 1)^2 + (b - 1)^2) }

cat(sprintf("=== %s | forcing %s: P=%s, E=%s%s ===\n", model, forcing, fv["P"], fv["E"],
            if (snow) paste0(", T=", fv["T"]) else ""))

# --- Loop over catchments (finished catchments are skipped) ------------------
t_all <- Sys.time()
for (i in seq_along(gauges)) {
  gid <- gauges[i]
  par_file <- file.path(out_dir, "params", paste0(model, "_", gid, ".csv"))
  if (file.exists(par_file)) { cat(sprintf("[%3d/%d] %s %-7s already done - skipped\n",
                                           i, length(gauges), model, gid)); next }
  t0 <- Sys.time()
  res <- tryCatch({
    d <- read.csv(files[[gid]])
    d$date <- as.POSIXct(substr(d$date, 1, 10), tz = "UTC")
    d <- d[d$date >= as.POSIXct(P_WARM[1], tz = "UTC") &
           d$date <= as.POSIXct(P_TEST[2], tz = "UTC"), ]
    idx <- function(p) which(d$date >= as.POSIXct(p[1], tz = "UTC") &
                             d$date <= as.POSIXct(p[2], tz = "UTC"))
    i_warm <- idx(P_WARM); i_train <- idx(P_TRAIN); i_test <- idx(P_TEST)
    i_all  <- c(i_train, i_test)

    # model inputs (temperature + 5 elevation layers only for CemaNeige)
    if (snow) {
      inputs <- CreateInputsModel(FUN_MOD = FUN, DatesR = d$date,
                                  Precip = d[[fv["P"]]], PotEvap = d[[fv["E"]]],
                                  TempMean = d[[fv["T"]]], HypsoData = hypso_for(gid),
                                  NLayers = 5, verbose = FALSE)
    } else {
      inputs <- CreateInputsModel(FUN_MOD = FUN, DatesR = d$date,
                                  Precip = d[[fv["P"]]], PotEvap = d[[fv["E"]]])
    }

    # calibration on 1988-97, maximising NSE
    run_cal  <- suppressWarnings(CreateRunOptions(FUN_MOD = FUN, InputsModel = inputs,
                                 IndPeriod_WarmUp = i_warm, IndPeriod_Run = i_train, verbose = FALSE))
    crit_cal <- CreateInputsCrit(FUN_CRIT = ErrorCrit_NSE, InputsModel = inputs,
                                 RunOptions = run_cal, Obs = d$discharge_spec[i_train])
    cal <- Calibration_Michel(InputsModel = inputs, RunOptions = run_cal, InputsCrit = crit_cal,
                              CalibOptions = CreateCalibOptions(FUN_MOD = FUN),
                              FUN_MOD = FUN, verbose = FALSE)
    par <- cal$ParamFinalR

    # continuous run 1988-2008 with calibrated parameters
    run_all <- suppressWarnings(CreateRunOptions(FUN_MOD = FUN, InputsModel = inputs,
                                IndPeriod_WarmUp = i_warm, IndPeriod_Run = i_all, verbose = FALSE))
    sim <- FUN(InputsModel = inputs, RunOptions = run_all, Param = par)$Qsim
    obs <- d$discharge_spec[i_all]
    tr  <- seq_along(i_train); te <- (length(i_train) + 1):length(i_all)

    write.csv(data.frame(date = format(d$date[i_all], "%Y-%m-%d"), q_obs = obs, q_sim = sim),
              file.path(out_dir, "sim", paste0(model, "_", gid, ".csv")), row.names = FALSE)
    out <- data.frame(gauge_id = gid, model = model, forcing = forcing,
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
cat(sprintf("%s (forcing %s) finished in %.1f min\n\n", model, forcing,
            as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
