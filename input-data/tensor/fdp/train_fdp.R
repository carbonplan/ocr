################################################################################
# Flood damage probability random forest for one HUC-2 watershed.
#
# Port of Code_FDP_RandomForest_code.R from the Collins et al. (2022) USGS data
# release (doi:10.5066/P954TTQN), run once per watershed by fdp_train.py.
# The modelling steps follow the released script, including its fold handling:
# the final model trains on the best CV fold's training split, with background
# folds taken from the initial `group` draw.
#
# The released script calls dismo's ssb() and pwdSample() with their default
# lonlat = TRUE on Albers metre coordinates, so those distances are haversine
# values of metres read as degrees, uncorrelated with true separation.
# `--distance lonlat` (the default) keeps that behaviour; `--distance planar`
# measures true distances.
#
# Inputs arrive already cropped to the watershed and masked to its HUC raster
# cells, so the script stacks them directly. Partial dependence and standard
# error rasters are not produced.
#
# Usage:
#   Rscript train_fdp.R --huc 03 --predictors DIR --points FILE --out DIR \
#     --tmp DIR [--threads 16] [--iters 70] [--trees 1000] [--seed 20261022] \
#     [--distance lonlat|planar]
################################################################################

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) {
    if (is.null(default)) stop("missing --", name)
    return(default)
  }
  args[[i + 1]]
}
huc <- arg("huc")
pred_dir <- arg("predictors")
points_path <- arg("points")
out <- arg("out")
threads <- as.integer(arg("threads", "8"))
iters <- as.integer(arg("iters", "70"))
trees <- as.integer(arg("trees", "1000"))
seed <- as.integer(arg("seed", "20261022"))
distance <- match.arg(arg("distance", "lonlat"), c("lonlat", "planar"))
lonlat <- distance == "lonlat"

suppressPackageStartupMessages({
  library(raster)
  library(sp)
  library(dismo)
  library(ranger)
  library(tuneRanger)
  library(mlr)
  library(data.table)
  library(jsonlite)
})

set.seed(seed)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
rasterOptions(tmpdir = arg("tmp"))

clock <- Sys.time()
seconds <- list()
mark <- function(stage) {
  now <- Sys.time()
  seconds[[stage]] <<- as.numeric(difftime(now, clock, units = "secs"))
  clock <<- now
  cat(sprintf("[%s] %s done in %.0fs\n", format(now, "%H:%M:%S"), stage, seconds[[stage]]))
}

cat("HUC", huc, "\n")

vars <- c("floodplain", "distwater", "elevation", "hand", "nlcd", "pcurv", "hydric", "twi", "channel", "roads", "m3dp")
factor_cols <- c("nlcd", "floodplain")
predictors <- stack(file.path(pred_dir, paste0(vars, ".tif")))
names(predictors) <- vars
predictors$nlcd <- as.factor(predictors$nlcd)
predictors$floodplain <- as.factor(predictors$floodplain)
print(predictors)

pts <- fread(points_path)[huc2 == as.integer(huc)]
to_sp <- function(d) {
  SpatialPointsDataFrame(as.matrix(d[, .(x, y)]), data.frame(flood = d$pa), proj4string = crs(predictors))
}
flood_points <- to_sp(pts[pa == 1])
abs_points <- to_sp(pts[pa == 0])
cat("presence points:", length(flood_points), " absence points:", length(abs_points), "\n")

nlcd_vals <- levels(predictors$nlcd)[[1]]$ID
nlcd_vals <- nlcd_vals[!is.na(nlcd_vals)]
fp_vals <- levels(predictors$floodplain)[[1]]$ID
fp_vals <- fp_vals[!is.na(fp_vals)]
mark("load")

backgr <- abs_points
group <- kfold(backgr, 4)

group_backg <- kfold(backgr, 4)
group_pres <- kfold(flood_points, 4)

sb <- list()
mt <- list()
pres <- list()
abs <- list()
thresh <- list()
conf_mat <- list()
sb_fix <- list()
predicted <- list()
auc <- list()
min_node_size <- list()
sample_fraction <- list()
n_test <- list()

for (i in 1:4) {
  backg_train <- backgr[group_backg != i, ]
  backg_test <- backgr[group_backg == i, ]
  pres_train <- flood_points[group_pres != i, ]
  pres_test <- flood_points[group_pres == i, ]

  sb1 <- ssb(pres_test, backg_test, pres_train, lonlat = lonlat)
  sb[[i]] <- as.vector(sb1[, 1] / sb1[, 2])

  pwd <- pwdSample(pres_test, backg_test, pres_train, n = 1, tr = 0.33, lonlat = lonlat)
  pres_test_pwd <- pres_test[!is.na(pwd[, 1]), ]
  back_test_pwd <- backg_test[na.omit(as.vector(pwd)), ]
  sb2 <- ssb(pres_test_pwd, back_test_pwd, pres_train, lonlat = lonlat)
  sb_fix[[i]] <- as.vector(sb2[, 1] / sb2[, 2])

  testpres <- data.frame(extract(predictors, pres_test_pwd))
  testbackg <- data.frame(extract(predictors, back_test_pwd))
  testpres[factor_cols] <- lapply(testpres[factor_cols], factor)
  testbackg[factor_cols] <- lapply(testbackg[factor_cols], factor)

  train <- rbind(pres_train, backg_train)
  pb_train <- c(rep(1, nrow(pres_train)), rep(0, nrow(backg_train)))
  envtrain <- extract(predictors, train)
  envtrain <- data.frame(cbind(pa = pb_train, envtrain))
  envtrain[factor_cols] <- lapply(envtrain[factor_cols], factor)
  envtrain <- envtrain[complete.cases(envtrain), ]
  envtrain$nlcd <- factor(envtrain$nlcd, levels = nlcd_vals)
  envtrain$floodplain <- factor(envtrain$floodplain, levels = fp_vals)

  cat("fold", i, ": tuning on", nrow(envtrain), "rows\n")
  envtrain$pa <- as.factor(envtrain$pa)
  # Reseed so tuning draws the same random numbers whichever distance mode
  # pwdSample ran under, and the two modes differ only in their test sets.
  set.seed(seed + i)
  fdp_task <- makeClassifTask(data = envtrain, target = "pa")
  estimateTimeTuneRanger(fdp_task, num.trees = trees, num.threads = threads, iters = iters)
  fdp_tune <- tuneRanger(fdp_task, measure = list(multiclass.brier), num.trees = trees, num.threads = threads, iters = iters, save.file.path = NULL)
  print(fdp_tune)
  params <- fdp_tune$recommended.pars
  model <- factor(pa) ~ .
  rf1 <- ranger(formula = model, data = envtrain, mtry = params$mtry, min.node.size = params$min.node.size, sample.fraction = params$sample.fraction, num.trees = trees, seed = 1, probability = TRUE, keep.inbag = TRUE, importance = "impurity", respect.unordered.factors = TRUE, num.threads = threads)
  mt[[i]] <- params$mtry
  min_node_size[[i]] <- params$min.node.size
  sample_fraction[[i]] <- params$sample.fraction

  testpres$nlcd <- factor(testpres$nlcd, levels = nlcd_vals)
  testpres$floodplain <- factor(testpres$floodplain, levels = fp_vals)
  testbackg$nlcd <- factor(testbackg$nlcd, levels = nlcd_vals)
  testbackg$floodplain <- factor(testbackg$floodplain, levels = fp_vals)

  testpres_preds <- testpres
  testpres_preds$pa <- "p"
  testbackg_preds <- as.data.frame(testbackg)
  testbackg_preds$pa <- "a"
  data <- rbind(testpres_preds, testbackg_preds)
  data$nlcd <- factor(data$nlcd, levels = nlcd_vals)
  data$floodplain <- factor(data$floodplain, levels = fp_vals)

  pres_df <- as.data.frame(coordinates(pres_test_pwd))
  names(pres_df) <- c("lon", "lat")
  back_df <- as.data.frame(coordinates(back_test_pwd))
  names(back_df) <- c("lon", "lat")
  data <- cbind(data, rbind(pres_df, back_df))
  data <- data[complete.cases(data), ]
  data$pred <- predict(rf1, data[, 1:(length(data) - 2)], type = "response")$predictions[, 2]
  predicted[[i]] <- data

  # Error-minimizer threshold, following dismo::evaluate as the released script does.
  testpres <- testpres[complete.cases(testpres), ]
  testbackg <- testbackg[complete.cases(testbackg), ]
  p <- stats::na.omit(predict(rf1, data.frame(testpres), type = "response")$prediction)[, 2]
  a <- stats::na.omit(predict(rf1, data.frame(testbackg), type = "response")$prediction)[, 2]
  np <- length(p)
  na <- length(a)
  n_test[[i]] <- c(presence = np, absence = na)

  tr <- if (np > 1000) as.vector(quantile(p, 0:1000 / 1000)) else p
  tr <- c(tr, if (na > 1000) as.vector(quantile(a, 0:1000 / 1000)) else a)
  tr <- sort(unique(round(tr, 8)))
  tr <- c(tr - 0.0001, tr[length(tr)] + c(0, 0.0001))

  R <- sum(rank(c(p, a))[1:np]) - (np * (np + 1) / 2)
  auc[[i]] <- R / (as.numeric(na) * as.numeric(np))
  cat("fold", i, "AUC", auc[[i]], "\n")

  tp <- sapply(tr, function(t) sum(p >= t))
  fp <- sapply(tr, function(t) sum(a >= t))
  fn <- np - tp
  tn <- na - fp
  pres[[i]] <- p
  abs[[i]] <- a
  errmin <- fp / (fp + tn) - fn / (tp + fn)
  thresh[[i]] <- tr[which.min(abs(errmin))]
  t <- thresh[[i]]
  conf_mat[[i]] <- c(tp = sum(p >= t), fp = sum(a >= t), fn = sum(p < t), tn = sum(a < t))
  mark(paste0("fold", i))
}

auc <- unlist(auc)
max_auc <- which.max(auc)
cat("mean AUC", mean(auc), " best fold", max_auc, "\n")

backg_train <- backgr[group != max_auc, ]
pres_train <- flood_points[group_pres != max_auc, ]
train <- rbind(pres_train, backg_train)
pb_train <- c(rep(1, nrow(pres_train)), rep(0, nrow(backg_train)))
envtrain <- extract(predictors, train)
envtrain <- data.frame(cbind(pa = pb_train, envtrain))
envtrain[factor_cols] <- lapply(envtrain[factor_cols], factor)
envtrain <- envtrain[complete.cases(envtrain), ]
envtrain$nlcd <- factor(envtrain$nlcd, levels = nlcd_vals)
envtrain$floodplain <- factor(envtrain$floodplain, levels = fp_vals)

model <- factor(pa) ~ .
rf1 <- ranger(formula = model, data = envtrain, mtry = mt[[max_auc]], min.node.size = min_node_size[[max_auc]], sample.fraction = sample_fraction[[max_auc]], num.trees = trees, seed = 1, probability = TRUE, keep.inbag = TRUE, importance = "impurity", respect.unordered.factors = TRUE, num.threads = threads)
print(rf1)
mark("final_model")

# One ranger thread per raster worker, so the cluster does not oversubscribe the cores.
beginCluster(threads)
f_response <- function(model, ...) predict(model, ...)$predictions[, 2]
clusterR(
  predictors, predict,
  args = list(rf1, type = "response", fun = f_response, num.threads = 1),
  filename = file.path(out, "fdp.tif"), datatype = "FLT4S", overwrite = TRUE,
  options = c("COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=YES")
)
endCluster()
mark("predict")

saveRDS(rf1, file.path(out, "rf.rds"))
saveRDS(envtrain, file.path(out, "train.rds"))
saveRDS(list(pres = pres, abs = abs, predicted = predicted), file.path(out, "cv.rds"))
fwrite(data.table(variable = names(rf1$variable.importance), importance = rf1$variable.importance), file.path(out, "importance.csv"))

summary <- list(
  huc = huc,
  points = list(presence = length(flood_points), absence = length(abs_points)),
  cv = list(
    auc = auc,
    threshold = unlist(thresh),
    mtry = unlist(mt),
    min_node_size = unlist(min_node_size),
    sample_fraction = unlist(sample_fraction),
    ssb = sb,
    ssb_after_pwd = sb_fix,
    n_test = n_test,
    confusion = conf_mat
  ),
  best_fold = max_auc,
  final = list(
    rows = nrow(envtrain),
    mtry = mt[[max_auc]],
    min_node_size = min_node_size[[max_auc]],
    sample_fraction = sample_fraction[[max_auc]],
    threshold = thresh[[max_auc]],
    oob_brier = rf1$prediction.error
  ),
  settings = list(threads = threads, iters = iters, trees = trees, seed = seed, distance = distance),
  seconds = seconds,
  r = R.version.string,
  packages = sapply(c("raster", "terra", "sp", "dismo", "ranger", "tuneRanger", "mlr"), function(p) as.character(packageVersion(p)))
)
write_json(summary, file.path(out, "summary.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("done\n")
