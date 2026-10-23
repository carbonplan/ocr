################################################################################
# Per-cell attributions of a trained FDP forest, run by fdp_train.py after
# train_fdp.R.
#
# Writes contributions.tif, one int16 band per predictor in the forest's
# variable order (value = contribution x 1e4, nodata -32768), so that per cell
#   baseline + sum(bands) / 1e4 = fdp
# with the baseline in contributions.json. The attributions are path based
# (Saabas): every split a cell passes credits its change in in-bag mean to the
# split variable. Internal node values come from the in-bag training rows,
# checked against the leaf values ranger stored, and the reconstructed
# prediction is checked against fdp.tif.
#
# Usage:
#   Rscript contrib_fdp.R --model rf.rds --train train.rds --predictors DIR \
#     --prediction fdp.tif --out DIR [--threads 8]
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
out <- arg("out")
threads <- as.integer(arg("threads", "8"))
scale <- 1e4
nodata <- -32768

suppressPackageStartupMessages({
  library(raster)
  library(ranger)
  library(parallel)
  library(jsonlite)
  library(Rcpp)
})

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
sourceCpp(file.path(script_dir, "fdp_contrib.cpp"))

t0 <- Sys.time()
rf <- readRDS(arg("model"))
train <- readRDS(arg("train"))
forest <- rf$forest
vars <- forest$independent.variable.names
ordered <- forest$is.ordered
if (!all(ordered)) stop("partition splits are not supported; every variable must be ordered")

# Cells and training rows coded as ranger codes them: factor levels by their
# position in the forest's (reordered) covariate levels.
encode <- function(df) {
  x <- sapply(vars, function(v) {
    levels <- forest$covariate.levels[[v]]
    if (is.null(levels)) as.numeric(df[[v]]) else match(as.character(df[[v]]), levels)
  })
  matrix(x, ncol = length(vars), dimnames = list(NULL, vars))
}

ints <- function(l) lapply(l, as.integer)
left <- ints(lapply(forest$child.nodeIDs, `[[`, 1))
right <- ints(lapply(forest$child.nodeIDs, `[[`, 2))
split_var <- ints(forest$split.varIDs)
split_value <- forest$split.values

x_train <- encode(train)
y_train <- as.numeric(as.character(train$pa) == "1")
node_values <- lapply(seq_len(rf$num.trees), function(t) {
  fdp_node_values(left, right, split_var, split_value, t - 1L, ordered, x_train, y_train, as.integer(rf$inbag.counts[[t]]))
})

# terminal.class.counts holds class fractions per leaf in forest$class.values
# order; find the column holding class "1" and require every leaf to match.
leaf_error <- function(k) {
  max(sapply(seq_len(rf$num.trees), function(t) {
    leaves <- which(left[[t]] == 0 & right[[t]] == 0)
    stored <- sapply(forest$terminal.class.counts[[t]][leaves], `[`, k)
    max(abs(stored - node_values[[t]][leaves]))
  }))
}
errors <- sapply(seq_along(forest$class.values), leaf_error)
if (min(errors) > 1e-9) stop("node values do not reproduce ranger's leaves (max error ", min(errors), ")")
baseline <- mean(sapply(node_values, `[`, 1))
cat(sprintf("node values for %d trees, leaf error %.1e, baseline %.4f\n", rf$num.trees, min(errors), baseline))

vars_raster <- c("floodplain", "distwater", "elevation", "hand", "nlcd", "pcurv", "hydric", "twi", "channel", "roads", "m3dp")
predictors <- stack(file.path(arg("predictors"), paste0(vars_raster, ".tif")))
names(predictors) <- vars_raster
prediction <- raster(arg("prediction"))

grid <- predictors[[1]]
template <- brick(extent(grid), nrows = nrow(grid), ncols = ncol(grid), crs = crs(grid), nl = length(vars))
bricked <- writeStart(
  template, file.path(out, "contributions.tif"), datatype = "INT2S", NAflag = nodata, overwrite = TRUE,
  options = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES", "INTERLEAVE=BAND")
)
rows <- max(1, floor(4e6 / ncol(template)))
max_error <- 0
cells <- 0
for (row in seq(1, nrow(template), by = rows)) {
  n <- min(rows, nrow(template) - row + 1)
  band <- as.data.frame(getValues(predictors, row, n))
  keep <- which(complete.cases(band))
  result <- matrix(NA_real_, nrow(band), length(vars))
  if (length(keep)) {
    x <- encode(band[keep, , drop = FALSE])
    parts <- split(seq_len(nrow(x)), cut(seq_len(nrow(x)), min(threads, nrow(x)), labels = FALSE))
    pieces <- mclapply(parts, function(i) {
      fdp_contributions(left, right, split_var, split_value, node_values, ordered, x[i, , drop = FALSE])
    }, mc.cores = threads)
    contrib <- do.call(rbind, pieces)
    predicted <- getValues(prediction, row, n)[keep]
    max_error <- max(max_error, abs(contrib[, length(vars) + 1] - predicted), na.rm = TRUE)
    result[keep, ] <- round(contrib[, seq_along(vars)] * scale)
    cells <- cells + length(keep)
  }
  bricked <- writeValues(bricked, result, row)
}
bricked <- writeStop(bricked)
cat(sprintf("%d cells, max |reconstructed - fdp.tif| = %.2e\n", cells, max_error))
if (max_error > 1e-5) stop("reconstructed predictions do not match fdp.tif")

write_json(
  list(
    drivers = vars,
    baseline = baseline,
    scale = 1 / scale,
    nodata = nodata,
    method = "saabas",
    cells = cells,
    max_reconstruction_error = max_error,
    leaf_error = min(errors),
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ),
  file.path(out, "contributions.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA
)
cat("done\n")
