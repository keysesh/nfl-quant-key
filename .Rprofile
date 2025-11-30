lib_path <- file.path(
  Sys.getenv("HOME"),
  "Desktop",
  "NFL QUANT",
  "Rlib"
)

if (!dir.exists(lib_path)) {
  dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
}

.libPaths(c(lib_path, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))
