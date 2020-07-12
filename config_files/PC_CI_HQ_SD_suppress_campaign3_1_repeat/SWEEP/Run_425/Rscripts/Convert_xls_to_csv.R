library("readr")   # Fast csv write



DefaultInFolder <- "."
# parse command line arguments
args = commandArgs(trailingOnly=TRUE)
InFolder <- if (length(args > 0)) args[1] else DefaultInFolder

Pattern <- "\\.xls"


files <- list.files(path = InFolder, pattern = Pattern,recursive = TRUE, full.names = TRUE)


for (Index in 1:length(files)) 
{
	xls_fileName  <- files[Index]
	csv_fileName  <- gsub("xls", "csv", xls_fileName)

	cat("\txls_fileName = ",paste0(xls_fileName, "\n"))
	cat("converted to -> \n")
	cat("\tcsv_fileName = ",paste0(csv_fileName, "\n\n"))

	indata   <- read.delim(xls_fileName, header = TRUE)
	write_csv(indata, path = csv_fileName)
}
