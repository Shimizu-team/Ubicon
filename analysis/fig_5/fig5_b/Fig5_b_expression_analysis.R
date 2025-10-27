# Library installation
# Install the following packages in advance
# install.packages(c("BiocManager", "dplyr", "ggplot2   ", "ggrepel", "ggExtra", "readr", "sva"))
# BiocManager::install(c("TCGAbiolinks", "SummarizedExperiment", "sva"))

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(sva)  
  library(dplyr)        
  library(DESeq2)
  library(RColorBrewer)
})

# -------------------------------
# Parameter settings (modify as needed)
# -------------------------------
# Set target cancer types for analysis
cancer_types <- c("TCGA-BRCA", "TCGA-COAD", "TCGA-CESC", "TCGA-STAD",
                  "TCGA-PRAD", "TCGA-PAAD", "TCGA-LUAD", "TCGA-LUSC",
                  "TCGA-READ", "TCGA-UCEC", "TCGA-KIRC", "TCGA-LIHC",
                  "TCGA-THCA", "TCGA-ESCA", "TCGA-UCS",  "TCGA-OV")

# E3 ligase gene list
# Load EnsID list: mapping between E3 ligase UniProt IDs and Ensembl IDs. Obtained via UniProt ID mapping (https://www.uniprot.org/id-mapping)
e3_mapping_path <- "e3_uniprotid_ensmbl_id.tsv"

# -------------------------------
# Functions
# -------------------------------
remove_version <- function(x) sub("\\..*", "", x)

load_e3_targets <- function(mapping_path) {
  e3_genes <- read.delim(mapping_path, stringsAsFactors = FALSE)
  e3_genes$EnsID
}

  # 1. Download or load SE for each project
load_or_download_se <- function(cancer) {
  query <- GDCquery(project = cancer, 
                    data.category = "Transcriptome Profiling",
                    data.type = "Gene Expression Quantification",
                    workflow.type = "STAR - Counts",
                    sample.type = c("Primary Tumor", "Solid Tissue Normal"))
  # If an RDS already exists, load it. Otherwise check GDCdata/<project>; if present, skip downloading and load with GDCprepare.
  # If neither exists, download, prepare, and save to an RDS file.
  dir.create("raw_se", showWarnings = FALSE, recursive = TRUE)
  rds_raw_path <- file.path("raw_se", paste0("raw_", cancer, ".rds"))

  # GDCdata の既存データ有無をチェック（デフォルトの保存先）
  gdc_root <- "GDCdata"
  proj_dir <- file.path(gdc_root, cancer)
  has_local_gdc <- dir.exists(proj_dir) && length(list.files(proj_dir, recursive = TRUE)) > 0

  if (file.exists(rds_raw_path)) {
    data <- readRDS(rds_raw_path)
    message("Loaded RDS for ", cancer, " (download skipped).\n")
  } else if (has_local_gdc) {
    message("Found existing data for ", cancer, " in GDCdata. Skipping download and loading.\n")
    data <- GDCprepare(query)
    saveRDS(data, rds_raw_path)
  } else {
    GDCdownload(query)
    message("Downloaded data for ", cancer, ".\n")
    data <- GDCprepare(query)
    saveRDS(data, rds_raw_path)
  }
  data
}

  # 2. Run DESeq2 for each project and save result CSV
run_deseq_for_cancer <- function(data, target_ids, cancer) {
  # Sample metadata
  sample_info <- colData(data)
  sample_info$sample_type <- as.factor(sample_info$sample_type)

  rownames(data) <- remove_version(rownames(data))
  target_ids <- remove_version(target_ids)
  filtered_data <- data[rownames(data) %in% target_ids, ]

  # Prepare DESeq2 objects
  dds <- DESeqDataSetFromMatrix(countData = assay(filtered_data),
                                colData = sample_info,
                                design = ~ sample_type)
  dds <- dds[rowSums(counts(dds)) > 1, ]

  # Run DESeq
  dds <- DESeq(dds)
  # Get results and apply lfcShrink
  res <- results(dds, name = "sample_type_Solid.Tissue.Normal_vs_Primary.Tumor")
  res <- lfcShrink(dds, coef = "sample_type_Solid.Tissue.Normal_vs_Primary.Tumor", res = res, type = "apeglm")

  # Order and save
  resOrdered <- res[order(res$padj), ]
  dir.create("DESeq_results", showWarnings = FALSE, recursive = TRUE)
  filename <- paste0("DESeq_results/resOrdered_", cancer, ".csv")
  write.csv(resOrdered, file = filename)

  invisible(resOrdered)
}

  # 3. Extract significant genes and select EnsIDs
select_target_e3 <- function(threshold_pvalue = 0.05, threshold_log2fc = 1) {
  dir.create("DESeq_results", showWarnings = FALSE, recursive = TRUE)
  csv_files <- list.files("DESeq_results", pattern = "\\.csv$", full.names = TRUE)

  filtered_list <- list()
  for (fp in csv_files) {
    df <- read.csv(fp, stringsAsFactors = FALSE, check.names = FALSE)
    if (!("EnsID" %in% colnames(df))) {
      if ("Unnamed: 0" %in% colnames(df)) {
        df$EnsID <- df[["Unnamed: 0"]]
      } else if ("Unnamed..0" %in% colnames(df)) {
        df$EnsID <- df[["Unnamed..0"]]
      } else if ("X" %in% colnames(df)) {
        df$EnsID <- df[["X"]]
      } else {
        df$EnsID <- df[[1]]
      }
    }
    keep <- with(df, (padj < threshold_pvalue) & (abs(log2FoldChange) > threshold_log2fc))
    filtered_df <- df[which(keep), , drop = FALSE]
    filtered_list[[basename(fp)]] <- filtered_df
    message("Filtered", basename(fp), ":", nrow(filtered_df), "rows\n")
  }

  ensids <- unlist(lapply(filtered_list, function(d) d$EnsID), use.names = FALSE)
  ensid_counts <- table(ensids)
  selected_ids <- names(ensid_counts[ensid_counts >= 7])

  target_e3 <- selected_ids

  if (length(target_e3) == 0) {
    message("select_target_e3: No EnsIDs met the criteria. Check that CSVs exist under DESeq_results and verify thresholds (threshold_pvalue / threshold_log2fc).")
  }
  target_e3
}

  # 4. Load each cancer dataset, preprocess, and store in an SE list
load_and_preprocess_se_list <- function(cancer_types, target_e3) {
  se_list <- list()
  for (cancer_name in cancer_types) {
    cat("Processing:", cancer_name, "\n")
    rds_path <- paste0("filtered_data/filtered_data_", cancer_name, ".rds")
    se <- readRDS(rds_path)

    se_filt <- se[rownames(se) %in% target_e3, ]
    tumor_idx <- substr(colnames(se_filt), 14, 15) != "11"
    se_filt_tumor <- se_filt[, tumor_idx]

    norm_counts <- log2(assay(se_filt_tumor) + 1)
    assay(se_filt_tumor) <- norm_counts

    se_list[[cancer_name]] <- se_filt_tumor
  }
  se_list
}

  # 5. cbind SEs in the list and perform batch correction with ComBat
combine_and_combat <- function(se_list) {
  common_genes <- Reduce(intersect, lapply(se_list, rownames))
  se_list_sub <- lapply(se_list, function(se) se[common_genes, ])

  all_coln <- c()
  for (se_tmp in se_list_sub) {
    all_coln <- union(all_coln, colnames(colData(se_tmp)))
  }
  for (i in seq_along(se_list_sub)) {
    for (cn in all_coln) {
      if (!cn %in% colnames(colData(se_list_sub[[i]]))) {
        colData(se_list_sub[[i]])[[cn]] <- NA
      }
    }
  }

  combined_data <- se_list_sub[[1]]
  if (length(se_list_sub) > 1) {
    for (i in 2:length(se_list_sub)) {
      combined_data <- cbind(combined_data, se_list_sub[[i]])
    }
  }

  batch <- factor(colData(combined_data)$project_id)
  expr_mat <- assay(combined_data)
  combat_expr_mat <- ComBat(
    dat         = expr_mat,
    batch       = batch,
    mod         = NULL,
    par.prior   = TRUE,
    prior.plots = FALSE,
    mean.only   = FALSE
  )
  assay(combined_data) <- combat_expr_mat
  combined_data
}

  # 6. Aggregate by patient ID (median)
aggregate_by_patient <- function(combined_data) {
  col.metadata <- as.data.frame(colData(combined_data))
  col.metadata$barcode <- rownames(col.metadata)
  col.metadata$bcr_patient_barcode <- substr(col.metadata$barcode, 1, 12)

  expr_mat2 <- assay(combined_data)
  unique_patients <- unique(col.metadata$bcr_patient_barcode)

  agg_expr <- sapply(unique_patients, function(pid) {
    idx <- col.metadata$bcr_patient_barcode == pid
    rowMedians(expr_mat2[, idx, drop = FALSE], na.rm = TRUE)  # as in the original code
  })
  agg_expr <- as.matrix(agg_expr)
  rownames(agg_expr) <- rownames(expr_mat2)

  agg_metadata <- do.call(rbind, lapply(unique_patients, function(pid) {
    tmp <- col.metadata[col.metadata$bcr_patient_barcode == pid, ]
    tmp[1, ]
  }))
  rownames(agg_metadata) <- unique_patients
  agg_metadata <- agg_metadata[, sapply(agg_metadata, function(x) !is.list(x))]
  agg_metadata <- agg_metadata[, c("bcr_patient_barcode", "project_id")]

  list(agg_expr = agg_expr, agg_metadata = agg_metadata)
}

  # 7. Build row metadata
build_row_metadata <- function(agg_expr) {
  row.metadata <- data.frame(
    gene_symbol = rownames(agg_expr),
    stringsAsFactors = FALSE
  )
  rownames(row.metadata) <- rownames(agg_expr)
  row.metadata
}

  # 8. Plot heatmap (preserve original parameters)
plot_heatmap <- function(agg_expr, agg_metadata) {
  unique_projects <- unique(agg_metadata$project_id)
  base_colors <- brewer.pal(12, "Paired")
  extra_colors <- c("#BFB500", "#006934", "#E4007F", "#898989")
  combined_palette <- c(base_colors, extra_colors)
  project_colors <- combined_palette[1:length(unique_projects)]
  names(project_colors) <- unique_projects
  col.colors <- list(
    project_id = project_colors
  )

  row.metadata <- build_row_metadata(agg_expr)

  TCGAvisualize_Heatmap(
    data              = agg_expr,
    col.metadata      = agg_metadata,
    row.metadata      = row.metadata,
    col.colors        = col.colors,
    row.colors        = NULL,
    show_column_names = FALSE,
    show_row_names    = TRUE,
    cluster_rows      = TRUE,
    cluster_columns   = TRUE,
    extremes          = seq(-2,2,1),
    rownames.size     = 12,
    title             = "Pan-cancer E3 ligase Heatmap (16 cancer types)",
    color.levels      = colorRampPalette(c("green", "black", "red"))(n = 5),
    values.label      = NULL,
    filename          = "heatmap_pancan_fig5_b.pdf",
    width             = 25,
    height            = 15,
    type              = "expression",
    scale             = "row",
    heatmap.legend.color.bar = "continuous"
  )
}

  # =====================
# ===================== Module execution wrapper
# =====================
run_module <- function(step, ...) {
  step <- match.arg(step, c(
    "download_deseq",          # Original for-loop (download/load + DESeq + save).
    "select_targets",          # Extract EnsIDs from significantly altered genes.
    "prep_se_list",            # Load and preprocess SE for each cancer type.
    "combine_combat",          # cbind + ComBat
    "aggregate",               # Aggregate by patient.
    "plot"                      # Draw heatmap.
  ))

  dots <- list(...)

  if (step == "download_deseq") {
    target_ids <- load_e3_targets(e3_mapping_path)
    for (cancer in cancer_types) {
      csv_path <- file.path("DESeq_results", paste0("resOrdered_", cancer, ".csv"))
      if (file.exists(csv_path)) {
        message("CSV result for ", cancer, " already exists in DESeq_results. Skipping.\n")
        next
      }
      data <- load_or_download_se(cancer)
      # Skip if the CSV for this cancer already exists in DESeq_results

      run_deseq_for_cancer(data, target_ids, cancer)
    }
    return(invisible(TRUE))
  }

  if (step == "select_targets") {
    return(select_target_e3())
  }

  if (step == "prep_se_list") {
    if (is.null(dots$target_e3)) stop("Please provide 'target_e3'")
    return(load_and_preprocess_se_list(cancer_types, dots$target_e3))
  }

  if (step == "combine_combat") {
    if (is.null(dots$se_list)) stop("Please provide 'se_list'")
    return(combine_and_combat(dots$se_list))
  }

  if (step == "aggregate") {
    if (is.null(dots$combined_data)) stop("Please provide 'combined_data'")
    return(aggregate_by_patient(dots$combined_data))
  }

  if (step == "plot") {
    if (is.null(dots$agg_expr) || is.null(dots$agg_metadata)) stop("Please provide 'agg_expr' and 'agg_metadata'")
    plot_heatmap(dots$agg_expr, dots$agg_metadata)
    return(invisible(TRUE))
  }
}

# =====================
# Full pipeline (call manually as needed)
# =====================
run_all <- function() {
  # 1) Download/load + DESeq + save CSV
  message("Step 1: Downloading data and running DESeq2...")
  run_module("download_deseq")
  # 2) Select EnsIDs
  message("Step 2: Selecting target E3 ligase genes...")
  target_e3 <- run_module("select_targets")
  # 3) Prepare SE list
  message("Step 3: Preparing SE list...")
  se_list <- run_module("prep_se_list", target_e3 = target_e3)
  # 4) Combine + ComBat
  message("Step 4: Combining data and applying ComBat...")
  combined_data <- run_module("combine_combat", se_list = se_list)
  # 5) Aggregate
  message("Step 5: Aggregating by patient...")
  agg <- run_module("aggregate", combined_data = combined_data)
  # 6) Plot heatmap
  message("Step 6: Plotting heatmap...")
  run_module("plot", agg_expr = agg$agg_expr, agg_metadata = agg$agg_metadata)
  invisible(list(
    target_e3 = target_e3,
    se_list = se_list,
    combined_data = combined_data,
    agg_expr = agg$agg_expr,
    agg_metadata = agg$agg_metadata
  ))
}

# =====================
# Run
# =====================
out <- run_all()