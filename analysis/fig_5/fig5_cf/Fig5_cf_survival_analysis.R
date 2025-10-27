# ===============================
# TCGA Analysis Script (DESeq2 → E3 mapping → Visualization → Survival Analysis)
# Figure 5c, f (BRCA example)
# ===============================
# Required packages

# Please install the following packages beforehand
# install.packages(c("BiocManager", "dplyr", "ggplot2", "ggrepel", "ggExtra", "readr", "survival", "survminer"))
# BiocManager::install(c("TCGAbiolinks", "DESeq2", "SummarizedExperiment", "apeglm"))

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(DESeq2)
  library(SummarizedExperiment)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(ggExtra)
  library(survival)
  library(survminer)
})

# -------------------------------
# Parameter settings (modify as needed)
# -------------------------------
target_cancer_type <- "BRCA"                  # TCGA abbreviation
alpha <- 0.05                                 # padj threshold
lfc_cutoff <- 1                               # |log2FC| threshold
ids_to_label <- c('O14965','Q12834', 'Q96T88')
# O14965: AURKA
# Q12834: CDC20
# Q96T88: UHRF1

map_path <- 'e3_uniprotid_ensmbl_id.tsv'      # E3 ligase UniProtID-ENSG mapping table
# Mapping table can be obtained from the UniProt ID mapping service: https://www.uniprot.org/id-mapping

clinical_path <- 'clinical.tsv'               # Clinical TSV
# Clinical data can be obtained from https://portal.gdc.cancer.gov/projects/TCGA-XXXX where XXXX is the TCGA abbreviation
# For BRCA: tar -xvf clinical.project-tcga-brca.2025-03-23.tar

out_dir <- 'graphic_output'                   # Output directory for images

# Create output directory
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# -------------------------------
# Functions
# -------------------------------
get_tcga_counts <- function(project_abbr) {
  target_project <- paste0("TCGA-", project_abbr)
  query <- GDCquery(
    project = target_project,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    sample.type = c("Primary Tumor", "Solid Tissue Normal")
  )
  GDCdownload(query)
  data <- GDCprepare(query, save = TRUE, save.filename = paste0(project_abbr, "_data.rda"))
  return(data)
}

perform_deseq_and_save <- function(se) {
  sample_info <- colData(se)
  sample_info$sample_type <- as.factor(sample_info$sample_type)

  dds <- DESeqDataSetFromMatrix(countData = assay(se), colData = sample_info, design = ~ sample_type)
  dds <- dds[rowSums(counts(dds)) > 1, ]
  dds <- DESeq(dds)

  res <- results(dds, name = "sample_type_Solid.Tissue.Normal_vs_Primary.Tumor")
  res <- lfcShrink(dds, coef = "sample_type_Solid.Tissue.Normal_vs_Primary.Tumor", res = res, type = "apeglm")

  resOrdered <- res[order(res$padj), ]
  write.csv(as.data.frame(resOrdered), file = paste0(target_cancer_type, "_resOrdered.csv"))
  message("DESeq2 results have been written to: ", paste0(target_cancer_type, "_resOrdered.csv"))
  invisible(resOrdered)
}

safe_read_DESeq_csv <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  # Auto-detect EnsemblID column (handle cases where row names dropped)
  if ('Unnamed: 0' %in% names(df)) {
    df <- df %>% rename(EnsemblID = `Unnamed: 0`)
  } else if ('...1' %in% names(df)) {
    df <- df %>% rename(EnsemblID = `...1`)
  } else if ('X1' %in% names(df)) {
    df <- df %>% rename(EnsemblID = X1)
  } else if (!('EnsemblID' %in% names(df)) && !is.null(df[[1]])) {
    names(df)[1] <- 'EnsemblID'
  }
  df
}

build_e3_table_and_plots <- function(map_path, deseq_csv) {
  # Load mapping table
  df_id <- read_tsv(map_path, col_names = FALSE, show_col_types = FALSE)
  colnames(df_id) <- c('UniProtID', 'EnsemblID')

  # Load DESeq results
  df_DESeq <- safe_read_DESeq_csv(deseq_csv) %>%
    mutate(`-log10(padj)` = -log10(padj),
           log2FoldChange = -log2FoldChange)  # Align to tumor/normal

  # Merge
  df_e3 <- df_id %>% inner_join(df_DESeq, by = 'EnsemblID')
  message(sprintf("Number of matched genes %s", nrow(df_e3)))

  # Plot: colored volcano + labels
  p1 <- ggplot(df_e3, aes(x = log2FoldChange, y = `-log10(padj)`, color = log2FoldChange)) +
    geom_point(alpha = 0.9, size = 1.2) +
    scale_color_gradient2(low = 'blue', mid = 'white', high = 'red', midpoint = 0) +
    geom_text_repel(data = dplyr::filter(df_e3, UniProtID %in% ids_to_label),
                    aes(label = UniProtID), size = 3, max.overlaps = 100) +
    labs(title = 'Volcano plot (colored by log2FC)',
         x = 'log2FoldChange (tumor/normal)', y = '-log10(padj)', color = 'log2FC') +
    theme_minimal()
  ggsave(file.path(out_dir, 'volcano_colored.png'), p1, width = 7, height = 6, dpi = 300)

  # Significant gene filter
  df_e3_sig <- df_e3 %>% filter(padj < alpha, abs(log2FoldChange) > lfc_cutoff)
  message(sprintf("Number of significantly differentially expressed genes %s", nrow(df_e3_sig)))
  write_csv(df_e3_sig, 'sigGenes_E3.csv')
  return(df_e3_sig)
}

load_clinical_df <- function(path) {
  clinical <- read.delim(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  clinical$patient_id <- substr(clinical$cases.submitter_id, 1, 12)
  clinical
}

run_survival_for_sig_genes <- function(se, sigGenes, clinical) {
  # Remove normal samples
  id <- substr(colnames(se), 14, 15)
  data_tumor <- se[, id != "11"]

  # Expression matrix & patient IDs
  expr_matrix <- assay(data_tumor)
  patient_ids <- substr(colnames(data_tumor), 1, 12)
  colnames(expr_matrix) <- patient_ids

  label_ids <- intersect(ids_to_label, sigGenes$UniProtID)
  sigGenes <- dplyr::filter(sigGenes, UniProtID %in% label_ids)
  if (nrow(sigGenes) == 0) {
    message("No significant genes matching ids_to_label found. Skipping survival analysis.")
    write.csv(data.frame(EnsemblID=character(), UniProtID=character(), pvalue=numeric()),
              file = "surv_results.csv", row.names = FALSE)
    return(invisible(data.frame(EnsemblID=character(), UniProtID=character(), pvalue=numeric())))
  }

  surv_results <- data.frame(EnsemblID = character(), UniProtID = character(), pvalue = numeric(), stringsAsFactors = FALSE)

  for (i in seq_len(nrow(sigGenes))) {
    gene_ensembl <- sigGenes$EnsemblID[i]
    gene_uniprot  <- sigGenes$UniProtID[i]

    if (!(gene_ensembl %in% rownames(expr_matrix))) {
      message("Gene ", gene_ensembl, " is not found in the expression matrix. Skipping.")
      next
    }

    gene_expr <- expr_matrix[gene_ensembl, ]

    df_gene <- data.frame(patient_id = names(gene_expr), expression = as.numeric(gene_expr), stringsAsFactors = FALSE) %>%
      group_by(patient_id) %>% summarize(expression = mean(expression, na.rm = TRUE), .groups = 'drop')

    df_merge <- clinical %>% inner_join(df_gene, by = 'patient_id')
    df_merge <- df_merge[!is.na(df_merge$expression), ]
    if (nrow(df_merge) < 10) { message("Too few samples for gene ", gene_ensembl, ". Skipping."); next }

    median_expr <- median(df_merge$expression, na.rm = TRUE)
    df_merge$expr_group <- factor(ifelse(df_merge$expression >= median_expr, "High", "Low"),
                                  levels = c("High", "Low"))
    if (length(unique(df_merge$expr_group)) < 2) { message("Gene ", gene_ensembl, " has only one group. Skipping."); next }

    # Survival time (handle NA)
    df_merge$time <- ifelse(is.na(df_merge$demographic.days_to_death), df_merge$diagnoses.days_to_last_follow_up, df_merge$demographic.days_to_death)
    df_merge$time <- suppressWarnings(as.numeric(df_merge$time))
    df_merge$status <- ifelse(df_merge$demographic.vital_status == "Dead", 1, 0)
    df_merge <- df_merge[!is.na(df_merge$time), ]
    if (nrow(df_merge) < 10) { message("Too few samples for gene ", gene_ensembl, ". Skipping."); next }

    fit <- survdiff(Surv(time, status) ~ expr_group, data = df_merge)
    pval <- 1 - pchisq(fit$chisq, df = 1)

    surv_results <- rbind(surv_results, data.frame(EnsemblID = gene_ensembl, UniProtID = gene_uniprot, pvalue = pval, stringsAsFactors = FALSE))

    if (pval <= 0.05) {
      km_fit <- survfit(Surv(time, status) ~ expr_group, data = df_merge)
      km_plot <- ggsurvplot(km_fit, data = df_merge, pval = TRUE, conf.int = TRUE,
                            title = paste("Kaplan-Meier Survival Curve for", gene_uniprot),
                            legend.title = "Expression Group", legend.labs = c("High Expression", "Low Expression"))
      ggsave(filename = file.path(out_dir, paste0(gene_uniprot, "_KM_curve.png")), plot = km_plot$plot, width = 8, height = 6, dpi = 300)
    }
  }

  write.csv(surv_results, file = "surv_results.csv", row.names = FALSE)
  print(surv_results)
  invisible(surv_results)
}

# -------------------------------
# Execution flow
# -------------------------------
message("[1/4] Downloading and preparing TCGA data...")
data <- get_tcga_counts(target_cancer_type)

message("[2/4] Running DESeq2 differential expression analysis...")
perform_deseq_and_save(data)

message("[3/4] Merging with E3 mapping table, visualization, and extracting significant genes...")
df_sig <- build_e3_table_and_plots(map_path, paste0(target_cancer_type, "_resOrdered.csv"))

message("[4/4] Running survival analysis...")
clinical <- load_clinical_df(clinical_path)
run_survival_for_sig_genes(data, df_sig, clinical)

message("Completed: Figures saved in ", out_dir, ", tables saved in the current directory.")
