
library(dada2); packageVersion("dada2")
#update("dada2")

path <- "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/poyet_et_al/fastq"
# Forward and reverse fastq filenames have format: SAMPLENAME_R1_001.fastq and SAMPLENAME_R2_001.fastq
fnFs <- sort(list.files(path, pattern="_1.fastq", full.names = TRUE))
fnRs <- sort(list.files(path, pattern="_2.fastq", full.names = TRUE))
# Extract sample names, assuming filenames have format: SAMPLENAME_XXX.fastq
sample.names <- sapply(strsplit(basename(fnFs), "_"), `[`, 1)

plotQualityProfile(fnFs[1:2])
plotQualityProfile(fnRs[1:2])

# Place filtered files in filtered/ subdirectory
filtFs <- file.path(path, "filtered", paste0(sample.names, "_1_filt.fastq.gz"))
filtRs <- file.path(path, "filtered", paste0(sample.names, "_2_filt.fastq.gz"))

#filtFs <- filtFs[file.exists(filtFs)]
#filtRs <- filtRs[file.exists(filtRs)]


names(filtFs) <- sample.names
names(filtRs) <- sample.names

# original parameter settings
#truncLen = 140
#truncQ = 2
out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs, truncLen=145, minLen=120, trimLeft=5,
                     maxN=0, maxEE=c(2,2), truncQ=10, rm.phix=TRUE,
                     compress=TRUE, multithread=TRUE) # On Windows set multithread=FALSE



head(out)


filtFs <- filtFs[file.exists(filtFs)]
filtRs <- filtRs[file.exists(filtRs)]


# learn error rates
errF <- learnErrors(filtFs, multithread=TRUE)
errR <- learnErrors(filtRs, multithread=TRUE)

plotErrors(errF, nominalQ=TRUE)

# sample inference
dadaFs <- dada(filtFs, err=errF, multithread=TRUE, pool=TRUE)
dadaRs <- dada(filtRs, err=errR, multithread=TRUE, pool=TRUE)

dadaFs[[1]]

# Merge paired reads
mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs, verbose=TRUE)
# Inspect the merger data.frame from the first sample
head(mergers[[1]])

# Construct sequence table
seqtab <- makeSequenceTable(mergers)
dim(seqtab)
# Inspect distribution of sequence lengths
table(nchar(getSequences(seqtab)))

# Remove chimeras
seqtab.nochim <- removeBimeraDenovo(seqtab, method="consensus", multithread=TRUE, verbose=TRUE)
dim(seqtab.nochim)
sum(seqtab.nochim)/sum(seqtab)



#Track reads through the pipeline
getN <- function(x) sum(getUniques(x))
track <- cbind(out, sapply(dadaFs, getN), sapply(dadaRs, getN), sapply(mergers, getN), rowSums(seqtab.nochim))
# If processing a single sample, remove the sapply calls: e.g. replace sapply(dadaFs, getN) with getN(dadaFs)
colnames(track) <- c("input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim")
rownames(track) <- sample.names
head(track)




#Assign taxonomy
taxa <- assignTaxonomy(seqtab.nochim, "/Users/williamrshoemaker/GitHub/strain_macroecology/data/poyet/poyet_16s/silva_nr99_v138.1_train_set.fa.gz", multithread=TRUE)

# make species level assignments based on exact matching between ASVs and sequenced reference strains
taxa <- addSpecies(taxa, "/Users/williamrshoemaker/GitHub/strain_macroecology/data/poyet/poyet_16s/silva_species_assignment_v138.1.fa.gz")

taxa.print <- taxa # Removing sequence rownames for display only
rownames(taxa.print) <- NULL
head(taxa.print)


# export site-by-species
write.table(t(seqtab.nochim), "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/poyet_et_al/seqtab-nochim-gut.txt", sep="\t", row.names=TRUE, col.names=TRUE, quote=FALSE)
# export taxonomy
write.table(taxa, "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/poyet_et_al/seqtab-nochim-taxa-gut.txt", sep="\t", row.names=TRUE, col.names=TRUE, quote=FALSE)
