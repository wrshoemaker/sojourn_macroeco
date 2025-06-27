
#install.packages("DECIPHER")
BiocManager::install("DECIPHER")
install.packages("BiocManager")
BiocManager::install("Biostrings")

library(tibble)
library(dplyr)
library("Biostrings")

#install.packages("devtools")
#devtools::install_github("slowkow/ggrepel")


####
### David et al
####

seqtab.nochim <- read.table("/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/david_et_al/seqtab-nochim-gut.txt")
seqtab.nochim <- t(seqtab.nochim)
asv_sequences <- colnames(seqtab.nochim)

# cast them as DNAString
dna <- Biostrings::DNAStringSet(asv_sequences)
#names(dna) <- asv_sequences

metadata(dna)$name <- asv_sequences

# run the clustering at 99% similarity
clusters <- DECIPHER::Clusterize(dna, cutoff = 0.03, processors=4)
# clusters corresponds to order of seqtab.nochim

# order corresponds to order in seqtab.nochim
cat(clusters$cluster, "\n", file="/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/david_et_al/seqtab-nochim-gut-otus.txt")




####
### Poyet et al
####

seqtab.nochim <- read.table("/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/poyet_et_al/seqtab-nochim-gut.txt")
seqtab.nochim <- t(seqtab.nochim)
asv_sequences <- colnames(seqtab.nochim)

# cast them as DNAString
dna <- Biostrings::DNAStringSet(asv_sequences)
#names(dna) <- asv_sequences

metadata(dna)$name <- asv_sequences

# run the clustering at 99% similarity
clusters <- DECIPHER::Clusterize(dna, cutoff = 0.03, processors=4)
# clusters corresponds to order of seqtab.nochim

# order corresponds to order in seqtab.nochim
cat(clusters$cluster, "\n", file="/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/poyet_et_al/seqtab-nochim-gut-otus.txt")




####
### Caporaso et al
####

seqtab.nochim <- read.table("/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/caporaso_et_al/seqtab-nochim-gut.txt")
seqtab.nochim <- t(seqtab.nochim)
asv_sequences <- colnames(seqtab.nochim)

# cast them as DNAString
dna <- Biostrings::DNAStringSet(asv_sequences)
#names(dna) <- asv_sequences

metadata(dna)$name <- asv_sequences

# run the clustering at 99% similarity
clusters <- DECIPHER::Clusterize(dna, cutoff = 0.03, processors=4)
# clusters corresponds to order of seqtab.nochim

# order corresponds to order in seqtab.nochim
cat(clusters$cluster, "\n", file="/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/caporaso_et_al/seqtab-nochim-gut-otus.txt")




# use `cutoff = 0.03` for a 97% OTU
#clusters <- DECIPHER::TreeLine(myDistMatrix=d, method = "complete", cutoff = 0.03, processors = nproc)

# proceed as shown in this post above: https://github.com/benjjneb/dada2/issues/947#issuecomment-589237919
# sum-up ASVs according to the new clustering order
#merged_seqtab <- seqtab.nochim %>% t %>% rowsum(clusters$cluster) %>% t

# Optional renaming of clusters to OTU<cluster #>
#colnames(merged_seqtab) <- paste0("OTU", colnames(merged_seqtab))



# try to make output 
#newdf<-as.data.frame(cbind.data.frame(rownames(clusters),clusters$cluster,clusters$sequence))
#colnames(newdf)<-c("ASV","C.ASV","sequence") 



