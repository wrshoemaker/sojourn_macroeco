from __future__ import division
import config
import os
import sys
import subprocess
import random
import re
import pickle
import gzip
from collections import Counter
import itertools
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller



import stats_utils
import numpy
#import phylo_utils
from datetime import datetime

mle_dict_path = '%smle_dict.pickle' %config.data_directory


dataset_all = ['david_et_al', 'poyet_et_al', 'caporaso_et_al']
#poyet_hosts = ['ae', 'am', 'an', 'ao']
poyet_hosts = ['ao', 'am']



n_fna_characters = 80
alpha = 0.05
min_run_length_data = 5
epsilon_fract_data = 0.1

poyet_samples_to_remove = ['SRR9218560', 'SRR9218562', 'SRR9218594', 'SRR9218655', 'SRR9218701', 'SRR9218909',
                        'SRR9218917', 'SRR9219073', 'SRR9219086', 'SRR9219088', 'SRR9219170' ,'SRR9218553',
                        'SRR9218643', 'SRR9218714', 'SRR9218936', 'SRR9219144', 'SRR9219297', 'SRR9218651', 'SRR9219141']


# dictionary of timepoints to remove in an individual host
#time_points_to_remove = {'david_et_al': {'DonorB_pre_travel': [26, 28, 100, 111, 112, 113], 'DonorB_post_travel': [229, 230, 231], 'DonorA_pre_travel': [3], 'DonorA_post_travel': [148, 216]},
#                            'poyet_et_al': {'ae':[35], 'am':[], 'an':[195], 'ao':[0]},
#                            'caporaso_et_al':{'F4':[], 'M3':[168, 234, 367]}}


time_points_to_remove = {'david_et_al': {'DonorB_pre_travel': [26, 28, 100, 111, 112, 113], 'DonorB_post_travel': [229, 230, 231], 'DonorA_pre_travel': [3], 'DonorA_post_travel': [148, 216]},
                            'poyet_et_al': {'ae':[35], 'am':[], 'an':[281], 'ao':[65]},
                            'caporaso_et_al':{'F4':[], 'M3':[168, 234, 367]}}


def move_fastq_files_to_fastq_folder(dataset):

    rootdir = '%sbarcode_data/%s/ena_files' % (config.data_directory, dataset)

    for subdir, dirs, files in os.walk(rootdir):

        fastq_file = '%s/%s' % (subdir, files[0])

        #fastq_file_destination = '%sbarcode_data/david_et_al/fastq/%s' % (config.data_directory, files[0])

        fastq_out_directory = '%sbarcode_data/%s/fastq/' % (config.data_directory, dataset)

        # move file to fastq folder
        #os.replace(fastq_file, fastq_file_destination)

        # orphaned
        filename_0 = '%s%s_0.fastq.gz' % (fastq_out_directory, file.split('.')[0])
        #R1
        filename_1 = '%s%s_1.fastq.gz' % (fastq_out_directory, file.split('.')[0])
        #R2
        filename_2 = '%s%s_2.fastq.gz' % (fastq_out_directory, file.split('.')[0])


        #os.system("split-paired-reads.py -d %s -0 %s -1 %s -2 %s --gzip %s" % (fastq_out_directory, filename_0, filename_1, filename_2, fastq_file))


#move_fastq_files_to_fastq_folder()

#fastq_out_directory = '%sbarcode_data/david_et_al/fastq/' % config.data_directory
#for subdir, dirs, files in os.walk(fastq_out_directory):

#    for file in files:

#        file_path = '%s%s' % (fastq_out_directory, file)

#        # orphaned
#        #filename_0 = '%s%s_0.fastq.gz' % (fastq_out_directory, file.split('.')[0])
#        #filename_1 = '%s%s_1.fastq.gz' % (fastq_out_directory, file.split('.')[0])
#        #filename_2 = '%s%s_2.fastq.gz' % (fastq_out_directory, file.split('.')[0])


#        #os.system("split-paired-reads.py -d %s -0 %s -1 %s -2 %s --gzip %s" % (fastq_out_directory, filename_0, filename_1, filename_2, file_path))


#for subdir, dirs, files in os.walk('/Users/williamrshoemaker/Downloads/ena_files/'):

#    fastq_file = '%s/%s' % (subdir, files[0])
#    fastq_file_destination = '%sbarcode_data/caporaso_et_al/fastq/%s' % (config.data_directory, files[0])

#    os.replace(fastq_file, fastq_file_destination)

#    print(files[0])



# split files by environment type

def split_fastq_by_environment(dataset):

    if dataset == 'david_et_al':

        metadata = '%barcode_data/david_et_al/13059_2013_3286_MOESM18_ESM.csv' % config.data_directory
        sample_to_ena_map =  '%sbarcode_data/david_et_al/filereport_read_run_PRJEB6518_tsv.txt' % config.data_directory

        sample_to_ena_dict = {}

        for line in open(sample_to_ena_map, 'r'):

            line_split = line.strip().split('\t')

            ena_id = line_split[3]
            sample_id = ".".join(line_split[7].split('/')[-1].split('_')[-1].split(".", 2)[:2])

            sample_to_ena_dict[sample_id] = ena_id


        for line in open(metadata, 'r'):

            if 'ENV_MATTER' in line:
                continue

            line_split = line.strip().split(',')
            environment_type = line_split[21].split(':')[1]
            sample_id = line_split[0]

            if environment_type == 'feces':
                new_directory = '%sbarcode_data/david_et_al/fastq_gut/' % config.data_directory

            else:
                new_directory = '%sbarcode_data/david_et_al/fastq_oral/' % config.data_directory

            old_path = '%sbarcode_data/david_et_al/fastq/%s.fastq.gz' % (config.data_directory, sample_to_ena_dict[sample_id])
            new_path = '%s%s.fastq.gz' % (new_directory, sample_to_ena_dict[sample_id])

            os.replace(old_path, new_path)


    elif dataset == 'caporaso_et_al':

        metadata = '%sbarcode_data/caporaso_et_al/mgp93_metadata_ep_host_associated.csv' % config.data_directory
        sample_to_ena_map =  '%sbarcode_data/caporaso_et_al/filereport_read_run_PRJEB19825_tsv.txt' % config.data_directory

        sample_to_ena_dict = {}

        for line in open(sample_to_ena_map, 'r'):

            if 'submitted_ftp' in line:
                continue

            line_split = line.strip().split('\t')

            ena_id = line_split[3]
            #sample_id = ".".join(line_split[7].split('/')[-1].split('_')[-1].split(".", 2)[:2])
            sample_id = line_split[7].split('/')[-1].split('.')[1]
            sample_to_ena_dict[sample_id] = ena_id


        for line in open(metadata, 'r'):

            #if 'ENV_MATTER' in line:
            #    continue

            line_split = line.strip().split(',')

            sample_id = line_split[0]
            environment_type = line_split[2]

            # set of environments
            # {'UBERON:oral cavity', 'UBERON:gut', 'UBERON:skin', 'original body habitat where the sample was obtained from', 'body_habitat'}
            if environment_type == 'UBERON:oral cavity':
                new_directory = '%sbarcode_data/caporaso_et_al/fastq_oral/' % config.data_directory

            elif environment_type == 'UBERON:gut':
                new_directory = '%sbarcode_data/caporaso_et_al/fastq_gut/' % config.data_directory

            elif environment_type == 'UBERON:skin':
                new_directory = '%sbarcode_data/caporaso_et_al/fastq_skin/' % config.data_directory

            else:
                continue

            old_path = '%sbarcode_data/caporaso_et_al/fastq/%s.fastq.gz' % (config.data_directory, sample_to_ena_dict[sample_id])
            new_path = '%s%s.fastq.gz' % (new_directory, sample_to_ena_dict[sample_id])

            os.replace(old_path, new_path)



    else:
        print('Database not recognized!')





class classFASTA:

    # class to load FASTA file

    def __init__(self, fileFASTA):
        self.fileFASTA = fileFASTA

    def readFASTA(self):
        '''Checks for fasta by file extension'''
        file_lower = self.fileFASTA.lower()
        '''Check for three most common fasta file extensions'''
        if file_lower.endswith('.txt') or file_lower.endswith('.fa') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.fna') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.frn') or \
        file_lower.endswith('.faa') or file_lower.endswith('.ffn'):
            with open(self.fileFASTA, "r") as f:
                return self.ParseFASTA(f)
        else:
            print("Not in FASTA format.")

    def ParseFASTA(self, fileFASTA):
        '''Gets the sequence name and sequence from a FASTA formatted file'''
        fasta_list=[]
        for line in fileFASTA:
            if line[0] == '>':
                try:
                    fasta_list.append(current_dna)
            	#pass if an error comes up
                except UnboundLocalError:
                    #print "Inproper file format."
                    pass
                current_dna = [line.lstrip('>').rstrip('\n'),'']
            else:
                current_dna[1] += "".join(line.split())
        fasta_list.append(current_dna)
        '''Returns fasa as nested list, containing line identifier \
            and sequence'''
        return fasta_list




def make_fasta_from_dada2_output(dataset, environment):

    outgroup = classFASTA('%sbarcode_data/outgroup.fna' % config.data_directory).readFASTA()

    dada2_path = "%sbarcode_data/%s/seqtab-nochim-%s.txt" % (config.data_directory, dataset, environment)
    fasta_path = "%sbarcode_data/%s/seqtab-nochim-%s.fna" % (config.data_directory, dataset, environment)
    fasta_with_outgroup_path = "%sbarcode_data/%s/seqtab-nochim-%s-with-outgroup.fna" % (config.data_directory, dataset, environment)
    fasta_file = open(fasta_path, 'w')
    fasta_with_outgroup_file = open(fasta_with_outgroup_path, 'w')

    # write outgroup
    fasta_with_outgroup_file.write('>%s\n' % outgroup[0][0])
    for i in range(0, len(outgroup[0][1]), n_fna_characters):
        sequence_i = outgroup[0][1][i : i + n_fna_characters]
        fasta_with_outgroup_file.write('%s\n' % sequence_i)
    fasta_with_outgroup_file.write('\n')

    for line_idx, line in enumerate(open(dada2_path, 'r')):

        if line_idx == 0:
            continue

        line = line.strip()

        sequence = line.split('\t')[0]
        fasta_file.write('>%s\n' % sequence)
        fasta_with_outgroup_file.write('>%s\n' % sequence)

        for i in range(0, len(sequence), n_fna_characters):
            sequence_i = sequence[i : i + n_fna_characters]
            fasta_file.write('%s\n' % sequence_i)
            fasta_with_outgroup_file.write('%s\n' % sequence_i)
        fasta_file.write('\n')
        fasta_with_outgroup_file.write('\n')

    fasta_file.close()
    fasta_with_outgroup_file.close()



def get_dada2_data(dataset, environment):

    metadata_dict = {}
    sample_to_ena_dict = {}
    sample_to_metadata_dict = {}

    n_days = 0
    if dataset == 'david_et_al':

        metadata_path = '%sbarcode_data/david_et_al/13059_2013_3286_MOESM18_ESM.csv' % config.data_directory
        sample_to_ena_map_path =  '%sbarcode_data/david_et_al/filereport_read_run_PRJEB6518_tsv.txt' % config.data_directory

        metadata = open(metadata_path, 'r')
        header_metadata = metadata.readline()
        header_metadata = header_metadata.strip().split(',')

        sample_to_ena_map = open(sample_to_ena_map_path, 'r')
        header_sample_to_ena = metadata.readline()
        header_sample_to_ena = header_sample_to_ena.strip().split(',')

        for line in sample_to_ena_map:

            line_split = line.strip().split('\t')
            ena = line_split[3]
            # lazy......not using regex......
            sample = ".".join(line_split[7].split('/')[-1].split('_')[1].split('.',2)[:2])
            sample_to_ena_dict[sample] = ena

        # parse metadata
        for line in metadata:

            line_split = line.strip().split(',')
            sample = line_split[0]
            # data is stored as "days" instead of a date-time format

            days = int(line_split[12])
            host = line_split[25].split(':')[1].strip()
            ena = sample_to_ena_dict[sample]

            if environment == 'gut':

                # ignore if data is not from the gut
                if line_split[6].split(':')[1] != 'feces':
                    continue

            # split timeseries for David et al timeseries
            # A = pre-travel = 0-70, post-travel = 123 - 364
            # B = pre-salmonella = 0-150, post-salmonella = 160-252

            if host == 'DonorA':

                if (days > 70) and (days < 123):
                    continue

                else:
                    # relabel
                    if days <= 70:
                        host = 'DonorA_pre_travel'

                    else:
                        host = 'DonorA_post_travel'


            else:

                if (days > 150) and (days < 160):
                    continue

                else:
                    # relabel
                    if days <= 150:
                        host = 'DonorB_pre_travel'

                    else:
                        host = 'DonorB_post_travel'


            # there are a few timepoints in a given host that was sampled multiple times
            # we will choose the sample with the higher total coverage and ignore the other

            #['ERR531797', 'ERR531566']
            #[377674 152164]
            #['ERR531855', 'ERR531547']
            #[263921 292824]
            #['ERR531802', 'ERR532085']
            #[ 71337 142181]
            #['ERR531887', 'ERR532078']
            #[346388 198347]
            #['ERR531990', 'ERR532100']
            #[93157 93888]

            ena_to_ignore = ['ERR531566', 'ERR531855', 'ERR531802', 'ERR532078', 'ERR531990']

            if ena in ena_to_ignore:
                continue

            sample_to_metadata_dict[ena] = {}
            sample_to_metadata_dict[ena]['host'] = host
            sample_to_metadata_dict[ena]['days'] = days



    elif dataset == 'caporaso_et_al':

        metadata = '%sbarcode_data/caporaso_et_al/mgp93_metadata_ep_host_associated.csv' % config.data_directory
        metadata_2 = '%sbarcode_data/caporaso_et_al/mgp93_metadata.csv' % config.data_directory
        sample_to_ena_map =  '%sbarcode_data/caporaso_et_al/filereport_read_run_PRJEB19825_tsv.txt' % config.data_directory

        # make sample to ENA map
        for line in open(sample_to_ena_map, 'r'):

            if 'submitted_ftp' in line:
                continue

            line_split = line.strip().split('\t')

            ena_id = line_split[3]
            sample_id = line_split[7].split('/')[-1].split('.')[1]
            sample_to_ena_dict[sample_id] = ena_id

        collection_date_all = []
        for line in open(metadata_2, 'r'):

            if ('host_individual' in line) or ('wastewater|sludge' in line):
                continue

            line_split = line.strip().split(',')

            host = line_split[27]
            days_since_epoch = line_split[26]
            #collection_time = line_split[12]
            collection_date = line_split[3]
            sample = line_split[0]

            # environment
            environment_line = line_split[-6]

            if environment_line != 'feces':
                continue

            collection_date = datetime.strptime(collection_date, '%Y-%m-%d')
            if host not in metadata_dict:
                metadata_dict[host] = {}

            metadata_dict[host][sample_to_ena_dict[sample]] = {}
            metadata_dict[host][sample_to_ena_dict[sample]]['collection_date'] = collection_date
            collection_date_all.append(collection_date)



    elif dataset == 'poyet_et_al':

        # metadat with timepoints....
        metadata_path = '%sbarcode_data/poyet_et_al/Poyet_metadata_full.csv' % config.data_directory
        #sample_to_ena_map =  '%s/barcode_data/poyet_et_al/filereport_read_run_PRJEB6518_tsv.txt' % config.data_directory

        metadata = open(metadata_path, 'r')
        for line in metadata:
            line_split = line.strip().split(',')

            if line_split[1] != 'AMPLICON':
                continue

            sample_srr = line_split[0]
            collection_date = line_split[8]
            host = line_split[-4].split('-')[0]

            # poyet_et_al criteria for rare (but annoying) long sampling intervals


            # skip hosts we dont care about
            if host not in poyet_hosts:
                continue

            if collection_date == 'missing':
                continue

            if host not in metadata_dict:
                metadata_dict[host] = {}


            collection_date = datetime.strptime(collection_date, '%d-%b-%Y')

            # dictionary contains ALL samples, regardless of whether they pass dada2 filters
            metadata_dict[host][sample_srr] = {}
            metadata_dict[host][sample_srr]['host'] = host
            metadata_dict[host][sample_srr]['collection_date'] = collection_date

        metadata.close()

    else:

        print('Dataset not recognized!')

    if (dataset == 'caporaso_et_al') or (dataset == 'poyet_et_al'):
        # turn dates to days
        # get start date
        # for all hosts
        start_collection_date_all = []
        for host, host_dict in metadata_dict.items():
            start_collection_date_all.extend([host_dict[s]['collection_date'] for s in host_dict.keys()])

        start_collection_date = min(start_collection_date_all)
        #start_collection_date = min(collection_date_all)
        for host, host_dict in metadata_dict.items():

            # get start date
            #start_collection_date = min(host_dict[s]['collection_date'] for s in host_dict.keys())

            #print(host)

            # get number of days
            for host_sample in host_dict.keys():

                if (dataset == 'poyet_et_al'):
                    if host_sample in poyet_samples_to_remove:
                        continue

                days = host_dict[host_sample]['collection_date'] - start_collection_date

                if (dataset == 'poyet_et_al'):
                    if host == 'am':
                        #### removing excess sampling interval regions
                        # [104, 406]
                        if (days.days < 104) or (days.days > 406): 
                            continue
                
                sample_to_metadata_dict[host_sample] = {}
                sample_to_metadata_dict[host_sample]['days'] = days.days
                sample_to_metadata_dict[host_sample]['host'] = host


    # load DADA2 data
    data_path = '%sbarcode_data/%s/seqtab-nochim-%s.txt' % (config.data_directory, dataset, environment)
    data_taxa = '%sbarcode_data/%s/seqtab-nochim-taxa-%s.txt' % (config.data_directory, dataset, environment)

    data = open(data_path, 'r')
    samples_data = data.readline()
    # ENA samples from dada2
    samples_data = numpy.asarray(samples_data.strip().split('\t'))

    asv_all = []
    read_counts_all = []
    for line in data:
        line_split = line.strip().split('\t')
        asv_all.append(line_split[0])
        read_counts_all.append(line_split[1:])

    data.close()


    asv_all = numpy.asarray(asv_all)
    read_counts_all = numpy.asarray(read_counts_all)
    read_counts_all = numpy.array(read_counts_all, dtype=int)

    # get samples that are in the metadata dict AND in the dada2 data
    samples_data_inter = numpy.asarray(list(set(samples_data.tolist()) & set(sample_to_metadata_dict.keys())))

    days_all = numpy.asarray([sample_to_metadata_dict[h]['days'] for h in samples_data_inter])
    host_all = numpy.asarray([sample_to_metadata_dict[h]['host'] for h in samples_data_inter])

    # sort by days AND host
    list_zip_days_host = list(zip(days_all.tolist(), host_all.tolist()))
    sort_list_zip_days_host = list(sorted(zip(days_all.tolist(), host_all.tolist())))
    days_and_host_all_sort_idx = numpy.asarray([list_zip_days_host.index(z) for z in sort_list_zip_days_host])

    days_all_sort = days_all[days_and_host_all_sort_idx]
    host_all_sort = host_all[days_and_host_all_sort_idx]
    samples_data_inter_sort = samples_data_inter[days_and_host_all_sort_idx]

    # read_counts_all needs to be sorted by ENA sample ID
    idx_to_keep_read_counts = numpy.asarray([numpy.where(samples_data==s)[0][0] for s in samples_data_inter_sort])
    read_counts_all_sort = read_counts_all[:,idx_to_keep_read_counts]

    # remove absent species
    asv_to_keep_idx = (read_counts_all_sort.sum(axis=1) > 0)
    read_counts_all_sort = read_counts_all_sort[asv_to_keep_idx,:]
    asv_all_sort = asv_all[asv_to_keep_idx]

    # david et al has a few samples with very low read counts, remove them
    idx_samples_to_keep = (read_counts_all_sort.sum(axis=0) >= 100)
    read_counts_all_sort_filter = read_counts_all_sort[:,idx_samples_to_keep]
    host_all_sort_filter = host_all_sort[idx_samples_to_keep]
    days_all_sort_filter = days_all_sort[idx_samples_to_keep]

    # david_et_al one sample with outlier timepoints (>50 days), remove this sample
    if dataset == 'david_et_al':

        idx_to_remove = (host_all_sort_filter == 'DonorB_post_travel') & (days_all_sort_filter>=318)
        days_all_sort_filter = days_all_sort_filter[~(idx_to_remove)]
        host_all_sort_filter = host_all_sort_filter[~(idx_to_remove)]
        read_counts_all_sort_filter = read_counts_all_sort_filter[:,~(idx_to_remove)]


    # remove the few samples with distances = 1, likely technical artifact
    hosts_set = list(set(host_all_sort_filter.tolist()))
    hosts_set.sort()
    samples_to_remove = []
    for host in hosts_set:
        days_to_remove = time_points_to_remove[dataset][host]
        samples_to_remove.extend([numpy.where(((host_all_sort_filter==host) & (days_all_sort_filter==d)) == True)[0][0] for d in days_to_remove])

    # find duplicate samples (same day, same host, same data) i.e., not technical replicates
    # there are only duplicates, no triplicates etc.
    # this is only necessary for poyet_et_al
    #if dataset == 'poyet_et_al':
    #    for host in hosts_set:

    #        days, days_counts = numpy.unique(days_all_sort_filter[(host_all_sort_filter==host)], return_counts=True)
    #        days_with_duplicates = days[days_counts==2]
    #        samples_to_remove.extend([numpy.where(((host_all_sort_filter==host) & (days_all_sort_filter==d)) == True)[0][0] for d in days_with_duplicates])
    

    # isolate poyet_et_al host am sub-timeseries with sufficient number of samples without reasonable samping intervals
    #print(host)
    #if (dataset == 'poyet_et_al') and (host == 'am'):

    #    print(days_all_sort_filter)  



    # remove redundant objects from list
    samples_to_remove = list(set(samples_to_remove))
    samples_to_remove.sort()
    # delete samples
    samples_to_remove = numpy.asarray(samples_to_remove)
    read_counts_all_sort_filter = numpy.delete(read_counts_all_sort_filter, samples_to_remove, axis=1)
    days_all_sort_filter = numpy.delete(days_all_sort_filter, samples_to_remove)
    host_all_sort_filter = numpy.delete(host_all_sort_filter, samples_to_remove)

    # remove absent ASVs
    asv_to_keep_final_idx = (read_counts_all_sort_filter.sum(axis=1) > 0)
    read_counts_all_sort_filter = read_counts_all_sort_filter[asv_to_keep_final_idx,:]
    asv_all_sort = asv_all_sort[asv_to_keep_final_idx]


    return read_counts_all_sort_filter, host_all_sort_filter, days_all_sort_filter, asv_all_sort








def make_otu_dict(dataset, environment):

    data_path = '%sbarcode_data/%s/seqtab-nochim-%s.txt' % (config.data_directory, dataset, environment)
    otu_path = '%sbarcode_data/%s/seqtab-nochim-%s-otus.txt' % (config.data_directory, dataset, environment)

    data_file = open(data_path, 'r')
    otu_file = open(otu_path, 'r')
    
    otu_ = otu_file.readline().strip().split(' ')
    otu_ = numpy.asarray([int(o) for o in otu_])

    data_header = data_file.readline()
    asv_all = []
    for line in data_file:
        asv_all.append(line.strip().split('\t')[0])

    data_file.close()
    otu_file.close()


    otu_to_asv_dict = {}
    for asv_idx in range(len(otu_)):
        #key_ = 'OTU_%d' % o_idx

        otu = otu_[asv_idx]
        asv = asv_all[asv_idx]

        if otu not in otu_to_asv_dict:
            otu_to_asv_dict[otu] = []

        otu_to_asv_dict[otu].append(asv)

    return otu_to_asv_dict



def fit_taylors_law_linregress(s_by_s, max_mean=1):

    # assumes base log10

    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))

    mean = numpy.mean(rel_s_by_s, axis=1)
    var = numpy.var(rel_s_by_s, axis=1)

    to_keep_idx = (mean <= max_mean)

    mean = mean[to_keep_idx]
    var = var[to_keep_idx]

    slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(mean), numpy.log10(var))

    return slope, intercept, r_valuer_value, p_value, std_err



def subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host):

    # subset read matrix
    samples_to_keep_idx = (host_status==host)
    read_counts_host = read_counts[:,samples_to_keep_idx]
    days_host = days[samples_to_keep_idx]

    # identify ASVs that are present in this host
    asv_to_keep_idx = (read_counts_host.sum(axis=1) > 0)

    read_counts_host = read_counts_host[asv_to_keep_idx,:]
    asv_names_host = asv_names[asv_to_keep_idx]

    return read_counts_host, days_host, asv_names_host


def make_survival_dist(data, range_):


    data = data[numpy.isfinite(data)]
    survival_array = [sum(data>=i)/len(data) for i in range_]
    #survival_array = [sum(data>=i)/len(data) for i in range_]
    survival_array = numpy.asarray(survival_array)

    return survival_array



def calculate_confidence_intervals_from_survival_dist(null_matrix, alpha=0.05):

    # assumes shape of (# iterations, #variables)
    # calculates lower and upper CIs for each variable

    n_iter = null_matrix.shape[0]

    lower_ci_all = []
    upper_ci_all = []

    for array_i in null_matrix.T:

        array_i = numpy.sort(array_i)

        lower_ci_i = array_i[int((alpha/2)*n_iter)]
        upper_ci_i = array_i[int((1-(alpha/2))*n_iter)]

        lower_ci_all.append(lower_ci_i)
        upper_ci_all.append(upper_ci_i)

    #(alpha/2)

    lower_ci_all = numpy.asarray(lower_ci_all)
    upper_ci_all = numpy.asarray(upper_ci_all)
    
    return lower_ci_all, upper_ci_all



def calculate_confidence_interval_array(array_, alpha=0.05):

    n_iter = len(array_)

    if type(array_) == type([]):
        array_ = numpy.asarray(array_)

    array_ = numpy.sort(array_)

    lower_ci = array_[int((alpha/2)*n_iter)]
    upper_ci = array_[int((1-(alpha/2))*n_iter)]

    return lower_ci, upper_ci



def estimate_k_and_sigma_old(s_by_s):

    n_reads = s_by_s.sum(axis=0)

    rel_s_by_s = s_by_s/n_reads

    mean = numpy.mean(rel_s_by_s, axis=1)

    # Eq. 3 in Zaoli and Grilli, 2022
    var_first_term = numpy.mean(((s_by_s**2) - s_by_s)/((n_reads**2) - n_reads), axis=1)
    var = var_first_term - mean**2

    #var = numpy.var(rel_s_by_s, axis=1)

    # no negative 
    positive_idx = var > 0

    mean_subet = mean[positive_idx]
    var_subet = var[positive_idx]
    cv_squared_subset = var_subet/(mean_subet**2)

    # mean = K(1 - sigma/2) = K*(2-sigma)/2
    # var = sigma*(mean**2)/(2-sigma)
    # CV**2 = var/(mean**2) = sigma/(2 - sigma)
    # sigma = 2*(CV**2)/(1 + (CV**2))
    # K = (2*mean)/(2-sigma)

    sigma = 2*cv_squared_subset/(1+cv_squared_subset)

    # cannot have sigma *greater than or equal to* 2
    # subset sigma
    # parameter range where sigma is meaningful
    sigma_filter_idx = (sigma > 0) & (sigma < 2)
    #sigma_filter_idx =  (sigma < 2)

    sigma_subset = sigma[sigma_filter_idx]
    mean_subet = mean_subet[sigma_filter_idx]
    k_subset = (2*mean_subet)/(2 - sigma_subset)


    return k_subset, sigma_subset
   


def estimate_k_and_sigma(s_by_s):

    n_reads = s_by_s.sum(axis=0)

    rel_s_by_s = s_by_s/n_reads

    occupancy = numpy.sum(rel_s_by_s>0, axis=1)/len(n_reads)

    mediasq = numpy.mean(numpy.divide(s_by_s.T*(s_by_s.T - numpy.ones(numpy.shape(s_by_s.T))), (n_reads*(n_reads-1))[:,None]), axis=0 )  
    meanrelabd = numpy.mean(rel_s_by_s, axis=1) 


    temp = 1 + meanrelabd**2/(mediasq-meanrelabd**2)  
    sigma = numpy.where((temp!=0) & (~numpy.isnan(temp)), 2/temp, numpy.nan) 
    k = 2*meanrelabd/(2-sigma) 
    ids = numpy.nonzero((sigma>0) & (occupancy>0.2)) 
    ids2 = numpy.nonzero(sigma>0)
    
    return k, sigma, ids, ids2, meanrelabd
   
    

def calculate_logfold_matrix(rel_s_by_s, days):

    rel_s_by_s_log10 = numpy.log10(rel_s_by_s)
    delta_l = rel_s_by_s_log10[:,1:] - rel_s_by_s_log10[:,:-1]
    # divide by days between sampling events (\delta t)
    delta_l_delta_t = delta_l / (days[1:] - days[:-1])

    return delta_l_delta_t




def find_runs(x, min_run_length=1):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = numpy.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return numpy.array([]), numpy.array([]), numpy.array([])

    else:
        # find run starts
        loc_run_start = numpy.empty(n, dtype=bool)
        loc_run_start[0] = True
        numpy.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = numpy.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = numpy.diff(numpy.append(run_starts, n))

        to_keep_idx = (run_lengths>=min_run_length)

        run_values_subset = run_values[to_keep_idx]
        run_starts_subseet = run_starts[to_keep_idx]
        run_lengths_subset = run_lengths[to_keep_idx]

        return run_values_subset, run_starts_subseet, run_lengths_subset



def extract_trajectory_epsilon(x_deviation, run_value, run_start, run_length, epsilon=None):

    # cannot have a sojourn time that is the length of the entire timeseries
    # weird fluctuations here, ignore..
    if (run_length >= len(x_deviation-1)):
        run_deviation = None

    if epsilon != None:
            
        # avoid issue of walk starting at zero
        # inclusive
        start_before_idx = max([(run_start-1), 0])
        start_after_idx = run_start
        #end_before_idx = run_start + run_length - 1
        #end_after_idx = min([(len(x_deviation)-1), (run_start + run_length)])
        end_before_idx = run_start + run_length
        # this value will be a different sign
        end_after_idx = min([(len(x_deviation)), (run_start + run_length+1)])

        start_before = abs(x_deviation[start_before_idx])
        start_after = abs(x_deviation[start_after_idx])
        # inclusive
        end_before = abs(x_deviation[end_before_idx-1])
        end_after = abs(x_deviation[end_after_idx-1])

        #print(start_before, start_after, end_before, end_after)

        start_before_bool = False
        start_after_bool = False
        end_before_bool = False
        end_after_bool = False
        
        if start_before <= epsilon:
            start_before_bool = True 

        if start_after <= epsilon:
            start_after_bool = True 

        if end_before <= epsilon:
            end_before_bool = True 

        if end_after <= epsilon:
            end_after_bool = True 

        # continue, not within epsilon at either end of the trajectory
        if ((start_before_bool+start_after_bool) == 0) or ((end_before_bool+end_after_bool) == 0):
            run_deviation = None

        else:
            
            #print(start_before, start_after, end_before, end_after, epsilon)
            #print(start_before_bool, start_after_bool, end_before_bool, end_after_bool)

            # Baldassarri used the first timepoint that reached epsilon 
            # inclusive
            # find the smallest value..
            if start_before_bool == True:
                new_run_start = start_before_idx
            # possible values (False, True)
            else:
                new_run_start = start_after_idx
            

            # first timepoint at end that reached epsilon
            # exclusive
            # should I make it exclusive?
            if end_before_bool == True:
                new_run_end = end_before_idx
            else:
                new_run_end = end_after_idx


            if new_run_start == new_run_end:
                run_deviation = None

            else:
                run_deviation = x_deviation[new_run_start:new_run_end]
                
                # negative deviation from the origin
                if run_value == False:
                    run_deviation = -1*run_deviation

            #print(x_deviation[run_start:run_start+run_length])

            # make sure start and end are within epsilon 
            if (abs(run_deviation[0]) > epsilon) or (abs(run_deviation[-1]) > epsilon):
                print("Error!")
        
            #if (start_before_bool == True) and (end_before_bool == False):
            #if len(run_deviation) != run_length:
            #    print( start_before_bool, start_after_bool, end_before_bool, end_after_bool )

    else:
        run_deviation = numpy.absolute(x_deviation[run_start:(run_start + run_length)])


    
    return run_deviation



def calculate_deviation_pattern_data(x_trajectory, x_0, days_array, min_run_length=min_run_length_data, epsilon=None, return_array=True):

    x_deviation = x_trajectory - x_0

    # at least min_run_length observations
    run_values, run_starts, run_lengths = find_runs(x_deviation>0, min_run_length=min_run_length)
    # get days
    run_values_new, run_starts_new, run_lengths_new, days_run_lengths, days_run_starts = run_lengths_to_days(run_values, run_starts, run_lengths, days_array)
    run_dict = {}

    for run_j_idx in range(len(run_values_new)):
        
        run_deviation_j = extract_trajectory_epsilon(x_deviation, run_values_new[run_j_idx], run_starts_new[run_j_idx], run_lengths_new[run_j_idx], epsilon=epsilon)
        
        if run_deviation_j is None:
            continue

        #run_lengths_new_j = len(run_deviation_j)
        # we want sojourn time in **days**
        run_lengths_new_j = days_run_lengths[run_j_idx]
        if run_lengths_new_j not in run_dict:
            run_dict[run_lengths_new_j] = []

        if return_array == False:
            run_deviation_j = run_deviation_j.tolist()

        run_dict[run_lengths_new_j].append(run_deviation_j)


    return run_dict



def make_list_from_count_dict(dict_):

    counts = []
    for key, value in dict_.items():
        counts.extend([key]*value)

    return counts



def calculate_mean_from_count_dict(dict_):

    sum_of_numbers = sum(number*count for number, count in dict_.items())
    count = sum(count for n, count in dict_.items())
    mean = sum_of_numbers / count

    return mean




def get_hist_and_bins(flat_array, n_bins=20, min_n_points=3):

    # make sure its an array
    flat_array = numpy.asarray(flat_array)
    flat_array = flat_array[~numpy.isnan(flat_array)]

    # null is too large, so we are binning it for the plot in this script
    hist_, bin_edges_ = numpy.histogram(flat_array, density=False, bins=n_bins)
    bins_mean_ = numpy.asarray([0.5 * (bin_edges_[i] + bin_edges_[i+1]) for i in range(0, len(bin_edges_)-1 )])
        
    hist_to_plot = hist_[hist_>0]
    bins_mean_to_plot = bins_mean_[hist_>0]

    hist_to_plot = hist_to_plot/sum(hist_to_plot)
    

    return hist_to_plot, bins_mean_to_plot



def run_lengths_to_days(run_values, run_starts, run_lengths, days_array):

    days_run_starts = []
    days_run_lengths = []
    # skip first and last since we do not know when sojourns end
    for run_j_idx in range(1, len(run_values)-1):

        run_start_j = run_starts[run_j_idx]
        run_end_j = run_starts[run_j_idx] + run_lengths[run_j_idx] + 1
        
        # end of timeseries
        #if run_end_j > len(days_array):
        #    continue
        days_run_j = days_array[run_start_j:run_end_j]

        # only one observation, skip
        #if len(days_run_j) == 1:
        #    continue

        days_run_starts.append(days_array[run_start_j])
        days_run_lengths.append(int(days_run_j[-1] - days_run_j[0]))

    run_values_new = run_values[1:-1]
    run_starts_new = run_starts[1:-1]
    run_lengths_new = run_lengths[1:-1]
    days_run_starts = days_run_starts[1:-1]


    return run_values_new, run_starts_new, run_lengths_new, days_run_lengths, days_run_starts






def make_mle_dict(epsilon_fract=epsilon_fract_data, min_run_length_data=min_run_length_data):

    mle_dict = {}
    mle_dict['params'] = {}
    mle_dict['params']['epsilon_fract'] = epsilon_fract
    mle_dict['params']['min_run_length_data'] = min_run_length_data

    for dataset in dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        mle_dict[dataset] = {}

        read_counts, host_status, days, asv_names = get_dada2_data(dataset, 'gut')

        host_all = list(set(host_status))
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            mle_dict[dataset][host] = {}

            # function subsets ASVs that are actually present
            read_counts_host, days_host, asv_names_host = subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)

            #if host == 'am':
            #    print(days_host)

            rel_read_counts_host = (read_counts_host/read_counts_host.sum(axis=0))
            total_abundance = numpy.sum(read_counts_host, axis=0)

            for asv_names_host_subset_i_idx, asv_names_host_subset_i in enumerate(asv_names_host):

                abundance_trajectory = read_counts_host[asv_names_host_subset_i_idx,:]

                # ignore ASVs with occupancy < 1
                if sum(abundance_trajectory==0) > 0:
                    continue

                rel_abundance_trajectory = abundance_trajectory/total_abundance

                # gamma MLE paramss
                gamma_sampling_model = stats_utils.mle_gamma_sampling(total_abundance, abundance_trajectory)
                mu_start = numpy.mean(abundance_trajectory/total_abundance)
                sigma_start = numpy.std(abundance_trajectory/total_abundance)
                start_params = numpy.asarray([mu_start, sigma_start])
                gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", start_params=start_params, bounds= [(0.000001,1), (0.00001,100)], full_output=False, disp=False)
                x_mean, x_std = gamma_sampling_result.params

                # get sojourn times for runs of all lengths
                #x_deviation = rel_abundance_trajectory - x_mean

                log_rescaled_rel_abundance_trajectory = numpy.log(rel_abundance_trajectory/x_mean)
                cv_asv = x_std/x_mean
                expected_value_log_gamma = stats_utils.expected_value_log_gamma(1, cv_asv)
                run_values, run_starts, run_lengths = find_runs((log_rescaled_rel_abundance_trajectory - expected_value_log_gamma)>0, min_run_length=1)

                run_values_new, run_starts_new, run_lengths_new, days_run_lengths, days_run_starts = run_lengths_to_days(run_values, run_starts, run_lengths, days_host)  

                if len(days_run_lengths) == 0:
                    continue

                mle_dict[dataset][host][asv_names_host_subset_i] = {}
                mle_dict[dataset][host][asv_names_host_subset_i]['rel_abundance'] = rel_abundance_trajectory.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['abundance'] = abundance_trajectory.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['total_abundance'] = total_abundance.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['days'] = days_host.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['x_mean'] = x_mean
                mle_dict[dataset][host][asv_names_host_subset_i]['x_std'] = x_std
                mle_dict[dataset][host][asv_names_host_subset_i]['run_starts'] = run_starts_new.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['run_lengths'] = run_lengths_new.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['days_run_lengths'] = days_run_lengths
                mle_dict[dataset][host][asv_names_host_subset_i]['days_run_values'] = run_values_new.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['days_run_starts'] = days_run_starts
                run_dict = calculate_deviation_pattern_data(log_rescaled_rel_abundance_trajectory, expected_value_log_gamma, days_host, min_run_length=min_run_length_data, epsilon=epsilon_fract_data, return_array=False)
                

                # run dickey-fuller test on log_rescaled_rel_abundance_trajectory
                #dickey_fuller_ = adfuller(rel_abundance_trajectory)
                ##mle_dict[dataset][host][asv_names_host_subset_i]['dickey_fuller_stat'] = dickey_fuller_[0]
                #mle_dict[dataset][host][asv_names_host_subset_i]['dickey_fuller_p_value'] = dickey_fuller_[1]

                #print(log_rescaled_rel_abundance_trajectory)
                #print(dickey_fuller_[1])

                # max possible sojourn time
                mle_dict[dataset][host][asv_names_host_subset_i]['max_possible_sojourn_time'] = max(days_host) - min(days_host)

                if len(run_dict) == 0:
                    mle_dict[dataset][host][asv_names_host_subset_i]['run_dict'] = None
                else:
                    mle_dict[dataset][host][asv_names_host_subset_i]['run_dict'] = run_dict

                #if len(run_dict) != 0:
                #    print(run_dict)



    sys.stderr.write("Saving dictionary...\n")
    with open(mle_dict_path, 'wb') as outfile:
        pickle.dump(mle_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def calculate_metadata_stats():

    mle_dict = pickle.load(open(mle_dict_path, "rb"))
    

    sys.stderr.write("Dataset, Host, # samples, # days, Mean # days b/w samples, Mean # reads, # ASVs used\n")

    for dataset in dataset_all:
        
        read_counts, host_status, days, asv_names = get_dada2_data(dataset, 'gut')

        host_all = list(set(host_status))
        host_all.sort()

        if dataset == 'david_et_al':
            host_all = ['DonorA_pre_travel', 'DonorB_pre_travel']

        # Num. samples
        # Num. # days
        # mean # days between samples
        # ASVs present in all samples
        # mean # reads per sample

        #n_samples_all = []


        for host in host_all:

            if dataset == 'david_et_al':
                host_post = '%s_post_travel' % host.split('_')[0]
                read_counts_host_pre, days_host_pre, asv_names_host_pre = subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)
                read_counts_host_post, days_host_post, asv_names_host_post = subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host_post)

                n_samples = len(days_host_pre) + len(days_host_post)
                n_days = max(days_host_post) - min(days_host_pre)
                n_asvs = len(set(list(mle_dict[dataset][host].keys())) | set(list(mle_dict[dataset][host_post].keys())))
                n_reads_per_sample = (numpy.mean(numpy.sum(read_counts_host_pre, axis=0)) + numpy.mean(numpy.sum(read_counts_host_post, axis=0)))/2


            else:
                # function subsets ASVs that are actually present
                read_counts_host, days_host, asv_names_host = subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)


                n_samples = len(days_host)
                n_days = max(days_host) - min(days_host)
                n_asvs = len(set(list(mle_dict[dataset][host].keys())))
                n_reads_per_sample = numpy.mean(numpy.sum(read_counts_host, axis=0))

            
            
            n_days_per_sample = n_days/n_samples

            sys.stderr.write("%s, %s, %d, %d, %.2f, %.2f, %d\n" % (dataset, host, n_samples, n_days, n_days_per_sample, n_reads_per_sample, n_asvs))

            #print(n_days, n_days_per_sample, n_asvs, n_reads_per_sample)




def chunk_list(flat_list, chunk_size):
    return [flat_list[i:i + chunk_size] for i in range(0, len(flat_list), chunk_size)]



if __name__ == "__main__":

    print("Running...")

    make_mle_dict()

    #calculate_metadata_stats()