set.seed(2021)

library("dplyr")
library("stringr")
library("tidyr")
library("readr")

## simply split a text file into two parts.
split_data_frame=function(a,ratio_for_testing=0.2){
    n_test <- ceiling(nrow(a)*ratio_for_testing)
    a <- a[sample(1:nrow(a),size=nrow(a),replace = FALSE),]
    
    test_data <- a[1:n_test,]
    train_data <- a[-c(1:n_test),]
    return(list(train_data=train_data,test_data=test_data))
}

## The input file is the file processed by the python script: process_mq_evidence.py
## rt must be >0;
## merge RTs for the same sequence (peptide + modification)
## RT range <= 3 (default)
## return: data.frame with columns: x, raw_file, rt_range, rt, ...
format_mq_evidence_file=function(formated_evidence_file,rt_range_cutoff=3.0){
    a <- read.delim(formated_evidence_file,stringsAsFactors =FALSE,check.names = FALSE)
    ## raw file level
    res <- a %>% group_by(x,raw_file) %>%
        filter(rt>0) %>%
        summarise(rt_range=max(rt)-min(rt),rt=mean(rt),n=n()) %>%
        ungroup() %>%
        filter(rt_range<=rt_range_cutoff)
    return(res)
}


## generate training and testing data.
generate_train_test_data=function(formated_evidence_file,
                                  ## for phosphorylation: 2|3|4
                                  ## for acetylation and ubiquitination: 2
                                  pattern_for_ptm="2|3|4",
                                  ratio_for_testing=0.2,
                                  rt_range_cutoff=3.0,
								  outdir="./"){
    set.seed(2021)
    formated_df <- format_mq_evidence_file(formated_evidence_file = formated_evidence_file,
                                           rt_range_cutoff = rt_range_cutoff)
    formated_df$is_ptm <- ifelse(str_detect(formated_df$x,pattern = pattern_for_ptm),"Y","N")
    formated_df$y = formated_df$rt
    
    d <- split_data_frame(formated_df,ratio_for_testing)
    cat("train data:",nrow(d$train_data),"\n")    
    cat("test data:",nrow(d$test_data),"\n")
    test_file <- paste0(outdir,"/","test_data.tsv")
    train_file <- paste0(outdir,"/","train_data.tsv")
    cat("train data file:",train_file,"\n")
    cat("test data file:",test_file,"\n")
    write_tsv(d$train_data,train_file)
    write_tsv(d$test_data,test_file)
}

para <- commandArgs(trailingOnly = TRUE)
formated_evidence_file <- para[1]
ratio_for_testing <- as.numeric(para[2]) 
out_dir <- para[3]

generate_train_test_data(formated_evidence_file = formated_evidence_file,
                         ratio_for_testing = ratio_for_testing,
                         outdir = out_dir)


