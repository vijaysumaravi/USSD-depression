source ~/.bashrc

#config file paths
compare16=/path/to/opensmile/config/compare16/ComParE_2016.conf



while read line; do
        wav_file=$(echo $line | cut -f1 -d';')
        echo $wav_file
        output_file=$(echo $line | cut -f2 -d';')
        SMILExtract -C $compare16 -I $wav_file -D $output_file
done < /path/to/compare16.scp

# compare16.scp has <wav-file-path>;<output-csv-file-path> in each line
# example: /data/vijay/dataset/daic_woz_wav_processed/wav_pcm_files/322_P_audio_data.wav;/data/vijay/workdir/model_arch/data_processing_depaudionet/comparE16/feats/322_P_audio_data.csv
