wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  


for SPLIT in train dev; do
    python -m fairseq.examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "data/$SPLIT.input0" \
        --outputs "data/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done


fairseq-preprocess \
    --only-source \
    --trainpref "data/train.input0.bpe" \
    --validpref "data/dev.input0.bpe" \
    --destdir "data-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "data/train.label" \
    --validpref "data/dev.label" \
    --destdir "data-bin/label" \
    --workers 60