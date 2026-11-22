#! /bin/bash
# controls the training of the wavtovec model. This is identical to the Raw audio model but uses wav2vec embedded audio vectors instead.

test ()
{
        # tests the performance of the previously trained model first argument is the name of the model, the second specifies the testing batch
        rm -r results/Wav2vecModel/hypothesis
        mkdir results/Wav2vecModel/hypothesis
        python neural_network/TestNetwork.py RawModel data/dev/wav2vec results/Wav2vecModel/hypothesis data/dev/table.csv results/Wav2vecModel 0
}

train ()
{
        python neural_network/TrainNetwork.py RawModel data/train/wav2vec data/train/balanced.csv results/Wav2vecModel/errorlog.txt 25 -r
        rm -r results/Wav2vecModel/hypothesis
        mkdir results/Wav2vecModel/hypothesis
        python neural_network/TestNetwork.py RawModel data/dev/wav2vec results/Wav2vecModel/hypothesis data/dev/table.csv results/Wav2vecModel 0
        rm -r results/Wav2vecModel/initial
        mkdir results/Wav2vecModel/initial
        mv results/Wav2vecModel/{errorlog,hypothesis_analysis,statistics}.txt results/Wav2vecModel/initial
        mv results/Wav2vecModel/hypothesis results/Wav2vecModel/initial

        for ((i=0;i<$1;++i));
                do
                python neural_network/TrainNetwork.py RawModel data/train/wav2vec data/train/balanced.csv results/Wav2vecModel/errorlog.txt 25
                rm -r results/Wav2vecModel/hypothesis
                mkdir results/Wav2vecModel/hypothesis
                python neural_network/TestNetwork.py RawModel data/dev/wav2vec results/Wav2vecModel/hypothesis data/dev/table.csv results/Wav2vecModel $i
                reset Wav2vecModel $i;
                done
        echo -e '\a'
}

reset ()
{
        # helper code that resets the training session for a model so that it can be retrained
        rm -r results/$1/"session"$2
        mkdir results/$1/"session"$2
        cp results/$1/{errorlog.txt,hypothesis_analysis.txt,statistics.txt} results/$1/"session"$2
        mkdir results/$1/"session"$2/hypothesis
        cp results/$1/hypothesis/* results/$1/"session"$2/hypothesis

}


train 100

