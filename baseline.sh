#!/bin/bash
# controls the training of the baselinemodel which takes slightly different input settings.

test ()
{
        # tests the performance of the previously trained model first argument is the name of the model, the second specifies the testing batch
        rm -r results/RawModel/hypothesis
        mkdir results/RawModel/hypothesis
        python neural_network/TestNetwork.py RawModel data/dev/Raw results/RawModel/hypothesis data/dev/table.csv results/RawModel 0
}

train ()
{
        python neural_network/TrainNetwork.py RawModel data/train/Raw data/train/balanced.csv results/RawModel/errorlog.txt 25 -r
        rm -r results/RawModel/hypothesis
        mkdir results/RawModel/hypothesis
        python neural_network/TestNetwork.py RawModel data/dev/Raw results/RawModel/hypothesis data/dev/table.csv results/RawModel 0
        rm -r results/RawModel/initial
        mkdir results/RawModel/initial
        mv results/RawModel/{errorlog,hypothesis_analysis,statistics}.txt results/RawModel/initial
        mv results/RawModel/hypothesis results/RawModel/initial

        for ((i=0;i<$1;++i));
                do
                python neural_network/TrainNetwork.py RawModel data/train/Raw data/train/balanced.csv results/RawModel/errorlog.txt 25
                rm -r results/RawModel/hypothesis
                mkdir results/RawModel/hypothesis
                python neural_network/TestNetwork.py RawModel data/dev/Raw results/RawModel/hypothesis data/dev/table.csv results/RawModel $i
                reset RawModel $i;
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
